"""M1.6b training for the carry-aware arithmetic-head world model (ArithWM).

Same objective as M1 (grounded decode at t and t+1 + JEPA + curriculum rollout),
but the digit fields are produced by the autoregressive carry-aware head, which is
*teacher-forced* during training (we feed the target digits) and greedily decoded
at eval. Because ``ArithWM.heads(z)`` greedily decodes when called with no teacher,
the M1 ``evaluate`` and ``rollout_horizon`` work unchanged.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.action_codec import ActionCodec
from ..data.dataset import collect_examples
from ..data.state_codec import CodecConfig, StateCodec
from ..data.torch_data import (EpisodeDataset, _ACTION_KEYS, _STATE_KEYS,
                               collate_episodes, flatten_time)
from ..model.arith import ArithWM
from ..model.world_model import (ModelConfig, exact_match, grounding_loss,
                                  per_var_accuracy, vicreg)
from ..substrate.generators import GenSpec
from .train_m1 import TrainConfig, evaluate, pick_device, rollout_horizon


def _flat_targets(batch: dict, sl, device) -> dict:
    out = {}
    for k in _STATE_KEYS:
        v = batch[f"s_{k}"][:, sl]
        out[k] = v.reshape(-1, *v.shape[2:]).to(device)
    return out


def _teacher(tgt: dict) -> dict:
    return {"reg_digits": tgt["reg_digits"], "heap_digits": tgt["heap_digits"]}


def compute_losses_arith(model: ArithWM, batch: dict, device, *, rollout_k: int,
                         tc: TrainConfig):
    valid = batch["valid"].to(device)
    B, L = valid.shape
    sel = valid.reshape(B * L)

    cur = {k: batch[f"s_{k}"][:, :L].to(device) for k in _STATE_KEYS}
    cur_flat, _, _ = flatten_time(cur)
    z_cur = model.encode(cur_flat)
    act = {k: batch[f"a_{k}"][:, :L].to(device) for k in _ACTION_KEYS}
    act_flat, _, _ = flatten_time(act)
    a_emb = model.action(act_flat)
    z_next = model.dynamics(z_cur, a_emb)

    tgt_next = _flat_targets(batch, slice(1, L + 1), device)
    logits_next = model.heads(z_next, teacher=_teacher(tgt_next))

    def pick(d):
        return {k: v[sel] for k, v in d.items()}

    L_next = grounding_loss(pick(logits_next), pick(tgt_next))

    with torch.no_grad():
        nxt_states = {k: batch[f"s_{k}"][:, 1:L + 1].to(device) for k in _STATE_KEYS}
        nxt_flat, _, _ = flatten_time(nxt_states)
        z_tgt = model.target_encoder(nxt_flat)
    zp, zt = z_next[sel], z_tgt[sel]
    L_jepa = (1 - F.cosine_similarity(zp, zt, dim=-1)).mean() + F.smooth_l1_loss(zp, zt) \
        + 0.1 * vicreg(zp.reshape(-1, zp.shape[-1]))

    z_bL = z_cur.view(B, L, *z_cur.shape[1:])
    a_bL = a_emb.view(B, L, -1)
    z = z_bL[:, 0]
    L_roll = z.new_zeros(())
    n_roll = 0
    for k in range(min(rollout_k, L)):
        z = model.dynamics(z, a_bL[:, k])
        sv = valid[:, k]
        if sv.any():
            tk = {kk: batch[f"s_{kk}"][:, k + 1].to(device)[sv] for kk in _STATE_KEYS}
            lk = model.heads(z[sv], teacher=_teacher(tk))
            L_roll = L_roll + grounding_loss(lk, tk)
            n_roll += 1
    if n_roll:
        L_roll = L_roll / n_roll

    total = L_next + tc.w_jepa * L_jepa + tc.w_rollout * L_roll

    with torch.no_grad():
        # teacher-forced metric (cheap; the honest greedy number comes from evaluate())
        em = exact_match(pick(logits_next), pick(tgt_next)).float().mean().item()
        pv = per_var_accuracy(pick(logits_next), pick(tgt_next)).item()
    return total, {"loss": total.item(), "L_next": L_next.item(), "L_roll": float(L_roll.detach()),
                   "L_jepa": L_jepa.item(), "step_em": em, "per_var": pv, "K": min(rollout_k, L)}


def build_arith(spec: GenSpec, codec_cfg: CodecConfig, **model_kw):
    cfg = spec.config()
    scodec = StateCodec(cfg, codec_cfg)
    acodec = ActionCodec(cfg, codec_cfg)
    mcfg = ModelConfig.from_codec(len(cfg.reg_names), scodec.num_cells,
                                  cfg.num_lists, codec_cfg, **model_kw)
    return ArithWM(mcfg), scodec, acodec


def train_arith(spec=None, codec_cfg=None, tc=None, n_train=4000, n_eval=600,
                seed=0, device=None, log_every=100, **model_kw) -> dict:
    spec = spec or GenSpec(num_vars=4, max_depth=2, num_stmts=5,
                           max_const=5, max_input_val=5, max_loop_count=3)
    codec_cfg = codec_cfg or CodecConfig(max_digits=6, base=10, max_pc=256)
    tc = tc or TrainConfig()
    device = device or pick_device()

    model, scodec, acodec = build_arith(spec, codec_cfg, **model_kw)
    model.to(device)
    t0 = time.perf_counter()
    train_ex, _ = collect_examples(spec, n_train, lambda e: True, seed, scodec, acodec)
    eval_ex, _ = collect_examples(spec, n_eval, lambda e: True, seed + 99, scodec, acodec)
    print(f"[arith] collected {len(train_ex)}+{len(eval_ex)} in {time.perf_counter()-t0:.1f}s", flush=True)
    train_ds = EpisodeDataset(train_ex, scodec, acodec, max_len=tc.max_len)
    eval_ds = EpisodeDataset(eval_ex, scodec, acodec, max_len=tc.max_len)
    train_loader = DataLoader(train_ds, batch_size=tc.batch_size, shuffle=True,
                              collate_fn=collate_episodes, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=tc.batch_size, shuffle=False,
                             collate_fn=collate_episodes)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[arith] device={device} params={n_params/1e6:.2f}M", flush=True)

    step = 0
    it = iter(train_loader)
    while step < tc.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader); batch = next(it)
        K = 1 if step < tc.rollout_warmup else min(
            tc.rollout_max_k, 1 + (step - tc.rollout_warmup) // tc.rollout_grow_every)
        loss, m = compute_losses_arith(model, batch, device, rollout_k=K, tc=tc)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step(); model.update_target(tc.ema_momentum); step += 1
        if step % log_every == 0 or step == 1:
            print(f"[arith] step {step:4d} loss {m['loss']:.3f} next {m['L_next']:.3f} "
                  f"roll {m['L_roll']:.3f} step_em {m['step_em']:.3f} per_var {m['per_var']:.3f} "
                  f"K={m['K']}", flush=True)

    ev = evaluate(model, eval_loader, device)
    horizon = rollout_horizon(model, eval_loader, device, max_k=tc.max_len)
    print(f"[arith] EVAL single-step exact-match {ev['step_exact_match']:.4f} "
          f"per-var {ev['per_var_acc']:.4f} (n={ev['n']})", flush=True)
    print("[arith] ROLLOUT-HORIZON " + "  ".join(
        f"k{k+1}:{v:.2f}" for k, v in enumerate(horizon[:12])), flush=True)
    return {"model": model, "eval": ev, "rollout_horizon": horizon,
            "scodec": scodec, "acodec": acodec}


if __name__ == "__main__":
    train_arith()
