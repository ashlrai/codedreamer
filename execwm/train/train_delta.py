"""M1.6 training/eval for the copy-vs-compute (delta) world model.

Reuses the M1 data pipeline (episode dataset + padded batches) and the slotted
encoder/dynamics; swaps the per-step objective for the delta heads: a per-slot
change gate + a value supervised only where the slot changed. Single-step and
rollout exact-match are computed by *composing* (copy unchanged slots, write the
gated value). Rollout carries the model's own predicted state forward as the copy
source, so it is a true autoregressive latent rollout.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.action_codec import ActionCodec
from ..data.dataset import collect_examples
from ..data.state_codec import CodecConfig, StateCodec
from ..data.torch_data import (EpisodeDataset, _ACTION_KEYS, _STATE_KEYS,
                               collate_episodes, flatten_time, slice_action,
                               slice_state)
from ..model.delta import (DeltaWM, compose_next, delta_loss, exact_match_labels,
                            gate_accuracy)
from ..model.world_model import ModelConfig, valued_mask
from ..substrate.generators import GenSpec
from .train_m1 import TrainConfig, pick_device


def _labels(batch: dict, sl, device) -> dict:
    out = {}
    for k in _STATE_KEYS:
        v = batch[f"s_{k}"][:, sl]
        out[k] = v.reshape(-1, *v.shape[2:]).to(device)
    return out


@torch.no_grad()
def _per_var_labels(pred: dict, tgt: dict) -> float:
    mask = valued_mask(tgt["reg_type"])
    correct = ((pred["reg_type"] == tgt["reg_type"])
               & (pred["reg_sign"] == tgt["reg_sign"])
               & (pred["reg_digits"] == tgt["reg_digits"]).all(-1)) & mask
    return (correct.sum().float() / mask.sum().clamp_min(1).float()).item()


def compute_delta_losses(model: DeltaWM, batch: dict, device, *, rollout_k: int,
                         tc: TrainConfig):
    cfg = model.cfg
    valid = batch["valid"].to(device)
    B, L = valid.shape
    sel = valid.reshape(B * L)

    cur_states = {k: batch[f"s_{k}"][:, :L].to(device) for k in _STATE_KEYS}
    cur_flat, _, _ = flatten_time(cur_states)
    z_cur = model.encode(cur_flat)
    act = {k: batch[f"a_{k}"][:, :L].to(device) for k in _ACTION_KEYS}
    act_flat, _, _ = flatten_time(act)
    a_emb = model.action(act_flat)
    z_next = model.dynamics(z_cur, a_emb)
    gate, value = model.delta(z_next)

    cur = _labels(batch, slice(0, L), device)
    nxt = _labels(batch, slice(1, L + 1), device)

    def pick(d):
        return {k: v[sel] for k, v in d.items()}

    g_sel = gate[sel]
    v_sel = {k: v[sel] for k, v in value.items()}
    L_step, gate_l, val_l = delta_loss(g_sel, v_sel, pick(cur), pick(nxt), cfg)

    # light JEPA to keep the latent informative
    with torch.no_grad():
        nxt_states = {k: batch[f"s_{k}"][:, 1:L + 1].to(device) for k in _STATE_KEYS}
        nxt_flat, _, _ = flatten_time(nxt_states)
        z_tgt = model.target_encoder(nxt_flat)
    zp, zt = z_next[sel], z_tgt[sel]
    L_jepa = (1 - F.cosine_similarity(zp, zt, dim=-1)).mean() + F.smooth_l1_loss(zp, zt)

    # rollout: unroll latent, supervise gate+value at each horizon (teacher copy)
    z_bL = z_cur.view(B, L, *z_cur.shape[1:])
    a_bL = a_emb.view(B, L, -1)
    z = z_bL[:, 0]
    L_roll = z.new_zeros(())
    n_roll = 0
    for k in range(min(rollout_k, L)):
        z = model.dynamics(z, a_bL[:, k])
        sv = valid[:, k]
        if sv.any():
            gk, vk = model.delta(z)
            ck = {kk: batch[f"s_{kk}"][:, k].to(device)[sv] for kk in _STATE_KEYS}
            nk = {kk: batch[f"s_{kk}"][:, k + 1].to(device)[sv] for kk in _STATE_KEYS}
            lk, _, _ = delta_loss(gk[sv], {kk: vv[sv] for kk, vv in vk.items()}, ck, nk, cfg)
            L_roll = L_roll + lk
            n_roll += 1
    if n_roll:
        L_roll = L_roll / n_roll

    total = L_step + tc.w_rollout * L_roll + tc.w_jepa * L_jepa

    with torch.no_grad():
        pred = compose_next(g_sel, v_sel, pick(cur), cfg)
        em = exact_match_labels(pred, pick(nxt)).float().mean().item()
        pv = _per_var_labels(pred, pick(nxt))
        gacc, grec = gate_accuracy(g_sel, pick(cur), pick(nxt), cfg)
    metrics = {"loss": total.item(), "L_step": L_step.item(), "gate": gate_l.item(),
               "val": val_l.item(), "L_roll": float(L_roll.detach()),
               "step_em": em, "per_var": pv, "gate_acc": gacc, "gate_recall": grec,
               "K": min(rollout_k, L)}
    return total, metrics


@torch.no_grad()
def evaluate_delta(model: DeltaWM, loader: DataLoader, device) -> dict:
    model.eval()
    n = em_sum = pv_sum = pv_n = 0
    for batch in loader:
        valid = batch["valid"].to(device)
        B, L = valid.shape
        sel = valid.reshape(B * L)
        cur_states = {k: batch[f"s_{k}"][:, :L].to(device) for k in _STATE_KEYS}
        cur_flat, _, _ = flatten_time(cur_states)
        z = model.encode(cur_flat)
        act = {k: batch[f"a_{k}"][:, :L].to(device) for k in _ACTION_KEYS}
        act_flat, _, _ = flatten_time(act)
        gate, value = model.delta(model.dynamics(z, model.action(act_flat)))
        cur = _labels(batch, slice(0, L), device)
        nxt = _labels(batch, slice(1, L + 1), device)
        g, v = gate[sel], {k: vv[sel] for k, vv in value.items()}
        cur, nxt = {k: vv[sel] for k, vv in cur.items()}, {k: vv[sel] for k, vv in nxt.items()}
        pred = compose_next(g, v, cur, model.cfg)
        em = exact_match_labels(pred, nxt)
        cnt = int(sel.sum())
        em_sum += int(em.sum()); n += cnt
        pv_sum += _per_var_labels(pred, nxt) * cnt; pv_n += cnt
    model.train()
    return {"step_exact_match": em_sum / max(n, 1), "per_var_acc": pv_sum / max(pv_n, 1), "n": n}


@torch.no_grad()
def rollout_horizon_delta(model: DeltaWM, loader: DataLoader, device, max_k: int = 16):
    """Autoregressive latent rollout: carry the model's own predicted state forward
    as the copy source. exact-match vs the true state after k steps."""
    model.eval()
    hit = [0] * max_k
    tot = [0] * max_k
    for batch in loader:
        valid = batch["valid"].to(device)
        B, L = valid.shape
        cur = {k: batch[f"s_{k}"][:, 0].to(device) for k in _STATE_KEYS}
        z = model.encode(cur)
        for k in range(min(max_k, L)):
            z = model.dynamics(z, model.action(slice_action(batch, k, device)))
            gate, value = model.delta(z)
            cur = compose_next(gate, value, cur, model.cfg)  # carry predicted state
            sv = valid[:, k]
            if not sv.any():
                continue
            tgt = {kk: batch[f"s_{kk}"][:, k + 1].to(device) for kk in _STATE_KEYS}
            em = exact_match_labels({kk: vv[sv] for kk, vv in cur.items()},
                                    {kk: vv[sv] for kk, vv in tgt.items()})
            hit[k] += int(em.sum()); tot[k] += int(sv.sum())
    model.train()
    return [hit[k] / tot[k] if tot[k] else float("nan") for k in range(max_k)]


def build_delta(spec: GenSpec, codec_cfg: CodecConfig, **model_kw):
    cfg = spec.config()
    scodec = StateCodec(cfg, codec_cfg)
    acodec = ActionCodec(cfg, codec_cfg)
    mcfg = ModelConfig.from_codec(len(cfg.reg_names), scodec.num_cells,
                                  cfg.num_lists, codec_cfg, **model_kw)
    return DeltaWM(mcfg), scodec, acodec


def train_delta(spec=None, codec_cfg=None, tc=None, n_train=4000, n_eval=600,
                seed=0, device=None, log_every=100, **model_kw) -> dict:
    spec = spec or GenSpec(num_vars=4, max_depth=2, num_stmts=5,
                           max_const=5, max_input_val=5, max_loop_count=3)
    codec_cfg = codec_cfg or CodecConfig(max_digits=6, base=10, max_pc=256)
    tc = tc or TrainConfig()
    device = device or pick_device()

    model, scodec, acodec = build_delta(spec, codec_cfg, **model_kw)
    model.to(device)
    pred = lambda ex: True
    import time as _t
    t0 = _t.perf_counter()
    train_ex, _ = collect_examples(spec, n_train, pred, seed, scodec, acodec)
    eval_ex, _ = collect_examples(spec, n_eval, pred, seed + 99, scodec, acodec)
    print(f"[delta] collected {len(train_ex)}+{len(eval_ex)} in {_t.perf_counter()-t0:.1f}s", flush=True)
    train_ds = EpisodeDataset(train_ex, scodec, acodec, max_len=tc.max_len)
    eval_ds = EpisodeDataset(eval_ex, scodec, acodec, max_len=tc.max_len)
    train_loader = DataLoader(train_ds, batch_size=tc.batch_size, shuffle=True,
                              collate_fn=collate_episodes, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=tc.batch_size, shuffle=False,
                             collate_fn=collate_episodes)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[delta] device={device} params={n_params/1e6:.2f}M", flush=True)

    step = 0
    it = iter(train_loader)
    while step < tc.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader); batch = next(it)
        K = 1 if step < tc.rollout_warmup else min(
            tc.rollout_max_k, 1 + (step - tc.rollout_warmup) // tc.rollout_grow_every)
        loss, m = compute_delta_losses(model, batch, device, rollout_k=K, tc=tc)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step(); model.update_target(tc.ema_momentum); step += 1
        if step % log_every == 0 or step == 1:
            print(f"[delta] step {step:4d} loss {m['loss']:.3f} gate {m['gate']:.3f} "
                  f"val {m['val']:.3f} roll {m['L_roll']:.3f} step_em {m['step_em']:.3f} "
                  f"per_var {m['per_var']:.3f} gate_acc {m['gate_acc']:.3f} K={m['K']}", flush=True)

    ev = evaluate_delta(model, eval_loader, device)
    horizon = rollout_horizon_delta(model, eval_loader, device, max_k=tc.max_len)
    print(f"[delta] EVAL single-step exact-match {ev['step_exact_match']:.4f} "
          f"per-var {ev['per_var_acc']:.4f} (n={ev['n']})", flush=True)
    print("[delta] ROLLOUT-HORIZON " + "  ".join(
        f"k{k+1}:{v:.2f}" for k, v in enumerate(horizon[:12])), flush=True)
    return {"model": model, "eval": ev, "rollout_horizon": horizon,
            "scodec": scodec, "acodec": acodec}


if __name__ == "__main__":
    train_delta()
