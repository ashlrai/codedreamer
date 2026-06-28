"""M1 training loop + evaluation battery for the grounded latent world model.

Losses combined per step (see the plan):
  L_ground_next  decode the *predicted* next latent -> next symbolic state (main)
  L_ground_cur   decode the encoded current latent  -> current state (anchor)
  L_jepa         predicted next latent vs EMA-target-encoded next state + VICReg
  L_rollout      unroll dynamics K steps from the true start latent, decode each,
                 keep it exact (curriculum on K) -- the compounding-error test

Eval:
  in-distribution single-step exact-match + per-variable accuracy
  rollout-horizon curve: exact-match vs number of unrolled steps (the R1 spike)
"""

from __future__ import annotations

import random
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
from ..model.world_model import (GroundedLatentWM, ModelConfig, exact_match,
                                  grounding_loss, per_var_accuracy, vicreg)
from ..substrate.generators import GenSpec


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    steps: int = 600
    batch_size: int = 48
    lr: float = 3e-4
    weight_decay: float = 1e-2
    max_len: int = 24
    grad_clip: float = 1.0
    ema_momentum: float = 0.996
    w_ground_cur: float = 1.0
    w_ground_next: float = 1.0
    w_jepa: float = 0.5
    w_rollout: float = 1.0
    rollout_warmup: int = 100   # steps before rollout horizon starts growing
    rollout_grow_every: int = 80
    rollout_max_k: int = 8


def _targets_at(batch: dict, t_slice) -> dict[str, torch.Tensor]:
    """State label dict at a time slice, flattened over (B, L')."""
    out = {}
    for k in _STATE_KEYS:
        v = batch[f"s_{k}"][:, t_slice]
        out[k] = v.reshape(-1, *v.shape[2:])
    return out


def compute_losses(model: GroundedLatentWM, batch: dict, device, *,
                   rollout_k: int, tc: TrainConfig) -> tuple[torch.Tensor, dict]:
    valid = batch["valid"].to(device)                      # (B, L)
    B, L = valid.shape
    valid_flat = valid.reshape(B * L)

    # --- encode current states s[:, :L] ---
    cur_states = {k: batch[f"s_{k}"][:, :L].to(device) for k in _STATE_KEYS}
    cur_flat, _, _ = flatten_time(cur_states)
    z_cur = model.encode(cur_flat)                         # (B*L, d)

    # --- action embeddings ---
    act = {k: batch[f"a_{k}"][:, :L].to(device) for k in _ACTION_KEYS}
    act_flat, _, _ = flatten_time(act)
    a_emb = model.action(act_flat)                         # (B*L, d)

    # --- single-step predicted next latent ---
    z_next_pred = model.dynamics(z_cur, a_emb)             # (B*L, d)
    logits_next = model.heads(z_next_pred)
    logits_cur = model.heads(z_cur)

    tgt_next = _targets_at(batch, slice(1, L + 1))
    tgt_next = {k: v.to(device) for k, v in tgt_next.items()}
    tgt_cur = _targets_at(batch, slice(0, L))
    tgt_cur = {k: v.to(device) for k, v in tgt_cur.items()}

    def select(d):
        return {k: v[valid_flat] for k, v in d.items()}

    sel = valid_flat
    L_next = grounding_loss({k: v[sel] for k, v in logits_next.items()}, select(tgt_next))
    L_cur = grounding_loss({k: v[sel] for k, v in logits_cur.items()}, select(tgt_cur))

    # --- JEPA against EMA target encoder ---
    with torch.no_grad():
        next_states = {k: batch[f"s_{k}"][:, 1:L + 1].to(device) for k in _STATE_KEYS}
        next_flat, _, _ = flatten_time(next_states)
        z_tgt = model.target_encoder(next_flat)            # (B*L, d)
    zp, zt = z_next_pred[sel], z_tgt[sel]
    L_jepa = (1 - F.cosine_similarity(zp, zt, dim=-1)).mean() \
        + F.smooth_l1_loss(zp, zt) + 0.1 * vicreg(zp.reshape(-1, zp.shape[-1]))

    # --- rollout from the true start latent (curriculum K) ---
    z_cur_bL = z_cur.view(B, L, *z_cur.shape[1:])   # (B, L, S, d)
    a_emb_bL = a_emb.view(B, L, -1)
    z = z_cur_bL[:, 0]                                     # (B, d) = encode(s_0)
    L_roll = z.new_zeros(())
    roll_steps = 0
    for k in range(min(rollout_k, L)):
        z = model.dynamics(z, a_emb_bL[:, k])
        step_valid = valid[:, k]
        if step_valid.any():
            logits = model.heads(z)
            tgt = {kk: batch[f"s_{kk}"][:, k + 1].to(device)[step_valid]
                   for kk in _STATE_KEYS}
            L_roll = L_roll + grounding_loss(
                {kk: vv[step_valid] for kk, vv in logits.items()}, tgt)
            roll_steps += 1
    if roll_steps:
        L_roll = L_roll / roll_steps

    total = (tc.w_ground_next * L_next + tc.w_ground_cur * L_cur
             + tc.w_jepa * L_jepa + tc.w_rollout * L_roll)

    with torch.no_grad():
        em = exact_match({k: v[sel] for k, v in logits_next.items()},
                         select(tgt_next)).float().mean()
        pv = per_var_accuracy({k: v[sel] for k, v in logits_next.items()},
                              select(tgt_next))
    metrics = {"loss": total.item(), "L_next": L_next.item(), "L_cur": L_cur.item(),
               "L_jepa": L_jepa.item(), "L_roll": float(L_roll.detach()),
               "step_em": em.item(), "per_var": pv.item(), "K": min(rollout_k, L)}
    return total, metrics


@torch.no_grad()
def evaluate(model: GroundedLatentWM, loader: DataLoader, device) -> dict:
    model.eval()
    n = em_sum = pv_sum = pv_n = 0
    for batch in loader:
        valid = batch["valid"].to(device)
        B, L = valid.shape
        valid_flat = valid.reshape(B * L)
        cur = {k: batch[f"s_{k}"][:, :L].to(device) for k in _STATE_KEYS}
        cur_flat, _, _ = flatten_time(cur)
        z = model.encode(cur_flat)
        act = {k: batch[f"a_{k}"][:, :L].to(device) for k in _ACTION_KEYS}
        act_flat, _, _ = flatten_time(act)
        zn = model.dynamics(z, model.action(act_flat))
        logits = model.heads(zn)
        tgt = {k: batch[f"s_{k}"][:, 1:L + 1].to(device).reshape(-1, *batch[f"s_{k}"].shape[2:])
               for k in _STATE_KEYS}
        sel = valid_flat
        em = exact_match({k: v[sel] for k, v in logits.items()},
                         {k: v[sel] for k, v in tgt.items()})
        em_sum += em.float().sum().item()
        n += int(sel.sum().item())
        pv_sum += per_var_accuracy({k: v[sel] for k, v in logits.items()},
                                   {k: v[sel] for k, v in tgt.items()}).item() * int(sel.sum())
        pv_n += int(sel.sum())
    model.train()
    return {"step_exact_match": em_sum / max(n, 1),
            "per_var_acc": pv_sum / max(pv_n, 1), "n": n}


@torch.no_grad()
def rollout_horizon(model: GroundedLatentWM, loader: DataLoader, device,
                    max_k: int = 16) -> list[float]:
    """Exact-match of the decoded state after k pure-latent rollout steps from the
    true start latent, for k=1..max_k. The compounding-error / R1 curve."""
    model.eval()
    hit = [0] * max_k
    tot = [0] * max_k
    for batch in loader:
        valid = batch["valid"].to(device)
        B, L = valid.shape
        s0 = {k: batch[f"s_{k}"][:, 0].to(device) for k in _STATE_KEYS}
        z = model.encode(s0)
        for k in range(min(max_k, L)):
            act_k = slice_action(batch, k, device)
            z = model.dynamics(z, model.action(act_k))
            step_valid = valid[:, k]
            if not step_valid.any():
                continue
            logits = model.heads(z)
            tgt = {kk: batch[f"s_{kk}"][:, k + 1].to(device) for kk in _STATE_KEYS}
            em = exact_match({kk: vv[step_valid] for kk, vv in logits.items()},
                             {kk: vv[step_valid] for kk, vv in tgt.items()})
            hit[k] += int(em.sum().item())
            tot[k] += int(step_valid.sum().item())
    model.train()
    return [hit[k] / tot[k] if tot[k] else float("nan") for k in range(max_k)]


def build(spec: GenSpec, codec_cfg: CodecConfig, **model_kw):
    cfg = spec.config()
    scodec = StateCodec(cfg, codec_cfg)
    acodec = ActionCodec(cfg, codec_cfg)
    mcfg = ModelConfig.from_codec(len(cfg.reg_names), scodec.num_cells,
                                  cfg.num_lists, codec_cfg, **model_kw)
    model = GroundedLatentWM(mcfg)
    return model, scodec, acodec


def train(spec: GenSpec | None = None, codec_cfg: CodecConfig | None = None,
          tc: TrainConfig | None = None, n_train: int = 1500, n_eval: int = 300,
          seed: int = 0, device=None, log_every: int = 50, **model_kw) -> dict:
    spec = spec or GenSpec(num_vars=4, max_depth=2, num_stmts=5,
                           max_const=5, max_input_val=5, max_loop_count=3)
    codec_cfg = codec_cfg or CodecConfig(max_digits=6, base=10, max_pc=256)
    tc = tc or TrainConfig()
    device = device or pick_device()

    model, scodec, acodec = build(spec, codec_cfg, **model_kw)
    model.to(device)

    import time as _time
    pred = lambda ex: True  # in-distribution training data
    t0 = _time.perf_counter()
    train_ex, _ = collect_examples(spec, n_train, pred, seed, scodec, acodec)
    eval_ex, _ = collect_examples(spec, n_eval, pred, seed + 99, scodec, acodec)
    print(f"[m1] collected {len(train_ex)}+{len(eval_ex)} examples "
          f"in {_time.perf_counter()-t0:.1f}s; encoding episodes...", flush=True)
    t0 = _time.perf_counter()
    train_ds = EpisodeDataset(train_ex, scodec, acodec, max_len=tc.max_len)
    eval_ds = EpisodeDataset(eval_ex, scodec, acodec, max_len=tc.max_len)
    print(f"[m1] encoded episodes in {_time.perf_counter()-t0:.1f}s", flush=True)
    train_loader = DataLoader(train_ds, batch_size=tc.batch_size, shuffle=True,
                              collate_fn=collate_episodes, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=tc.batch_size, shuffle=False,
                             collate_fn=collate_episodes)

    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[m1] device={device} params={n_params/1e6:.2f}M "
          f"train_eps={len(train_ds)} eval_eps={len(eval_ds)}")

    step = 0
    data_iter = iter(train_loader)
    while step < tc.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        # curriculum rollout horizon
        if step < tc.rollout_warmup:
            K = 1
        else:
            K = min(tc.rollout_max_k,
                    1 + (step - tc.rollout_warmup) // tc.rollout_grow_every)
        loss, metrics = compute_losses(model, batch, device, rollout_k=K, tc=tc)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        model.update_target(tc.ema_momentum)
        step += 1
        if step % log_every == 0 or step == 1:
            print(f"[m1] step {step:4d}  loss {metrics['loss']:.3f}  "
                  f"next {metrics['L_next']:.3f}  roll {metrics['L_roll']:.3f}  "
                  f"jepa {metrics['L_jepa']:.3f}  step_em {metrics['step_em']:.3f}  "
                  f"per_var {metrics['per_var']:.3f}  K={metrics['K']}")

    ev = evaluate(model, eval_loader, device)
    horizon = rollout_horizon(model, eval_loader, device, max_k=tc.max_len)
    print(f"[m1] EVAL single-step exact-match {ev['step_exact_match']:.4f}  "
          f"per-var {ev['per_var_acc']:.4f}  (n={ev['n']})")
    hs = "  ".join(f"k{ k+1}:{v:.2f}" for k, v in enumerate(horizon[:12]))
    print(f"[m1] ROLLOUT-HORIZON exact-match  {hs}")
    return {"model": model, "eval": ev, "rollout_horizon": horizon,
            "scodec": scodec, "acodec": acodec, "device": device}


if __name__ == "__main__":
    train()
