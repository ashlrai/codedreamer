"""Magnitude-curriculum training for the carry-aware arithmetic head (ArithWM).

Identical objective and model to ``train_arith`` (grounded decode at t+1 + JEPA +
rollout curriculum, digits teacher-forced through the carry-aware head), with one
change: the TRAINING data is resampled at each magnitude-curriculum stage
boundary, ramping operand size small->large so carries are learned progressively
(Abacus / Learning-to-Execute). The pool is rebuilt only when the stage changes,
not every step.

EVALUATION is ALWAYS at the full target magnitude (``base_spec``), so the reported
single-step exact-match measures real generalization to the hard distribution —
never the current easy stage. The active stage's magnitude is logged alongside
loss / exact-match.
"""

from __future__ import annotations

import time

import torch
from torch.utils.data import DataLoader

from ..data.state_codec import CodecConfig
from ..data.torch_data import EpisodeDataset, collate_episodes
from ..substrate.generators import GenSpec
from .curriculum import (MagnitudeCurriculum, spec_for_step, stage_examples,
                         stage_index_for_step)
from .train_arith import build_arith, compute_losses_arith
from .train_m1 import TrainConfig, evaluate, pick_device, rollout_horizon


def _loader_for_stage(spec: GenSpec, n_train: int, seed: int, scodec, acodec,
                      tc: TrainConfig) -> DataLoader:
    """Resample the training pool at ``spec``'s magnitude and wrap it in a
    shuffling DataLoader (mirrors ``train_arith``'s loader construction)."""
    ex = stage_examples(spec, n_train, seed, scodec, acodec)
    ds = EpisodeDataset(ex, scodec, acodec, max_len=tc.max_len)
    return DataLoader(ds, batch_size=tc.batch_size, shuffle=True,
                      collate_fn=collate_episodes, drop_last=True)


def train_arith_curriculum(base_spec: GenSpec | None = None,
                           codec_cfg: CodecConfig | None = None,
                           tc: TrainConfig | None = None,
                           curriculum: MagnitudeCurriculum | None = None,
                           *, n_train: int = 4000, n_eval: int = 600, seed: int = 0,
                           device=None, log_every: int = 100, **model_kw) -> dict:
    """Train ArithWM with a magnitude curriculum; evaluate at full target
    magnitude. Returns the same contract as ``train_arith`` plus ``curriculum``.

    Args mirror ``train_arith``; ``base_spec`` is the FULL TARGET magnitude (the
    hardest stage and the eval distribution). ``curriculum`` defaults to a
    3-stage linear ramp from magnitude 1 up to ``base_spec``.
    """
    base_spec = base_spec or GenSpec(num_vars=4, max_depth=2, num_stmts=5,
                                     max_const=5, max_input_val=5, max_loop_count=3)
    codec_cfg = codec_cfg or CodecConfig(max_digits=6, base=10, max_pc=256)
    tc = tc or TrainConfig()
    device = device or pick_device()
    if curriculum is None:
        from .curriculum import linear_magnitude_curriculum
        curriculum = linear_magnitude_curriculum(base_spec, n_stages=3, start_max=1)

    # Magnitude does not change the VM register shape, so the model and codecs
    # are built once from the (target) base spec and reused across all stages.
    model, scodec, acodec = build_arith(base_spec, codec_cfg, **model_kw)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[curric] device={device} params={n_params/1e6:.2f}M "
          f"stages={curriculum.n_stages}", flush=True)

    # Eval pool: FULL TARGET magnitude (base_spec) — fixed across the whole run.
    t0 = time.perf_counter()
    eval_ex = stage_examples(base_spec, n_eval, seed + 99, scodec, acodec)
    eval_ds = EpisodeDataset(eval_ex, scodec, acodec, max_len=tc.max_len)
    eval_loader = DataLoader(eval_ds, batch_size=tc.batch_size, shuffle=False,
                             collate_fn=collate_episodes)
    print(f"[curric] eval pool {len(eval_ex)} @ target "
          f"max_const={base_spec.max_const} max_input_val={base_spec.max_input_val} "
          f"in {time.perf_counter()-t0:.1f}s", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)

    cur_stage = -1
    loader: DataLoader | None = None
    it = None
    step = 0
    while step < tc.steps:
        stage_idx = stage_index_for_step(curriculum, step, tc.steps)
        if stage_idx != cur_stage:
            cur_stage = stage_idx
            spec = spec_for_step(curriculum, base_spec, step, tc.steps)
            t0 = time.perf_counter()
            loader = _loader_for_stage(spec, n_train, seed + 1000 * stage_idx,
                                       scodec, acodec, tc)
            it = iter(loader)
            print(f"[curric] step {step:4d} -> stage {stage_idx} "
                  f"max_const={spec.max_const} max_input_val={spec.max_input_val} "
                  f"(resampled {len(loader.dataset)} eps in {time.perf_counter()-t0:.1f}s)",
                  flush=True)

        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)

        K = 1 if step < tc.rollout_warmup else min(
            tc.rollout_max_k, 1 + (step - tc.rollout_warmup) // tc.rollout_grow_every)
        loss, m = compute_losses_arith(model, batch, device, rollout_k=K, tc=tc)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step(); model.update_target(tc.ema_momentum); step += 1

        if step % log_every == 0 or step == 1:
            st = curriculum.stages[cur_stage]
            print(f"[curric] step {step:4d} stage {cur_stage} "
                  f"mag(c={st.max_const},v={st.max_input_val}) "
                  f"loss {m['loss']:.3f} next {m['L_next']:.3f} roll {m['L_roll']:.3f} "
                  f"step_em {m['step_em']:.3f} per_var {m['per_var']:.3f} K={m['K']}",
                  flush=True)

    # Evaluation: ALWAYS at full target magnitude.
    ev = evaluate(model, eval_loader, device)
    horizon = rollout_horizon(model, eval_loader, device, max_k=tc.max_len)
    print(f"[curric] EVAL@target single-step exact-match {ev['step_exact_match']:.4f} "
          f"per-var {ev['per_var_acc']:.4f} (n={ev['n']})", flush=True)
    print("[curric] ROLLOUT-HORIZON " + "  ".join(
        f"k{k+1}:{v:.2f}" for k, v in enumerate(horizon[:12])), flush=True)
    return {"model": model, "eval": ev, "rollout_horizon": horizon,
            "scodec": scodec, "acodec": acodec, "device": device,
            "curriculum": curriculum}


if __name__ == "__main__":
    train_arith_curriculum()
