"""OOD axes BEYOND magnitude: does execution STRUCTURE generalize on trace-length
and nesting-depth the way it does on numeric magnitude?

The CodeDreamer finding (see scripts/neurosym_spike.py + docs) is that on
magnitude-OOD the net predicts transition STRUCTURE robustly (pc/sign/cmp/branch,
which-slot-changed) and only the arithmetic DIGIT payload collapses -- so
``em_digits_oracle`` (digits supplied by a perfect ALU, structure still the net's
job) stays high while ``em_learned`` (net decodes digits too) falls.

Here we re-run the SAME frozen checkpoint on two NON-magnitude OOD axes, holding
magnitude SMALL (<=5) so the only thing that moves is structure:

  * in-dist   -- the exact training spec (max_depth=2, num_stmts=5, max_const=5).
  * depth-OOD -- replace(spec, max_depth=4), kept only where realized nesting>=3.
  * length-OOD-- replace(spec, num_stmts=10, max_loop_count=5), kept only where
                 realized trace length exceeds the in-dist max.

If em_digits_oracle stays high on these axes too, structure generalizes off the
magnitude axis. If pc / cmp_result / em_digits_oracle degrade, the structure
prediction itself is hurt by depth/length -- a different and more interesting
failure than the digit-head magnitude wall.

CPU ONLY (a training job holds the GPU). Run:

    PYTHONPATH=. python scripts/analysis_ood_axes.py
"""
from __future__ import annotations

import numpy as np
import torch
from dataclasses import replace

from execwm.data.dataset import collect_examples
from execwm.eval.checkpoint import load_checkpoint
from execwm.eval.neurosym import field_breakdown
from execwm.substrate.generators import realized_metrics

CKPT = "artifacts/neurosym_model.pt"
N_EVAL = 300          # episodes per split
MAX_LEN = 48          # measurement window; long-trace episodes contribute 48 steps
SEED = 0

COLS = ["em_learned", "em_digits_oracle", "pc", "cmp_result", "written_digits"]


def _ranges(examples) -> dict:
    """Realized axis ranges over a set of examples, for honest reporting."""
    tl = [realized_metrics(e)["trace_len"] for e in examples]
    nd = [realized_metrics(e)["nesting_depth"] for e in examples]
    mg = [realized_metrics(e)["max_magnitude"] for e in examples]
    return {
        "trace_len": (min(tl), max(tl)),
        "nesting_depth": (min(nd), max(nd)),
        "max_magnitude": (min(mg), max(mg)),
    }


def _row(name: str, m: dict) -> str:
    g = lambda k: f"{m.get(k, float('nan')):.3f}"
    return ("| " + name + " | "
            + " | ".join(g(c) for c in COLS)
            + f" | {m.get('n', 0)} |")


def main() -> None:
    device = torch.device("cpu")   # CPU ONLY -- do not touch MPS/GPU
    print(f"[ood-axes] loading {CKPT} on {device} ...", flush=True)
    ck = load_checkpoint(CKPT, device=device)
    model, scodec, acodec = ck["model"], ck["scodec"], ck["acodec"]
    spec = ck["spec"]
    print(f"[ood-axes] training spec: max_depth={spec.max_depth} "
          f"num_stmts={spec.num_stmts} max_loop_count={spec.max_loop_count} "
          f"max_const={spec.max_const} max_input_val={spec.max_input_val}", flush=True)

    # --- in-distribution baseline: the exact training spec ---
    print("[ood-axes] collecting in-dist episodes ...", flush=True)
    indist_ex, indist_att = collect_examples(
        spec, N_EVAL, lambda ex: True, SEED + 99, scodec, acodec)
    indist_tls = [realized_metrics(e)["trace_len"] for e in indist_ex]
    # in-dist trace lengths are capped by the training spec's max_steps (128); use a
    # high percentile (not the lone cap-hitting outlier) as the OOD threshold.
    len_thr = int(np.percentile(indist_tls, 95))
    print(f"[ood-axes] in-dist trace_len: min={min(indist_tls)} "
          f"median={int(np.median(indist_tls))} p95={len_thr} "
          f"max={max(indist_tls)}  ->  length-OOD requires trace_len>{len_thr}",
          flush=True)

    # --- nesting-depth OOD: deeper than trained (kept only where realized>=3) ---
    # magnitude stays small (max_const/max_input_val unchanged at 5).
    print("[ood-axes] collecting depth-OOD episodes (max_depth=4, nesting>=3) ...",
          flush=True)
    depth_spec = replace(spec, max_depth=4)
    depth_pred = lambda ex: realized_metrics(ex)["nesting_depth"] >= 3
    depth_ex, depth_att = collect_examples(
        depth_spec, N_EVAL, depth_pred, SEED + 311, scodec, acodec,
        max_attempts=N_EVAL * 2000)

    # --- trace-length OOD: longer executions than any in-dist episode ---
    # magnitude stays small; only num_stmts/max_loop_count grow.
    print(f"[ood-axes] collecting length-OOD episodes "
          f"(num_stmts=10, max_loop_count=5, max_steps=256, trace_len>{len_thr}) ...",
          flush=True)
    # raise max_steps so executions can run LONGER than the training cap; magnitude
    # stays small (max_const/max_input_val unchanged). pc stays bounded by program
    # size (loops revisit pcs), so the checkpoint's codec (max_pc=128) still applies.
    length_spec = replace(spec, num_stmts=10, max_loop_count=5, max_steps=256)
    length_pred = lambda ex: realized_metrics(ex)["trace_len"] > len_thr
    length_ex, length_att = collect_examples(
        length_spec, N_EVAL, length_pred, SEED + 733, scodec, acodec,
        max_attempts=N_EVAL * 2000)

    # --- realized axis ranges (honesty about what each split actually is) ---
    print("\n# Realized axis ranges per split (min..max)")
    for name, ex in (("in-dist", indist_ex), ("depth-OOD", depth_ex),
                     ("length-OOD", length_ex)):
        r = _ranges(ex)
        print(f"  {name:<11} n={len(ex):>3}  trace_len={r['trace_len'][0]}..{r['trace_len'][1]}"
              f"  nesting={r['nesting_depth'][0]}..{r['nesting_depth'][1]}"
              f"  |val|={r['max_magnitude'][0]}..{r['max_magnitude'][1]}", flush=True)

    # --- field breakdowns (the frozen model read out on each split) ---
    print("\n[ood-axes] running field_breakdown on each split ...", flush=True)
    indist = field_breakdown(model, indist_ex, scodec, acodec, device, max_len=MAX_LEN)
    depth = field_breakdown(model, depth_ex, scodec, acodec, device, max_len=MAX_LEN)
    length = field_breakdown(model, length_ex, scodec, acodec, device, max_len=MAX_LEN)

    print("\n# OOD axes beyond magnitude -- structure vs digit readout\n")
    print("| split | em_learned | em_digits_oracle | pc | cmp_result | written_digits | n |")
    print("|---|---|---|---|---|---|---|")
    print(_row("in-dist", indist))
    print(_row("depth-OOD", depth))
    print(_row("length-OOD", length))

    print("\nReading the result:")
    print("  - em_digits_oracle = structure-only EM (perfect ALU supplies digits).")
    print("    If it stays high OOD, the net's STRUCTURE prediction generalizes off")
    print("    the magnitude axis. If it drops, depth/length hurt structure itself.")
    print("  - pc / cmp_result are pure structure signals; watch whether THEY degrade.")
    print("  - written_digits is the arithmetic payload (expected to be the weak one).")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
