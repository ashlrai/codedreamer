"""Multi-step payoff: the neurosymbolic executor runs whole programs (net drives
control flow, ALU computes values) at in-distribution AND magnitude-OOD scale.

Contrast: a pure-net executor (values from the digit head) has single-step OOD
exact-match 0.000 (see `scripts/neurosym_spike.py`), so its full-trajectory success
at OOD magnitude is 0. This measures how far the neurosymbolic split gets instead.

    PYTHONPATH=. python scripts/neurosym_exec_eval.py [--n 300]
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import torch

from execwm.data.dataset import collect_examples
from execwm.eval.checkpoint import load_checkpoint
from execwm.eval.neurosym_exec import evaluate_executor


def _row(name: str, e: dict) -> str:
    g = lambda k: f"{e.get(k, float('nan')):.3f}"
    return (f"| {name} | {g('full_trajectory_success')} | {g('per_step_state_exact')} "
            f"| {g('control_accuracy')} | {e.get('mean_exact_horizon', float('nan')):.1f} "
            f"| {e.get('n_programs', 0)} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/neurosym_model.pt")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cpu")  # batch-1 per-step; CPU avoids MPS launch overhead
    ck = load_checkpoint(args.ckpt, device=device)
    model, scodec, acodec, spec = ck["model"], ck["scodec"], ck["acodec"], ck["spec"]
    model.to(device).eval()
    ood_spec = replace(spec, max_const=400, max_input_val=400)

    indist, _ = collect_examples(spec, args.n, lambda e: True, args.seed + 99, scodec, acodec)
    ood, att = collect_examples(ood_spec, args.n, lambda e: True, args.seed + 777, scodec, acodec)
    print(f"[exec] {len(indist)} in-dist / {len(ood)} OOD programs (OOD from {att} attempts)",
          flush=True)

    ei = evaluate_executor(model, scodec, acodec, indist, device)
    eo = evaluate_executor(model, scodec, acodec, ood, device)

    print("\n# Neurosymbolic executor — net-control + ALU-values, whole programs\n")
    print("| split | full-traj success | per-step exact | control acc | mean horizon | n |")
    print("|---|---|---|---|---|---|")
    print(_row("in-distribution (val<=30)", ei))
    print(_row("magnitude-OOD (val~300-800)", eo))
    print("\n(pure-net executor OOD full-trajectory success = 0.000 — single-step EM is 0.)")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
