"""Finish the headline comparison: the latent half already ran and saved
artifacts/execwm_bench_report.json + latent_model.pt. This trains the matched
token-space baseline and grades BOTH models on the SAME held-out subset so the
token model's slow autoregressive greedy decode stays tractable, then writes the
latent-vs-token comparison.

    PYTHONPATH=. caffeinate -i python scripts/compare_token.py [--n-eval 150] [--steps 800]
"""
from __future__ import annotations

import argparse

import torch

from execwm.data.dataset import collect_examples
from execwm.data.state_codec import CodecConfig
from execwm.eval.checkpoint import load_checkpoint
from execwm.eval.execwm_bench import core_metrics
from execwm.eval.report import BenchReport, compare_reports
from execwm.substrate.generators import GenSpec
from execwm.train.train_m1 import TrainConfig, pick_device
import os

from execwm.train.train_token import (evaluate_token_baseline, load_token_baseline,
                                       save_token_baseline, train_token_baseline)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-eval", type=int, default=150,
                    help="examples to grade BOTH models on (token greedy decode is slow)")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--eval-batch", type=int, default=8,
                    help="small: autoregressive greedy decode is memory-heavy on MPS")
    ap.add_argument("--reuse-token", action="store_true",
                    help="load artifacts/token_model.pt instead of retraining")
    args = ap.parse_args()

    device = pick_device()              # latent eval (cheap) stays on MPS/GPU
    token_device = torch.device("cpu")  # token baseline train+eval on CPU (see below)
    # must match scripts/run_execwm_bench.py exactly so the latent checkpoint + its
    # report were trained/graded on the same distribution.
    spec = GenSpec(num_vars=4, max_depth=2, num_stmts=5,
                   max_const=5, max_input_val=5, max_loop_count=3)
    codec = CodecConfig(max_digits=6, base=10, max_pc=256)
    tc = TrainConfig(steps=args.steps, batch_size=48, max_len=18,
                     rollout_warmup=max(1, args.steps // 3),
                     rollout_grow_every=120, rollout_max_k=6)

    # --- load the already-trained latent model ---
    ckpt = load_checkpoint("artifacts/latent_model.pt", device=device)
    latent, scodec, acodec = ckpt["model"], ckpt["scodec"], ckpt["acodec"]
    latent.to(device).eval()
    print(f"[cmp] loaded latent checkpoint ({sum(p.numel() for p in latent.parameters())/1e6:.2f}M params)",
          flush=True)

    # same eval seed as the original run; take a tractable prefix for BOTH models
    examples, _ = collect_examples(spec, args.n_eval, lambda e: True, 12345, scodec, acodec)
    print(f"[cmp] grading both models on {len(examples)} shared examples", flush=True)

    # --- re-grade the latent on this shared subset (cheap) ---
    lcore = core_metrics(latent, scodec, acodec, examples, device)
    latent_report = BenchReport(model_name="latent-m1", core=lcore,
                                meta={"steps": args.steps, "eval_n": len(examples)})
    print(f"[cmp] latent: single-step EM {lcore['single_step_exact_match']:.4f} "
          f"per-var {lcore['per_var_acc']:.4f} (n={lcore['n']})", flush=True)

    # --- train (or reload) + grade the matched token-space baseline ---
    os.makedirs("artifacts", exist_ok=True)
    tok_path = "artifacts/token_model.pt"
    if args.reuse_token and os.path.exists(tok_path):
        print(f"[cmp] reusing trained token baseline from {tok_path}", flush=True)
        tout = load_token_baseline(tok_path, device=token_device)
        bmodel, bser = tout["model"], tout["serializer"]
    else:
        print("\n=== Training token-space baseline (matched data/steps, CPU) ===", flush=True)
        # Train on CPU: the token model's 375-token sequences make the MPS allocator
        # accumulate to an OOM, and it competes with Ollama's ~54GB GPU footprint.
        # CPU uses pageable RAM (~70GB free) and never OOMs the unified pool.
        tout = train_token_baseline(spec=spec, codec_cfg=codec, tc=tc,
                                    n_train=args.n_train, n_eval=40, device=token_device)
        bmodel, bser = tout["model"], tout["serializer"]
        # persist BEFORE the memory-heavy eval, so an eval OOM never costs the retrain.
        save_token_baseline(tok_path, bmodel, bser, meta={"steps": args.steps})
        print(f"[cmp] saved token baseline -> {tok_path}", flush=True)
    # Greedy decode is autoregressive + KV-cache-free -> it fragments MPS unified
    # memory and competes with Ollama's ~54GB. Run it on CPU (pageable RAM): slower
    # but immune to the MPS OOM and to GPU-memory contention.
    bmodel.to(token_device).eval()
    print(f"[cmp] greedy-decoding token baseline on {token_device} "
          f"(autoregressive, eval-batch={args.eval_batch})...", flush=True)
    bcore = evaluate_token_baseline(bmodel, bser, scodec, acodec, examples, token_device,
                                    batch_size=args.eval_batch)
    baseline = BenchReport(model_name="token-baseline", core={
        "single_step_exact_match": bcore["step_exact_match"],
        "per_var_acc": bcore["per_var_acc"], "rollout_horizon": [], "n": bcore["n"]})
    print(f"[cmp] token: single-step EM {bcore['step_exact_match']:.4f} "
          f"per-var {bcore['per_var_acc']:.4f} (n={bcore['n']})", flush=True)

    latent_report.to_json("artifacts/latent_subset_report.json")
    baseline.to_json("artifacts/token_baseline_report.json")
    print("\n" + compare_reports(latent_report, baseline), flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
