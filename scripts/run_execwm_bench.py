"""Run ExecWM-Bench: train (or load) a latent world model and a token-space
baseline, grade both on the SAME eval set, and print the comparison.

    PYTHONPATH=. python scripts/run_execwm_bench.py [--steps N] [--quick] [--no-baseline]

--quick runs only the core + counterfactual families (skips OOD/interp) for speed.
The latent model is the slotted GroundedLatentWM (M1); swap to ArithWM with
--arith. Writes execwm_bench_report.json and a checkpoint under ./artifacts/.
"""

from __future__ import annotations

import argparse
import os

from execwm.data.dataset import collect_examples
from execwm.data.state_codec import CodecConfig
from execwm.eval.execwm_bench import run_bench
from execwm.eval.report import BenchReport, compare_reports, scorecard_markdown
from execwm.substrate.generators import GenSpec
from execwm.train.train_m1 import TrainConfig, pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-baseline", action="store_true")
    ap.add_argument("--arith", action="store_true", help="use the carry-aware ArithWM")
    ap.add_argument("--n-eval", type=int, default=500)
    args = ap.parse_args()

    device = pick_device()
    spec = GenSpec(num_vars=4, max_depth=2, num_stmts=5,
                   max_const=5, max_input_val=5, max_loop_count=3)
    codec = CodecConfig(max_digits=6, base=10, max_pc=256)
    tc = TrainConfig(steps=args.steps, batch_size=48, max_len=18,
                     rollout_warmup=max(1, args.steps // 3),
                     rollout_grow_every=120, rollout_max_k=6)
    families = ("core", "counterfactual") if args.quick else \
        ("core", "ood", "interp", "counterfactual")

    # --- train the latent model ---
    if args.arith:
        from execwm.train.train_arith import train_arith
        out = train_arith(spec=spec, codec_cfg=codec, tc=tc, n_train=4000,
                          n_eval=300, d_model=192, n_heads=8, enc_layers=3, dyn_layers=3)
        model_name = "latent-arith"
    else:
        from execwm.train.train_m1 import train
        out = train(spec=spec, codec_cfg=codec, tc=tc, n_train=4000, n_eval=300,
                    d_model=256, n_heads=8, enc_layers=4, dyn_layers=4)
        model_name = "latent-m1"
    model, scodec, acodec = out["model"], out["scodec"], out["acodec"]

    # shared eval set so latent and baseline are graded on identical transitions
    examples, _ = collect_examples(spec, args.n_eval, lambda e: True, 12345, scodec, acodec)

    print("\n=== Running ExecWM-Bench on the latent model ===", flush=True)
    latent_report = run_bench(model, scodec, acodec, spec, codec, device=device,
                              model_name=model_name, families=families,
                              examples=examples,
                              meta={"steps": args.steps, "device": str(device)})

    os.makedirs("artifacts", exist_ok=True)
    latent_report.to_json("artifacts/execwm_bench_report.json")
    try:
        from execwm.eval.checkpoint import save_checkpoint
        save_checkpoint("artifacts/latent_model.pt", model,
                        model_cfg=model.cfg, codec_cfg=codec, spec=spec,
                        meta={"name": model_name})
        print("[bench] saved checkpoint artifacts/latent_model.pt", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[bench] checkpoint skipped: {e}")

    print("\n" + latent_report.to_markdown())
    print("\n" + scorecard_markdown(latent_report))

    # --- token-space baseline ---
    if not args.no_baseline:
        print("\n=== Training token-space baseline ===", flush=True)
        from execwm.train.train_token import (evaluate_token_baseline,
                                              train_token_baseline)
        tout = train_token_baseline(spec=spec, codec_cfg=codec, tc=tc,
                                    n_train=4000, n_eval=300, device=device)
        bmodel, bser = tout["model"], tout["serializer"]
        bcore = evaluate_token_baseline(bmodel, bser, scodec, acodec, examples, device)
        baseline = BenchReport(model_name="token-baseline", core={
            "single_step_exact_match": bcore["step_exact_match"],
            "per_var_acc": bcore["per_var_acc"], "rollout_horizon": [], "n": bcore["n"]})
        print("\n" + compare_reports(latent_report, baseline))
        baseline.to_json("artifacts/token_baseline_report.json")


if __name__ == "__main__":
    main()
