"""Magnitude-curriculum vs baseline, head-to-head on a MAGNITUDE-STRESSED spec.

The project's binding constraint (M1.5/M1.6) is the single-step arithmetic error:
rollout@k decays ~geometrically in single-step exact-match, and the arithmetic
head's value loss never converges to 0 on hard magnitudes. The candidate fix is
the *magnitude curriculum* (execwm/train/curriculum.py): ramp the DATA's operand
magnitude small->large during training (codec digit width FIXED) so carries are
learned progressively (Abacus 2405.17399, Learning-to-Execute 1410.4615). Its
tiny-budget test so far was inconclusive; this script is the clean, matched-config
head-to-head and is GPU-ready.

WHAT IT DOES
  Trains BOTH, at the SAME budget / model size / seed, on the SAME spec:
    (a) baseline   train_arith            -- hard magnitude from step 0
    (b) curriculum train_arith_curriculum -- ramps magnitude 1 -> target
  EVALUATION for BOTH is ALWAYS at the FULL TARGET (stressed) magnitude, on an
  identical eval pool (same spec + same seed+99 -> collect_examples is
  deterministic), so the reported single-step exact-match / per-var is a fair
  measure of generalization to the hard distribution. Prints a comparison table.

THE MAGNITUDE-STRESSED SPEC (why it is hard)
  The default arith spec is GenSpec(max_const=5, max_input_val=5): operands are
  essentially single-digit, so a single arithmetic step has almost no carry chain
  to get right -- arithmetic is trivially easy and the curriculum has nothing to
  bite on. This experiment instead targets:

      max_const = max_input_val = 300   (vs 5)   -> operands span 1..3 digits
      max_digits = 6 (FIXED codec width, base 10) -> range up to 10^6

  With operands up to ~300, additions routinely produce multi-position carry
  chains (e.g. 287 + 158) and products reach 4-5 digits (300*300 = 90,000); the
  rare triple-product overflow (>10^6) is simply filtered out by the codec's
  _encodable check, so the surviving training/eval distribution is genuinely
  multi-digit with real carry propagation -- exactly the regime where carry-aware
  + curriculum is hypothesized to help, and where the easy default is silent.
  Every other GenSpec field matches the baseline arith default, so magnitude is
  the ONLY axis that differs -- isolating the curriculum's effect.

USAGE
  Tiny smoke / inconclusive signal (CPU, what this agent ran):
    PYTHONPATH=. python3 scripts/curriculum_experiment.py \
        --steps 300 --d-model 64 --layers 2 --n-train 1200 --n-eval 300 --device cpu

  Full budget (GPU -- the real comparison; needs ~1000+ steps to converge):
    PYTHONPATH=. python3 scripts/curriculum_experiment.py \
        --steps 1500 --d-model 256 --layers 4 --n-train 4000 --n-eval 600
"""
from __future__ import annotations

import argparse
import time

import torch

from execwm.data.state_codec import CodecConfig
from execwm.substrate.generators import GenSpec
from execwm.train.curriculum import linear_magnitude_curriculum
from execwm.train.train_arith import train_arith
from execwm.train.train_arith_curriculum import train_arith_curriculum
from execwm.train.train_m1 import TrainConfig, pick_device


def _horizon_str(h: list[float]) -> str:
    pick = {0: "k1", 2: "k3", 4: "k5"}
    return "  ".join(f"{lbl}:{h[i]:.3f}" for i, lbl in pick.items() if i < len(h))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=300, help="training steps (both arms)")
    ap.add_argument("--device", type=str, default=None,
                    help="cpu|cuda|mps (default: auto via pick_device)")
    ap.add_argument("--d-model", type=int, default=64, help="model width (both arms)")
    ap.add_argument("--layers", type=int, default=2,
                    help="encoder & dynamics depth (both arms)")
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-train", type=int, default=1200, help="training episodes")
    ap.add_argument("--n-eval", type=int, default=300, help="eval episodes (target mag)")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stages", type=int, default=3, help="curriculum stages")
    ap.add_argument("--max-const", type=int, default=300,
                    help="target literal-constant magnitude (stressed)")
    ap.add_argument("--max-input-val", type=int, default=300,
                    help="target input/heap magnitude (stressed)")
    ap.add_argument("--max-digits", type=int, default=6,
                    help="FIXED codec digit width (never ramped)")
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else pick_device()

    # --- the magnitude-stressed target spec (see module docstring) ---
    # Identical to the baseline arith default EXCEPT the two magnitude fields.
    spec = GenSpec(num_vars=4, max_depth=2, num_stmts=5,
                   max_const=args.max_const, max_input_val=args.max_input_val,
                   max_loop_count=3)
    codec = CodecConfig(max_digits=args.max_digits, base=10, max_pc=256)
    tc = TrainConfig(steps=args.steps, batch_size=args.batch_size)
    model_kw = dict(d_model=args.d_model, n_heads=args.n_heads,
                    enc_layers=args.layers, dyn_layers=args.layers)
    curriculum = linear_magnitude_curriculum(spec, n_stages=args.stages, start_max=1)

    print("=" * 78, flush=True)
    print("MAGNITUDE-CURRICULUM EXPERIMENT (matched config, eval @ full target mag)",
          flush=True)
    print(f"  device={device}  steps={args.steps}  d_model={args.d_model} "
          f"layers={args.layers}  n_train={args.n_train} n_eval={args.n_eval}",
          flush=True)
    print(f"  TARGET (stressed) spec: max_const={spec.max_const} "
          f"max_input_val={spec.max_input_val}  codec max_digits={codec.max_digits} "
          f"(range 10^{codec.max_digits})", flush=True)
    print(f"  curriculum stages (frac, max_const, max_input_val): "
          + " -> ".join(f"({s.max_const},{s.max_input_val})" for s in curriculum.stages),
          flush=True)
    print("=" * 78, flush=True)

    # --- arm (a): baseline, hard magnitude from step 0 ---
    print("\n### BASELINE (train_arith) -- hard magnitude from step 0\n", flush=True)
    t0 = time.perf_counter()
    base = train_arith(spec=spec, codec_cfg=codec, tc=tc, n_train=args.n_train,
                       n_eval=args.n_eval, seed=args.seed, device=device,
                       log_every=args.log_every, **model_kw)
    t_base = time.perf_counter() - t0

    # --- arm (b): curriculum, ramps magnitude 1 -> target ---
    print("\n### CURRICULUM (train_arith_curriculum) -- ramps magnitude 1 -> target\n",
          flush=True)
    t0 = time.perf_counter()
    curr = train_arith_curriculum(base_spec=spec, codec_cfg=codec, tc=tc,
                                  curriculum=curriculum, n_train=args.n_train,
                                  n_eval=args.n_eval, seed=args.seed, device=device,
                                  log_every=args.log_every, **model_kw)
    t_curr = time.perf_counter() - t0

    # --- comparison (eval ALWAYS at full target magnitude) ---
    be, ce = base["eval"], curr["eval"]
    print("\n" + "=" * 78, flush=True)
    print("# RESULTS -- single-step exact-match @ FULL TARGET magnitude "
          f"(max_const={spec.max_const}, max_input_val={spec.max_input_val})\n",
          flush=True)
    print("| arm        | step exact-match | per-var acc | rollout            | wall(s) |",
          flush=True)
    print("| ---------- | ---------------- | ----------- | ------------------ | ------- |",
          flush=True)
    print(f"| baseline   | {be['step_exact_match']:.4f}           | "
          f"{be['per_var_acc']:.4f}      | {_horizon_str(base['rollout_horizon'])} | {t_base:6.1f} |",
          flush=True)
    print(f"| curriculum | {ce['step_exact_match']:.4f}           | "
          f"{ce['per_var_acc']:.4f}      | {_horizon_str(curr['rollout_horizon'])} | {t_curr:6.1f} |",
          flush=True)

    d_em = ce["step_exact_match"] - be["step_exact_match"]
    d_pv = ce["per_var_acc"] - be["per_var_acc"]
    print(f"\ncurriculum - baseline:  step exact-match {d_em:+.4f}   per-var {d_pv:+.4f}",
          flush=True)
    verdict = ("curriculum HELPS" if d_em > 0.01 else
               "curriculum HURTS" if d_em < -0.01 else "INCONCLUSIVE / tie")
    print(f"verdict (this budget): {verdict}", flush=True)
    print("\nNOTE: convergence on this stressed spec needs ~1000+ steps and a larger "
          "model; a short CPU run is only a smoke signal, not a conclusion.", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
