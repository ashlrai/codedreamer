"""Does the neurosymbolic result generalize to HARDER arithmetic — multiplication?

ADD/SUB is the easy case. Multiplication makes the digit-decode problem far worse (the
result's magnitude is the product, so the digits a pure-net must emit are even further
out of distribution) — while a symbolic ALU computes it exactly for free. If "offload
arithmetic -> magnitude-invariant" is real, it should hold *more* strongly with MUL in
the mix: pure-net collapses harder, neurosymbolic stays high.

Trains ONE model on small-magnitude ADD/SUB/MUL programs with a wide codec, then runs
the same neurosymbolic readout breakdown in-distribution vs magnitude-OOD.

    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 \
      PYTHONPATH=. caffeinate -i python scripts/neurosym_mul.py --steps 1500
"""
from __future__ import annotations

import argparse
import os

import torch

from execwm.data.dataset import collect_examples
from execwm.data.state_codec import CodecConfig
from execwm.eval.neurosym import field_breakdown
from execwm.substrate.generators import GenSpec
from execwm.substrate.vm import Op
from execwm.train.train_m1 import TrainConfig, pick_device, train


def _spec(mc: int, mi: int) -> GenSpec:
    return GenSpec(num_vars=4, num_inputs=2, num_temps=10, max_depth=2, num_stmts=5,
                   max_const=mc, max_input_val=mi, max_loop_count=2,
                   arith_ops=(Op.ADD, Op.SUB, Op.MUL), use_heap=True, num_lists=1,
                   list_len=4, max_steps=128)


def _row(name: str, m: dict) -> str:
    g = lambda k: f"{m.get(k, float('nan')):.3f}"
    return (f"| {name} | {g('em_learned')} | {g('em_digits_oracle')} | {g('pc')} | "
            f"{g('written_digits')} | {g('arith_digits')} | {g('cmp_result')} | {m.get('n', 0)} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device()
    codec = CodecConfig(max_digits=6, base=10, max_pc=128)   # up to 999,999 for products
    train_spec = _spec(4, 4)        # in-dist: small values (products <= ~16)
    ood_spec = _spec(40, 40)        # OOD: products up to ~1600 (80x the trained value range)
    tc = TrainConfig(steps=args.steps, batch_size=48, max_len=18, lr=4e-4,
                     rollout_warmup=max(1, args.steps // 5),
                     rollout_grow_every=120, rollout_max_k=6)
    os.makedirs("artifacts", exist_ok=True)

    print("=== Training ONE model on small-magnitude ADD/SUB/MUL (wide codec) ===", flush=True)
    out = train(spec=train_spec, codec_cfg=codec, tc=tc, n_train=args.n_train,
                n_eval=args.n_eval, log_every=200, d_model=256, n_heads=8,
                enc_layers=3, dyn_layers=3, seed=args.seed)
    model, scodec, acodec = out["model"], out["scodec"], out["acodec"]
    try:
        from execwm.eval.checkpoint import save_checkpoint
        save_checkpoint("artifacts/neurosym_mul.pt", model, model_cfg=model.cfg,
                        codec_cfg=codec, spec=train_spec, meta={"name": "neurosym-mul"})
        print("[mul] saved artifacts/neurosym_mul.pt", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[mul] save skipped: {e}", flush=True)

    indist, _ = collect_examples(train_spec, args.n_eval, lambda e: True,
                                 args.seed + 99, scodec, acodec)
    ood, att = collect_examples(ood_spec, args.n_eval, lambda e: True,
                                args.seed + 777, scodec, acodec)
    print(f"[mul] in-dist {len(indist)} / OOD {len(ood)} (from {att} attempts)", flush=True)

    di = field_breakdown(model, indist, scodec, acodec, device, max_len=18)
    do = field_breakdown(model, ood, scodec, acodec, device, max_len=18)
    print("\n# Neurosymbolic readout with MULTIPLICATION — vs magnitude\n")
    print("| split | EM learned | EM digits-oracle | pc acc | written digits | arith digits | cmp result | n |")
    print("|---|---|---|---|---|---|---|---|")
    print(_row("in-distribution (val<=4, products<=16)", di))
    print(_row("magnitude-OOD (val~40, products<=1600)", do))
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
