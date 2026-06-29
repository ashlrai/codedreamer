"""v2 frontier attempt: does a FIXED sinusoidal digit-position encoding close the
magnitude-OOD comparison gap?

FINDINGS_FRONTIER localized the residual to the encoder's representation of large
*input* operands: with LEARNED per-position embeddings, the high-order digit positions
are always zero in small-magnitude training and so are undertrained, making the encoding
of large values itself out-of-distribution. A FIXED (sinusoidal) position signal is never
OOD, so a large value stays a clean composition of (well-trained digit) + (fixed
position). This trains a model with `fixed_pos=True` on the SAME small-magnitude slice as
the baseline and measures whether OOD order/comparison + oracle-EM improve.

Baseline (learned positions, artifacts/neurosym_model.pt) OOD reference:
  em_digits_oracle 0.790 · cmp_result 0.626 · written_sign 0.798 · pc 0.986

    PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 \
      PYTHONPATH=. caffeinate -i python scripts/neurosym_v2_encoding.py --steps 1500
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
                   max_const=mc, max_input_val=mi, max_loop_count=3,
                   arith_ops=(Op.ADD, Op.SUB), use_heap=True, num_lists=1,
                   list_len=4, max_steps=128)


def _row(name: str, m: dict) -> str:
    g = lambda k: f"{m.get(k, float('nan')):.3f}"
    return (f"| {name} | {g('em_learned')} | {g('em_digits_oracle')} | {g('pc')} | "
            f"{g('written_sign')} | {g('cmp_result')} | {g('written_digits')} | {m.get('n', 0)} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device()
    codec = CodecConfig(max_digits=4, base=10, max_pc=128)
    train_spec = _spec(5, 5)
    ood_spec = _spec(400, 400)
    tc = TrainConfig(steps=args.steps, batch_size=48, max_len=18, lr=4e-4,
                     rollout_warmup=max(1, args.steps // 5),
                     rollout_grow_every=120, rollout_max_k=6)
    os.makedirs("artifacts", exist_ok=True)

    print("=== Training v2 (fixed_pos=True) on small-magnitude / wide-codec slice ===",
          flush=True)
    out = train(spec=train_spec, codec_cfg=codec, tc=tc, n_train=args.n_train,
                n_eval=args.n_eval, log_every=200, d_model=256, n_heads=8,
                enc_layers=3, dyn_layers=3, fixed_pos=True, seed=args.seed)
    model, scodec, acodec = out["model"], out["scodec"], out["acodec"]
    try:
        from execwm.eval.checkpoint import save_checkpoint
        save_checkpoint("artifacts/neurosym_v2.pt", model, model_cfg=model.cfg,
                        codec_cfg=codec, spec=train_spec, meta={"name": "neurosym-v2-fixedpos"})
        print("[v2] saved artifacts/neurosym_v2.pt", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[v2] save skipped: {e}", flush=True)

    indist, _ = collect_examples(train_spec, args.n_eval, lambda e: True,
                                 args.seed + 99, scodec, acodec)
    ood, att = collect_examples(ood_spec, args.n_eval, lambda e: True,
                                args.seed + 777, scodec, acodec)
    print(f"[v2] in-dist {len(indist)} / OOD {len(ood)} (from {att} attempts)", flush=True)

    di = field_breakdown(model, indist, scodec, acodec, device, max_len=18)
    do = field_breakdown(model, ood, scodec, acodec, device, max_len=18)
    print("\n# v2 fixed-position encoding — neurosymbolic readout vs magnitude\n")
    print("| split | EM learned | EM digits-oracle | pc acc | written sign | cmp result | written digits | n |")
    print("|---|---|---|---|---|---|---|---|")
    print(_row("in-distribution (val<=30)", di))
    print(_row("magnitude-OOD (val~300-800)", do))
    print("\n(baseline OOD, learned positions: em_digits_oracle 0.790 · cmp 0.626 · "
          "written_sign 0.798 · pc 0.986)")
    print(f"\nv2 OOD deltas vs baseline: em_oracle {do.get('em_digits_oracle',0)-0.790:+.3f}  "
          f"cmp {do.get('cmp_result',0)-0.626:+.3f}  sign {do.get('written_sign',0)-0.798:+.3f}  "
          f"pc {do.get('pc',0)-0.986:+.3f}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
