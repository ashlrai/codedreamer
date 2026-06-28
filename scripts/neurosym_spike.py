"""THE SPIKE: is the magnitude wall a capability limit, or a self-imposed choice?

Train ONE grounded latent world model on SMALL-magnitude programs (values <=~30) with
a codec WIDE enough to represent big numbers (max_digits=4 -> up to 9999). Then read
the SAME frozen model out two ways on an in-distribution split AND a magnitude-OOD
split (values ~300-800):

  em_learned        — every field decoded by the net (the status-quo readout)
  em_digits_oracle  — the net's predictions, but the numeric digit payload is supplied
                      by a perfect ALU (offloaded arithmetic); pc/type/sign/flags/
                      which-slot-changed are STILL the net's job

Hypothesis (the neurosymbolic thesis): on OOD magnitude, em_learned collapses to ~0
(the known wall) but em_digits_oracle stays high — because the latent encodes the
*structure* of the transition correctly across magnitude; only the learned digit
readout fails. If so, the wall is the digit head, and arithmetic should be offloaded.

    PYTHONPATH=. caffeinate -i python scripts/neurosym_spike.py [--steps 1500]
"""
from __future__ import annotations

import argparse
import os

import torch

from execwm.data.action_codec import ActionCodec
from execwm.data.dataset import collect_examples
from execwm.data.state_codec import CodecConfig, StateCodec
from execwm.eval.neurosym import field_breakdown
from execwm.substrate.generators import GenSpec
from execwm.substrate.vm import Op
from execwm.train.train_m1 import TrainConfig, pick_device, train


def _spec(max_const: int, max_input_val: int) -> GenSpec:
    return GenSpec(num_vars=4, num_inputs=2, num_temps=10, max_depth=2, num_stmts=5,
                   max_const=max_const, max_input_val=max_input_val, max_loop_count=3,
                   arith_ops=(Op.ADD, Op.SUB), use_heap=True, num_lists=1, list_len=4,
                   max_steps=128)


def _row(name: str, m: dict) -> str:
    g = lambda k: f"{m.get(k, float('nan')):.3f}"
    return (f"| {name} | {g('em_learned')} | {g('em_digits_oracle')} | {g('pc')} | "
            f"{g('written_sign')} | {g('written_digits')} | {g('arith_digits')} | "
            f"{g('cmp_result')} | {g('branch_pc')} | {m.get('n', 0)} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-eval", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device()
    codec = CodecConfig(max_digits=4, base=10, max_pc=128)   # represents up to 9999
    train_spec = _spec(max_const=5, max_input_val=5)         # in-dist: values <= ~30
    ood_spec = _spec(max_const=400, max_input_val=400)       # OOD: values ~300-800
    tc = TrainConfig(steps=args.steps, batch_size=48, max_len=18, lr=4e-4,
                     rollout_warmup=max(1, args.steps // 5),
                     rollout_grow_every=120, rollout_max_k=6)
    os.makedirs("artifacts", exist_ok=True)

    print("=== Training ONE model on small-magnitude / wide-codec slice (MPS) ===",
          flush=True)
    out = train(spec=train_spec, codec_cfg=codec, tc=tc,
                n_train=args.n_train, n_eval=args.n_eval, log_every=200,
                d_model=256, n_heads=8, enc_layers=3, dyn_layers=3, seed=args.seed)
    model, scodec, acodec = out["model"], out["scodec"], out["acodec"]

    try:
        from execwm.eval.checkpoint import save_checkpoint
        save_checkpoint("artifacts/neurosym_model.pt", model, model_cfg=model.cfg,
                        codec_cfg=codec, spec=train_spec, meta={"name": "neurosym"})
        print("[spike] saved artifacts/neurosym_model.pt", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[spike] checkpoint save skipped: {e}", flush=True)

    # --- build eval splits: in-distribution (small) and magnitude-OOD (large) ---
    print("\n=== Building eval splits ===", flush=True)
    indist_ex, _ = collect_examples(train_spec, args.n_eval, lambda ex: True,
                                    args.seed + 99, scodec, acodec)
    ood_ex, ood_att = collect_examples(ood_spec, args.n_eval, lambda ex: True,
                                       args.seed + 777, scodec, acodec)
    print(f"[spike] in-dist eval episodes {len(indist_ex)}; "
          f"OOD eval episodes {len(ood_ex)} (from {ood_att} attempts)", flush=True)

    indist = field_breakdown(model, indist_ex, scodec, acodec, device, max_len=18)
    ood = field_breakdown(model, ood_ex, scodec, acodec, device, max_len=18)

    print("\n# THE SPIKE — neurosymbolic readout vs magnitude\n")
    print("| split | EM learned | EM digits-oracle | pc acc | written sign | "
          "written digits | arith digits | cmp result | branch pc | n |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    print(_row("in-distribution (val<=30)", indist))
    print(_row("magnitude-OOD (val~300-800)", ood))
    print("\nReading the result:")
    print(f"  - If EM-learned collapses OOD ({ood.get('em_learned', float('nan')):.3f}) "
          f"but EM-digits-oracle stays high ({ood.get('em_digits_oracle', float('nan')):.3f}),")
    print("    the latent encodes transition STRUCTURE across magnitude; the wall is the digit head.")
    print(f"  - written-digits OOD = {ood.get('written_digits', float('nan')):.3f} "
          f"(the arithmetic that fails); pc/sign/cmp/branch = the structure that should survive.")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
