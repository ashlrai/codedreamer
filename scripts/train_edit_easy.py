"""Train the edit-conditioned dynamics model (M3 step-3) on the EASY-ARITHMETIC slice,
where single-step arithmetic IS learnable (latent single-step EM 0.84). This tests
whether the no-rollout edit model can predict (a) the divergence point and (b) the
edited next-states accurately, in the regime where the underlying arithmetic works —
the precondition for a divergence-aware planner to realize the ~46% saving.

    PYTHONPATH=. caffeinate -i python scripts/train_edit_easy.py [--steps 1500]

Saves artifacts/edit_easy.pt and prints div_first_acc / div_step_acc /
edited_exact_match / edited_per_var_acc.
"""
from __future__ import annotations

import argparse

import torch

from execwm.data.state_codec import CodecConfig
from execwm.substrate.generators import GenSpec
from execwm.substrate.vm import Op
from execwm.train.train_edit import TrainConfig, train_edit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-eval", type=int, default=500)
    args = ap.parse_args()

    spec = GenSpec(num_vars=4, num_inputs=2, num_temps=10,
                   max_depth=2, num_stmts=5, max_const=3, max_input_val=3,
                   max_loop_count=3, arith_ops=(Op.ADD, Op.SUB),
                   use_heap=True, num_lists=1, list_len=4, max_steps=128)
    codec = CodecConfig(max_digits=2, base=10, max_pc=128)
    tc = TrainConfig(steps=args.steps, batch_size=32, lr=3e-4, max_len=20,
                     w_div=1.0, w_ground=1.0)

    out = train_edit(spec=spec, codec_cfg=codec, tc=tc,
                     n_train=args.n_train, n_eval=args.n_eval, log_every=150,
                     d_model=256, n_heads=8, enc_layers=3, dyn_layers=3)
    ev = out["eval"]
    print("\n# Edit-conditioned dynamics — easy-arith eval")
    print(f"  divergence first-step acc : {ev['div_first_acc']:.4f}  "
          f"(finds the right first-divergence step)")
    print(f"  divergence per-step acc   : {ev['div_step_acc']:.4f}")
    print(f"  edited-state exact-match  : {ev['edited_exact_match']:.4f}")
    print(f"  edited per-var acc        : {ev['edited_per_var_acc']:.4f}")
    print(f"  (n_eps={ev['n_eps']}, n_steps={ev['n_steps']})")

    try:
        torch.save({"state_dict": out["model"].state_dict(),
                    "model_config": vars(out["model"].cfg)},
                   "artifacts/edit_easy.pt")
        print("[edit] saved artifacts/edit_easy.pt")
    except Exception as e:  # noqa: BLE001
        print(f"[edit] save skipped: {e}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
