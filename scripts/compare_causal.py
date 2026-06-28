"""The thesis test: grounded-latent vs token-space on CAUSAL counterfactuals.

Both models are already trained and saved (artifacts/latent_model.pt,
artifacts/token_model.pt). This builds ONE set of counterfactual intervention
pairs and grades BOTH models on the identical pairs:
  do(register)  — intervene on a register value, predict the next state
  do(action)    — swap the instruction, predict the next state
vs the identity ("no change") baseline. This is where the grounded latent should
show causal structure a token-space predictor lacks — the project's core claim.

    PYTHONPATH=. python scripts/compare_causal.py [--n 400]

Requires scripts/compare_token.py to have been run once (to produce token_model.pt)
and run_execwm_bench.py / compare_token.py to have produced latent_model.pt.
Token model runs on CPU (autoregressive greedy decode is memory-heavy on MPS).
"""
from __future__ import annotations

import argparse
import random

import torch

from execwm.data.state_codec import CodecConfig
from execwm.eval import counterfactual as cf
from execwm.eval.checkpoint import load_checkpoint
from execwm.eval.token_eval import evaluate_counterfactual_token
from execwm.substrate.generators import GenSpec
from execwm.train.train_m1 import pick_device
from execwm.train.train_token import load_token_baseline


def _row(name, reg, act, base):
    return (f"| {name} | {reg['exact_match']:.4f} | {act['exact_match']:.4f} "
            f"| {base:.4f} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="base transitions to sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device()
    token_device = torch.device("cpu")
    # must match the spec/codec the checkpoints were trained on (run_execwm_bench.py)
    spec = GenSpec(num_vars=4, max_depth=2, num_stmts=5,
                   max_const=5, max_input_val=5, max_loop_count=3)
    codec = CodecConfig(max_digits=6, base=10, max_pc=256)

    # --- load both trained models ---
    lk = load_checkpoint("artifacts/latent_model.pt", device=device)
    latent, scodec, acodec = lk["model"], lk["scodec"], lk["acodec"]
    latent.to(device).eval()
    tk = load_token_baseline("artifacts/token_model.pt", device=token_device)
    token, serializer = tk["model"], tk["serializer"]
    token.to(token_device).eval()
    print(f"[causal] latent {sum(p.numel() for p in latent.parameters())/1e6:.1f}M "
          f"| token {sum(p.numel() for p in token.parameters())/1e6:.1f}M", flush=True)

    # --- ONE set of intervention pairs, graded by BOTH models ---
    base = cf.sample_base_transitions(spec, args.n, args.seed, codec_cfg=codec)
    rng = random.Random(args.seed + 1)
    reg_pairs = cf.make_register_pairs(base, rng, value_range=(-10, 10))
    act_pairs = cf.make_action_pairs(base, rng)
    print(f"[causal] {len(reg_pairs)} register-do pairs, {len(act_pairs)} action-swap pairs",
          flush=True)

    # latent (fast)
    l_reg = cf.evaluate_counterfactual(latent, scodec, acodec, reg_pairs, device)
    l_act = cf.evaluate_counterfactual(latent, scodec, acodec, act_pairs, device)
    # token (CPU, chunked greedy decode)
    print("[causal] grading token model (CPU greedy decode)...", flush=True)
    t_reg = evaluate_counterfactual_token(token, serializer, scodec, acodec, reg_pairs, token_device)
    t_act = evaluate_counterfactual_token(token, serializer, scodec, acodec, act_pairs, token_device)

    ident_reg = cf.identity_baseline(reg_pairs)
    ident_act = cf.identity_baseline(act_pairs)

    print("\n# Causal counterfactuals — latent vs token-space\n")
    print("| Model | do(register) EM | do(action) EM | identity baseline |")
    print("| --- | --- | --- | --- |")
    print(_row("grounded latent", l_reg, l_act, ident_reg))
    print(_row("token-space", t_reg, t_act, ident_act))
    print(f"\nidentity baseline: reg {ident_reg:.4f}, action {ident_act:.4f}")
    print(f"latent  Δ over identity: reg {l_reg['exact_match']-ident_reg:+.4f}, "
          f"action {l_act['exact_match']-ident_act:+.4f}")
    print(f"token   Δ over identity: reg {t_reg['exact_match']-ident_reg:+.4f}, "
          f"action {t_act['exact_match']-ident_act:+.4f}")
    print(f"latent − token: do(register) {l_reg['exact_match']-t_reg['exact_match']:+.4f}, "
          f"do(action) {l_act['exact_match']-t_act['exact_match']:+.4f}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
