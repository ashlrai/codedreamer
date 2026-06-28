"""Sharpen the M2 causal thesis test: re-run latent-vs-token counterfactuals on the
EASY-ARITHMETIC slice (ADD/SUB only, 2-digit values) where single-step exact-match is
high (~0.86, per M1.5), so the causal signal isn't masked by arithmetic noise.

If the grounded latent has a causal-structure advantage the token-space model lacks,
it should show here — with arithmetic error largely removed as a confound. A tie here
too is strong evidence the advantage is genuinely absent (not masked).

Trains both models on the same easy spec/steps (latent on MPS, token on CPU), saves
checkpoints, then grades both on ONE identical set of intervention pairs.

    PYTHONPATH=. caffeinate -i python scripts/causal_easy.py [--steps 1500] [--n 300]
"""
from __future__ import annotations

import argparse
import os
import random

import torch

from execwm.data.state_codec import CodecConfig
from execwm.eval import counterfactual as cf
from execwm.eval.checkpoint import save_checkpoint
from execwm.eval.token_eval import evaluate_counterfactual_token
from execwm.substrate.generators import GenSpec
from execwm.substrate.vm import Op
from execwm.train.train_m1 import TrainConfig, pick_device, train
from execwm.train.train_token import save_token_baseline, train_token_baseline


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n", type=int, default=300, help="base transitions to sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = pick_device()
    token_device = torch.device("cpu")
    # the M1.5 easy-arithmetic slice: ADD/SUB only, values in [-99, 99]
    spec = GenSpec(num_vars=4, num_inputs=2, num_temps=10,
                   max_depth=2, num_stmts=5, max_const=3, max_input_val=3,
                   max_loop_count=3, arith_ops=(Op.ADD, Op.SUB),
                   use_heap=True, num_lists=1, list_len=4, max_steps=128)
    codec = CodecConfig(max_digits=2, base=10, max_pc=128)
    tc = TrainConfig(steps=args.steps, batch_size=48, max_len=18, lr=4e-4,
                     rollout_warmup=max(1, args.steps // 5),
                     rollout_grow_every=120, rollout_max_k=6)
    os.makedirs("artifacts", exist_ok=True)

    # --- latent (MPS); reuse saved checkpoint if present ---
    if os.path.exists("artifacts/latent_easy.pt"):
        from execwm.eval.checkpoint import load_checkpoint
        print("=== Reusing saved latent_easy.pt ===", flush=True)
        lk = load_checkpoint("artifacts/latent_easy.pt", device=device)
        latent, scodec, acodec = lk["model"], lk["scodec"], lk["acodec"]
        latent.to(device).eval()
        latent_em = 0.8427  # from the saved run's eval (logged)
    else:
        print("\n=== Training latent on easy-arithmetic slice (MPS) ===", flush=True)
        lout = train(spec=spec, codec_cfg=codec, tc=tc, n_train=4000, n_eval=600,
                     log_every=200, d_model=256, n_heads=8, enc_layers=3, dyn_layers=3)
        latent, scodec, acodec = lout["model"], lout["scodec"], lout["acodec"]
        save_checkpoint("artifacts/latent_easy.pt", latent, model_cfg=latent.cfg,
                        codec_cfg=codec, spec=spec, meta={"name": "latent-easy"})
        latent_em = lout["eval"]["step_exact_match"]
        print(f"[easy] latent single-step EM {latent_em:.4f} "
              f"per-var {lout['eval']['per_var_acc']:.4f}", flush=True)

    # --- token (CPU; easy codec -> short sequences, cheap) ---
    print("\n=== Training token baseline on easy-arithmetic slice (CPU) ===", flush=True)
    tout = train_token_baseline(spec=spec, codec_cfg=codec, tc=tc, n_train=4000,
                                n_eval=300, device=token_device, log_every=100)
    token, serializer = tout["model"], tout["serializer"]
    save_token_baseline("artifacts/token_easy.pt", token, serializer,
                        meta={"name": "token-easy", "steps": args.steps})
    token.to(token_device).eval()
    print(f"[easy] token single-step EM {tout['eval']['step_exact_match']:.4f} "
          f"per-var {tout['eval']['per_var_acc']:.4f}", flush=True)

    # --- causal comparison on identical pairs ---
    latent.to(device).eval()
    base = cf.sample_base_transitions(spec, args.n, args.seed, codec_cfg=codec)
    rng = random.Random(args.seed + 1)
    reg_pairs = cf.make_register_pairs(base, rng, value_range=(-10, 10))
    act_pairs = cf.make_action_pairs(base, rng)
    print(f"\n[easy] {len(reg_pairs)} register-do, {len(act_pairs)} action-swap pairs",
          flush=True)

    l_reg = cf.evaluate_counterfactual(latent, scodec, acodec, reg_pairs, device)
    l_act = cf.evaluate_counterfactual(latent, scodec, acodec, act_pairs, device)
    t_reg = evaluate_counterfactual_token(token, serializer, scodec, acodec, reg_pairs, token_device)
    t_act = evaluate_counterfactual_token(token, serializer, scodec, acodec, act_pairs, token_device)
    ident_reg, ident_act = cf.identity_baseline(reg_pairs), cf.identity_baseline(act_pairs)

    print("\n# Causal counterfactuals on EASY-ARITHMETIC — latent vs token\n")
    print("| Model | do(register) EM | do(action) EM | identity |")
    print("| --- | --- | --- | --- |")
    print(f"| grounded latent | {l_reg['exact_match']:.4f} | {l_act['exact_match']:.4f} | {ident_reg:.4f} |")
    print(f"| token-space | {t_reg['exact_match']:.4f} | {t_act['exact_match']:.4f} | {ident_act:.4f} |")
    print(f"\nlatent single-step EM {latent_em:.4f}, "
          f"token {tout['eval']['step_exact_match']:.4f}")
    print(f"latent − token: do(register) {l_reg['exact_match']-t_reg['exact_match']:+.4f}, "
          f"do(action) {l_act['exact_match']-t_act['exact_match']:+.4f}")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
