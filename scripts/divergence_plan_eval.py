"""M3 payoff measurement: does the TRAINED edit-conditioned model, used as a
divergence-aware planning scorer, solve goal tasks with zero search-time VM calls and
no latent rollout? Compares against brute-force VM search and the cheap structural
scorer on the same easy-arith goal tasks.

    PYTHONPATH=. caffeinate -i python scripts/divergence_plan_eval.py [--n 30] [--edit-budget 1]
"""
from __future__ import annotations

import argparse
import random

import torch

from execwm.data.state_codec import CodecConfig
from execwm.plan.divergence_planner import EditConditionedWMScorer, divergence_beam_plan
from execwm.plan.goal_tasks import make_goal_task
from execwm.plan.planner import beam_plan, cheap_scorer
from execwm.plan.search_baseline import vm_search
from execwm.substrate.generators import GenSpec
from execwm.substrate.vm import Op
from execwm.train.train_edit import build
from execwm.train.train_m1 import pick_device


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--edit-budget", type=int, default=1)
    ap.add_argument("--max-executions", type=int, default=4000)
    args = ap.parse_args()

    device = torch.device("cpu")  # planner search is light; edit model is small
    spec = GenSpec(num_vars=4, num_inputs=2, num_temps=10,
                   max_depth=2, num_stmts=5, max_const=3, max_input_val=3,
                   max_loop_count=3, arith_ops=(Op.ADD, Op.SUB),
                   use_heap=True, num_lists=1, list_len=4, max_steps=128)
    codec = CodecConfig(max_digits=2, base=10, max_pc=128)

    # rebuild the model skeleton with the SAME config used in training, load weights
    from execwm.substrate.edits import EditConfig
    edit_cfg = EditConfig(max_program_len=codec.max_pc)
    model, scodec, ecodec = build(spec, codec, edit_cfg,
                                  d_model=256, n_heads=8, enc_layers=3, dyn_layers=3)
    ck = torch.load("artifacts/edit_easy.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    print(f"[divplan] loaded edit_easy.pt ({sum(p.numel() for p in model.parameters())/1e6:.1f}M)",
          flush=True)

    rng = random.Random(args.seed)
    tasks = []
    for _ in range(args.n * 3):
        if len(tasks) >= args.n:
            break
        try:
            tasks.append(make_goal_task(rng, spec, codec_cfg=codec,
                                        edit_budget=args.edit_budget))
        except Exception:  # noqa: BLE001 - rare degenerate construction
            continue
    print(f"[divplan] built {len(tasks)} goal tasks (edit_budget={args.edit_budget})",
          flush=True)

    scorer = EditConditionedWMScorer(model, scodec, ecodec, device=device)
    planner_kw = dict(beam_width=args.beam, max_depth=args.edit_budget,
                      max_executions=args.max_executions)

    base_solved = base_ex = wm_solved = wm_ex = ch_solved = ch_ex = 0
    saved_fracs = []
    for i, task in enumerate(tasks):
        cfg = task.config
        b = vm_search(task, cfg, max_executions=args.max_executions, strategy="bfs")
        c = beam_plan(task, cfg, scorer=cheap_scorer, **planner_kw)
        scorer.reset()
        w = divergence_beam_plan(task, cfg, scorer=scorer, **planner_kw)
        base_solved += b.solved; base_ex += b.executions
        ch_solved += c.solved; ch_ex += c.executions
        wm_solved += w.solved; wm_ex += w.executions
        if b.solved and w.solved and b.executions > 0:
            saved_fracs.append((b.executions - w.executions) / b.executions)
        print(f"[divplan] task{i} len{len(task.base_bytecode)}: "
              f"base={b.solved}/{b.executions} cheap={c.solved}/{c.executions} "
              f"wm={w.solved}/{w.executions}", flush=True)

    n = len(tasks)
    print("\n# M3 payoff: divergence-aware planner (TRAINED edit model)\n")
    print("| Method | Success | Mean VM execs |")
    print("|---|---|---|")
    print(f"| vm_search brute-force | {base_solved}/{n} | {base_ex/max(n,1):.1f} |")
    print(f"| beam + cheap scorer | {ch_solved}/{n} | {ch_ex/max(n,1):.1f} |")
    print(f"| beam + edit-WM scorer | {wm_solved}/{n} | {wm_ex/max(n,1):.1f} |")
    print(f"\nWM vs brute-force (both solved): mean executions saved = {_mean(saved_fracs):.1%} "
          f"over {len(saved_fracs)} tasks")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
