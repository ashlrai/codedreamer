"""M3 payoff experiment: is the trained latent world model a viable *cheap* scorer
for planning over program edits -- one that scores goal tasks while running ZERO VM
executions during search?

Loads the trained ``artifacts/latent_model.pt`` (a ``GroundedLatentWM`` on the M2
hard spec), builds N constructively-solvable goal tasks on that model's spec, and
compares FOUR scoring methods on the SAME tasks:

    (a) vm_search            brute-force VM search (no world model) -- the baseline
    (b) beam_plan + cheap    structural single-pass heuristic (ignores control flow)
    (c) beam_plan + WM       the LEARNED latent world model (this experiment)
    (d) beam_plan + oracle   beam search scored by the real VM -- planning upper bound

For each method it reports success rate and mean real-VM executions (over solved
tasks), plus the executions-saved fraction of the WM scorer vs the brute-force
baseline (over tasks both solved). The cheap and WM scorers run NO VM during search,
so their reported executions are purely the planner's verification runs.

    PYTHONPATH=. python3 scripts/wm_plan_eval.py [--n 40] [--seed 0] [--beam 6]

Runs on CPU; small N keeps it to a few minutes. HONESTY: the WM scorer is only as
good as the model's single-step decode compounded over a closed-loop rollout, so it
is expected to MISS goals on this ~0.40 single-step-exact-match model. The numbers
are reported as-is.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import time

import torch

from execwm.eval.checkpoint import load_checkpoint
from execwm.plan.goal_tasks import GoalTask, make_goal_task
from execwm.plan.metrics import executions_saved
from execwm.plan.planner import OracleScorer, beam_plan, cheap_scorer
from execwm.plan.search_baseline import SearchResult, vm_search
from execwm.plan.wm_scorer import WorldModelScorer


def build_tasks(spec, codec_cfg, n: int, seed: int, edit_budget: int) -> list[GoalTask]:
    """Construct ``n`` solvable goal tasks on the model's own spec, keeping only
    edits whose resulting states are codec-encodable (so the WM can represent them)."""
    rng = random.Random(seed)
    tasks: list[GoalTask] = []
    while len(tasks) < n:
        tasks.append(make_goal_task(rng, spec, codec_cfg=codec_cfg,
                                    edit_budget=edit_budget))
    return tasks


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def summarize(name: str, results: list[SearchResult], n: int) -> dict:
    solved = [r for r in results if r.solved]
    return {
        "name": name,
        "solved": len(solved),
        "rate": len(solved) / n if n else 0.0,
        "mean_execs_solved": _mean([r.executions for r in solved]),
        "mean_execs_all": _mean([r.executions for r in results]),
    }


_METHODS = ("base", "cheap", "wm", "oracle")
_LABELS = {
    "base": "(a) vm_search brute-force",
    "cheap": "(b) beam + cheap scorer",
    "wm": "(c) beam + WM scorer",
    "oracle": "(d) beam + oracle (upper bound)",
}


def _rec(r: SearchResult) -> dict:
    return {"solved": bool(r.solved), "execs": int(r.executions)}


def report_from_jsonl(path: str) -> None:
    """Aggregate per-task records appended across short batches into the table.

    Each line is one task: {"seed","i","base","cheap","wm","oracle"} where every
    method value is {"solved","execs"}. De-duplicates on (seed, i)."""
    seen: dict[tuple[int, int], dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            seen[(d["seed"], d["i"])] = d
    recs = list(seen.values())
    n = len(recs)

    print(f"\n## M3 payoff: WM-as-planning-scorer (aggregated, n={n} distinct tasks)\n")
    print("| Method | Success | Mean real-VM execs (solved) |")
    print("|---|---|---|")
    for m in _METHODS:
        solved = [d[m]["execs"] for d in recs if d[m]["solved"]]
        rate = len(solved) / n if n else 0.0
        mean_e = _mean([float(x) for x in solved])
        print(f"| {_LABELS[m]} | {len(solved)}/{n} ({rate:.0%}) | {mean_e:.1f} |")

    def saved_frac(method: str) -> tuple[int, float]:
        fracs = []
        for d in recs:
            b, p = d["base"], d[method]
            if b["solved"] and p["solved"] and b["execs"] > 0:
                fracs.append((b["execs"] - p["execs"]) / b["execs"])
        return len(fracs), _mean(fracs)

    wm_n, wm_f = saved_frac("wm")
    ch_n, ch_f = saved_frac("cheap")
    wm_to = sum(1 for d in recs if d.get("wm_timeout"))
    print(f"\n- WM scorer runs **0** VM executions during search (by construction).")
    print(f"- WM vs brute-force: both solved on {wm_n}/{n} tasks; "
          f"mean executions saved = {wm_f:.1%}.")
    print(f"- cheap vs brute-force: both solved on {ch_n}/{n} tasks; "
          f"mean executions saved = {ch_f:.1%}.")
    print(f"- WM compute-timeouts (rollout exceeded budget, not a latent miss): "
          f"{wm_to}/{n} tasks.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/latent_model.pt")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beam", type=int, default=6)
    ap.add_argument("--edit-budget", type=int, default=2)
    ap.add_argument("--skip-oracle", action="store_true",
                    help="skip the slow VM-oracle upper-bound method (already calibrated)")
    ap.add_argument("--wm-max-steps", type=int, default=64)
    ap.add_argument("--max-executions", type=int, default=200_000,
                    help="planner verification cap (WM/cheap run few real VMs)")
    ap.add_argument("--baseline-max-executions", type=int, default=20_000,
                    help="brute-force VM-search cap; tasks needing more count as "
                         "baseline failures (honest within-budget comparison)")
    ap.add_argument("--jsonl", default=None,
                    help="append one per-task result record to this JSONL file")
    ap.add_argument("--report-from", default=None,
                    help="aggregate an existing JSONL file into the table and exit")
    ap.add_argument("--guard", type=float, default=6.0,
                    help="per-method wall-clock guard (s) for base/cheap/oracle")
    ap.add_argument("--wm-guard", type=float, default=15.0,
                    help="wall-clock guard (s) for the WM scorer; exceeding it is "
                         "recorded as a timeout (separated from genuine misses)")
    ap.add_argument("--time-budget", type=float, default=22.0,
                    help="stop this invocation after this many seconds of new work "
                         "(resume the rest on a later run; 0 = no budget)")
    args = ap.parse_args()

    if args.report_from:
        report_from_jsonl(args.report_from)
        return

    torch.set_num_threads(1)  # predictable + polite on a shared/loaded machine
    device = torch.device("cpu")
    ck = load_checkpoint(args.ckpt, device=device)
    model, scodec, acodec, spec, codec_cfg = (
        ck["model"], ck["scodec"], ck["acodec"], ck["spec"], ck["codec_cfg"])
    print(f"[wm-plan] loaded {args.ckpt}  meta={ck['meta']}", flush=True)

    t0 = time.perf_counter()
    tasks = build_tasks(spec, codec_cfg, args.n, args.seed, args.edit_budget)
    print(f"[wm-plan] built {len(tasks)} goal tasks in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    wm_scorer = WorldModelScorer(model, scodec, acodec, device,
                                 max_steps=args.wm_max_steps)
    oracle = OracleScorer(max_steps=spec.max_steps)

    planner_kw = dict(beam_width=args.beam, max_depth=args.edit_budget,
                      max_executions=args.max_executions)

    def guarded(fn, budget: float) -> tuple[SearchResult, bool]:
        """Run ``fn`` under a wall-clock guard. Returns ``(result, timed_out)``;
        on timeout returns a failed :class:`SearchResult` and ``True`` so the report
        can separate genuine misses from compute-timeouts. Requires the main thread."""
        if budget <= 0:
            return fn(), False
        def _timeout(signum, frame):
            raise TimeoutError
        old = signal.signal(signal.SIGALRM, _timeout)
        signal.setitimer(signal.ITIMER_REAL, budget)
        try:
            return fn(), False
        except TimeoutError:
            return SearchResult(False, 0, None, 0), True
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)

    # Resume: skip (seed, i) pairs already recorded so repeated short invocations
    # accumulate coverage (the machine kills jobs that run much past ~30s).
    done: set[int] = set()
    if args.jsonl and os.path.exists(args.jsonl):
        with open(args.jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    d = json.loads(line)
                    if d["seed"] == args.seed:
                        done.add(d["i"])

    base_results: list[SearchResult] = []
    cheap_results: list[SearchResult] = []
    wm_results: list[SearchResult] = []
    oracle_results: list[SearchResult] = []

    t0 = time.perf_counter()
    processed = 0
    for i, task in enumerate(tasks):
        if i in done:
            continue
        if args.time_budget > 0 and time.perf_counter() - t0 > args.time_budget:
            print(f"[wm-plan] time budget hit; processed {processed} new tasks",
                  flush=True)
            break
        b, _ = guarded(lambda t=task: vm_search(
            t, max_executions=args.baseline_max_executions, strategy="bfs"),
            args.guard)
        c, _ = guarded(lambda t=task: beam_plan(t, scorer=cheap_scorer, **planner_kw),
                       args.guard)
        w, w_to = guarded(lambda t=task: beam_plan(t, scorer=wm_scorer, **planner_kw),
                          args.wm_guard)
        if args.skip_oracle:
            o = SearchResult(solved=False, executions=0, plan=None, depth=0)
        else:
            o, _ = guarded(lambda t=task: beam_plan(t, scorer=oracle, **planner_kw),
                           args.guard)
        base_results.append(b)
        cheap_results.append(c)
        wm_results.append(w)
        oracle_results.append(o)
        processed += 1
        if args.jsonl:
            with open(args.jsonl, "a") as f:
                f.write(json.dumps({
                    "seed": args.seed, "i": i, "prog_len": len(task.base_bytecode),
                    "base": _rec(b), "cheap": _rec(c),
                    "wm": _rec(w), "oracle": _rec(o), "wm_timeout": bool(w_to),
                }) + "\n")
        print(f"[wm-plan] seed{args.seed} task{i} len{len(task.base_bytecode)}: "
              f"base={b.solved}/{b.executions} cheap={c.solved}/{c.executions} "
              f"wm={w.solved}/{w.executions}{'(TO)' if w_to else ''} "
              f"oracle={o.solved}/{o.executions} "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)

    assert wm_scorer.executions == 0, "WM scorer must run zero VM executions"

    n = len(base_results)
    if n == 0:
        print(f"[wm-plan] no new tasks processed for seed {args.seed} "
              f"(all recorded or budget 0)", flush=True)
        return
    rows = [
        summarize("(a) vm_search brute-force", base_results, n),
        summarize("(b) beam + cheap scorer", cheap_results, n),
        summarize("(c) beam + WM scorer", wm_results, n),
        summarize("(d) beam + oracle (upper bound)", oracle_results, n),
    ]

    # executions-saved of the WM scorer vs the brute-force baseline (both solved).
    wm_saved = [executions_saved(b, p)
                for b, p in zip(base_results, wm_results)]
    wm_both = [s for s in wm_saved if s["both_solved"]]
    wm_saved_frac = _mean([s["saved_frac"] for s in wm_both])
    cheap_saved = [executions_saved(b, p)
                   for b, p in zip(base_results, cheap_results)]
    cheap_both = [s for s in cheap_saved if s["both_solved"]]
    cheap_saved_frac = _mean([s["saved_frac"] for s in cheap_both])

    print(f"\n## M3 payoff: WM-as-planning-scorer  (n={n}, beam={args.beam}, "
          f"edit_budget={args.edit_budget}, wm_max_steps={args.wm_max_steps})\n")
    print("| Method | Success | Mean real-VM execs (solved) |")
    print("|---|---|---|")
    for r in rows:
        print(f"| {r['name']} | {r['solved']}/{n} ({r['rate']:.0%}) | "
              f"{r['mean_execs_solved']:.1f} |")

    print(f"\n- WM scorer ran **{wm_scorer.executions}** VM executions during search "
          f"(by construction).")
    print(f"- WM vs brute-force: both solved on {len(wm_both)}/{n} tasks; "
          f"mean executions saved = {wm_saved_frac:.1%}.")
    print(f"- cheap vs brute-force: both solved on {len(cheap_both)}/{n} tasks; "
          f"mean executions saved = {cheap_saved_frac:.1%}.")


if __name__ == "__main__":
    main()
