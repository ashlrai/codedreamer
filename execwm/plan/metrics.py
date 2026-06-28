"""The "real-VM-executions saved" metric — the R4 calibration headline.

:func:`executions_saved` compares one baseline search against one planned search,
claiming a saving ONLY when both actually solved the task (otherwise there is no
honest apples-to-apples comparison and ``saved`` is reported as ``0``).
:func:`evaluate_planning` aggregates this over a list of :class:`GoalTask` s,
reporting per-method success rates and mean executions, plus the mean saving
fraction over the tasks BOTH methods solved — and is explicit about how many
tasks each method solved.

No neural model is involved; this measures whether the symbolic re-encode planner
already saves real interpreter executions over brute-force VM search. That number
is the bar a learned edit-conditioned model (M3 step 3) must clear.
"""

from __future__ import annotations

from typing import Iterable

from ..substrate.vm import Config
from .goal_tasks import GoalTask
from .planner import beam_plan
from .search_baseline import SearchResult, vm_search


def executions_saved(baseline: SearchResult, planned: SearchResult) -> dict:
    """Real VM executions the planner saved over the baseline, honestly gated.

    A saving is claimed only when ``both_solved``; otherwise ``saved`` is ``0`` and
    ``saved_frac`` is ``0.0`` (we never credit a planner that failed, nor count a
    "saving" against a baseline that itself failed).
    """
    both_solved = bool(baseline.solved and planned.solved)
    baseline_execs = baseline.executions
    planned_execs = planned.executions
    if both_solved:
        saved = baseline_execs - planned_execs
        saved_frac = saved / baseline_execs if baseline_execs > 0 else 0.0
    else:
        saved = 0
        saved_frac = 0.0
    return {
        "both_solved": both_solved,
        "baseline_execs": baseline_execs,
        "planned_execs": planned_execs,
        "saved": saved,
        "saved_frac": saved_frac,
    }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def evaluate_planning(tasks: Iterable[GoalTask], config: Config | None = None, *,
                      baseline_kw: dict, planner_kw: dict) -> dict:
    """Run ``vm_search`` and ``beam_plan`` over ``tasks`` and aggregate the metric.

    ``baseline_kw`` / ``planner_kw`` are forwarded to :func:`vm_search` /
    :func:`beam_plan` respectively (e.g. ``max_executions``, ``strategy``;
    ``scorer``, ``beam_width``, ``max_depth``, ``max_executions``). Per task,
    ``config`` falls back to the task's own ``config`` when ``None``. The returned
    dict reports, for each method, how many of the ``n`` tasks it solved, its
    success rate, mean executions (over all tasks and over its solved tasks), and
    the mean ``saved_frac`` over the tasks BOTH methods solved.
    """
    tasks = list(tasks)
    base_results: list[SearchResult] = []
    plan_results: list[SearchResult] = []
    saved_fracs: list[float] = []

    for task in tasks:
        cfg = config or task.config
        b = vm_search(task, cfg, **baseline_kw)
        p = beam_plan(task, cfg, **planner_kw)
        base_results.append(b)
        plan_results.append(p)
        sv = executions_saved(b, p)
        if sv["both_solved"]:
            saved_fracs.append(sv["saved_frac"])

    n = len(tasks)
    b_solved = sum(1 for r in base_results if r.solved)
    p_solved = sum(1 for r in plan_results if r.solved)
    return {
        "num_tasks": n,
        "baseline_solved": b_solved,
        "planned_solved": p_solved,
        "baseline_success_rate": (b_solved / n) if n else 0.0,
        "planned_success_rate": (p_solved / n) if n else 0.0,
        "baseline_mean_execs": _mean([r.executions for r in base_results]),
        "planned_mean_execs": _mean([r.executions for r in plan_results]),
        "baseline_mean_execs_solved":
            _mean([r.executions for r in base_results if r.solved]),
        "planned_mean_execs_solved":
            _mean([r.executions for r in plan_results if r.solved]),
        "num_both_solved": len(saved_fracs),
        "mean_saved_frac": _mean(saved_fracs),
    }
