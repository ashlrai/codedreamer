"""M3 step-2 planning harness: goal tasks, baselines, and the executions-saved metric.

This package implements *step 2* of the M3 build order (see ``PLAN_M3.md`` §6):
calibrate risk **R4** — *is there ANY real-VM-execution saving to be had from
planning over program edits* — BEFORE investing in a learned, edit-conditioned
world model (step 3).

Crucially, there is **no neural network here**. The "world model" in this step is
a *symbolic stand-in*: the planner scores candidate edits with a pluggable scorer
(an exact VM oracle as an upper bound, or a cheap structural heuristic that does
not run the full VM) and only spends real ``run_traced`` executions to *verify*
the candidates it commits to. The no-world-model baseline (:mod:`search_baseline`)
brute-forces the edit space by actually running the VM on every candidate. The
metric (:mod:`metrics`) reports executions-to-solution at matched success — the
number a learned model in step 3 must later beat.
"""

from __future__ import annotations

from .goal_tasks import (Goal, GoalKind, GoalTask, goal_distance, make_goal_task,
                         satisfies)
from .metrics import evaluate_planning, executions_saved
from .planner import OracleScorer, beam_plan, cheap_scorer, oracle_scorer
from .search_baseline import SearchResult, vm_search

__all__ = [
    "Goal", "GoalKind", "GoalTask", "goal_distance", "make_goal_task", "satisfies",
    "SearchResult", "vm_search",
    "OracleScorer", "beam_plan", "cheap_scorer", "oracle_scorer",
    "evaluate_planning", "executions_saved",
]
