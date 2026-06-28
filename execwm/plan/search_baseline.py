"""No-world-model search baseline: brute-force the edit space by running the VM.

This is the control the planner (:mod:`execwm.plan.planner`) — and, later, a
learned edit-conditioned world model — must beat. It searches sequences of edits
(depth up to the task's ``edit_budget``) and evaluates **every** candidate by
actually executing it on the VM. The cost we count and report is exactly that:
the number of real ``run_traced`` evaluations performed. ``best_first`` may use
the VM's own output (goal distance) to order exploration, but it still has no
*model* that predicts an outcome without running — every candidate is run.
"""

from __future__ import annotations

import heapq
import itertools
import random
from collections import deque
from dataclasses import dataclass

from ..substrate.edits import (Edit, EditConfig, EditError, apply_edit,
                               enumerate_valid_edits)
from ..substrate.vm import Config, Instr, MachineState, Trace, VMError, run_traced
from .goal_tasks import GoalTask, goal_distance, satisfies


@dataclass(frozen=True)
class SearchResult:
    """Outcome of a search: whether it solved, real VM executions spent, the
    solving edit sequence (or ``None``), and the depth at which it stopped."""

    solved: bool
    executions: int
    plan: list[Edit] | None
    depth: int


def _safe_run(program: list[Instr], init: MachineState,
              max_steps: int) -> Trace | None:
    """Run the VM, returning ``None`` if the program traps on an undefined read."""
    try:
        return run_traced(program, init, max_steps=max_steps)
    except VMError:
        return None


def vm_search(task: GoalTask, config: Config | None = None, *,
              max_executions: int, strategy: str = "bfs",
              beam: int | None = None,
              rng: random.Random | None = None) -> SearchResult:
    """Search edit sequences (depth ``1..edit_budget``), running the VM on each.

    ``strategy`` is ``"bfs"`` (level-by-level breadth-first) or ``"best_first"``
    (a priority queue ordered by the VM-measured goal distance of each candidate;
    ``beam`` optionally bounds the live queue to its ``beam`` best nodes). Every
    candidate program is executed exactly once; duplicate programs are memoised so
    they are not re-run. ``executions`` is the count of real ``run_traced`` calls
    and ``max_executions`` is a hard cap (once reached, returns ``solved=False``).
    """
    config = config or task.config
    init = task.init_state
    goal = task.goal
    edit_cfg = task.edit_config
    max_steps = task.max_steps
    budget = task.edit_budget

    base = list(task.base_bytecode)
    seen: set[tuple[Instr, ...]] = {tuple(base)}
    executions = 0

    def expand(prog: list[Instr]):
        for edit in enumerate_valid_edits(prog, config, rng, edit_cfg):
            try:
                child = apply_edit(prog, edit)
            except EditError:
                continue
            sig = tuple(child)
            if sig in seen:
                continue
            seen.add(sig)
            yield edit, child

    if strategy == "bfs":
        frontier: deque[tuple[list[Instr], tuple[Edit, ...]]] = deque(
            [(base, ())])
        for depth in range(1, budget + 1):
            nxt: list[tuple[list[Instr], tuple[Edit, ...]]] = []
            while frontier:
                prog, plan = frontier.popleft()
                for edit, child in expand(prog):
                    if executions >= max_executions:
                        return SearchResult(False, executions, None, depth)
                    executions += 1
                    trace = _safe_run(child, init, max_steps)
                    cplan = plan + (edit,)
                    if trace is not None and satisfies(goal, trace):
                        return SearchResult(True, executions, list(cplan), depth)
                    nxt.append((child, cplan))
            frontier = deque(nxt)
        return SearchResult(False, executions, None, budget)

    if strategy == "best_first":
        tie = itertools.count()
        # (priority, tiebreak, program, plan, depth)
        heap: list[tuple[float, int, list[Instr], tuple[Edit, ...], int]] = [
            (0.0, next(tie), base, (), 0)]
        while heap:
            _, _, prog, plan, depth = heapq.heappop(heap)
            if depth >= budget:
                continue
            for edit, child in expand(prog):
                if executions >= max_executions:
                    return SearchResult(False, executions, None, depth + 1)
                executions += 1
                trace = _safe_run(child, init, max_steps)
                cplan = plan + (edit,)
                if trace is not None and satisfies(goal, trace):
                    return SearchResult(True, executions, list(cplan), depth + 1)
                if depth + 1 < budget:
                    dist = (goal_distance(goal, trace)
                            if trace is not None else float("inf"))
                    heapq.heappush(
                        heap, (dist, next(tie), child, cplan, depth + 1))
            if beam is not None and len(heap) > beam:
                heap = heapq.nsmallest(beam, heap)
                heapq.heapify(heap)
        return SearchResult(False, executions, None, budget)

    raise ValueError(f"unknown strategy {strategy!r}")
