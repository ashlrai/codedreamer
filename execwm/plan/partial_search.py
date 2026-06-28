"""No-world-model brute-force baseline for partial-program tasks (regime #2).

This is the control a latent world model must beat in the "missing inputs" regime
(``PLAN_M3.md`` section 4, #2). It searches sequences of edits (depth up to the
task's ``edit_budget``) and scores **every** candidate by enumerating the full
input domain and running the VM on each assignment via
:func:`~execwm.plan.partial_tasks.evaluate_candidate_vm`. The cost we report is
exactly that: the total number of real ``run_traced`` executions, which is

    executions = (#candidates evaluated) * I

where ``I = task.domain_size`` is the per-candidate enumeration cost. This makes
the cost explosion explicit — ``branching_factor^depth * |input domain|`` — and is
precisely the work a world model that scores over the input distribution in latent
space could amortize. ``max_executions`` is a hard cap: a candidate is only run if
its full ``I``-execution enumeration fits under the cap, so ``executions`` is
always an exact multiple of ``I`` and never exceeds the cap.
"""

from __future__ import annotations

import random
from collections import deque

from ..substrate.edits import (Edit, EditError, apply_edit,
                               enumerate_valid_edits)
from ..substrate.vm import Config, Instr
from .partial_tasks import PartialTask, evaluate_candidate_vm
from .search_baseline import SearchResult


def vm_partial_search(task: PartialTask, config: Config | None = None, *,
                      max_executions: int,
                      rng: random.Random | None = None) -> SearchResult:
    """Breadth-first edit search; each candidate is scored by enumerating inputs.

    Expands edit sequences level by level (depth ``1..edit_budget``). For each
    distinct candidate program it calls :func:`evaluate_candidate_vm`, which runs
    the VM on all ``I = task.domain_size`` input assignments and reports whether the
    quantified goal holds. ``executions`` accumulates those per-candidate costs and
    ``max_executions`` is a hard cap (a candidate whose ``I`` runs would breach the
    cap is not run, and the search returns ``solved=False``). Returns the first
    edit sequence whose program satisfies the quantified goal.
    """
    config = config or task.config
    n_inputs = task.domain_size
    base = list(task.base_bytecode)
    seen: set[tuple[Instr, ...]] = {tuple(base)}
    executions = 0

    def expand(prog: list[Instr]):
        for edit in enumerate_valid_edits(prog, config, rng, task.edit_config):
            try:
                child = apply_edit(prog, edit)
            except EditError:
                continue
            sig = tuple(child)
            if sig in seen:
                continue
            seen.add(sig)
            yield edit, child

    frontier: deque[tuple[list[Instr], tuple[Edit, ...]]] = deque([(base, ())])
    for depth in range(1, task.edit_budget + 1):
        nxt: list[tuple[list[Instr], tuple[Edit, ...]]] = []
        while frontier:
            prog, plan = frontier.popleft()
            for edit, child in expand(prog):
                if executions + n_inputs > max_executions:
                    return SearchResult(False, executions, None, depth)
                holds, ran = evaluate_candidate_vm(child, task, config)
                executions += ran
                cplan = plan + (edit,)
                if holds:
                    return SearchResult(True, executions, list(cplan), depth)
                nxt.append((child, cplan))
        frontier = deque(nxt)

    return SearchResult(False, executions, None, task.edit_budget)
