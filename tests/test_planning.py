"""Tests for the M3 step-2 planning harness (goal tasks, baselines, metric).

Load-bearing properties:
* ``make_goal_task`` builds tasks the BASE program does not satisfy but a known
  ``solution_plan`` does — a solution exists by construction.
* ``vm_search`` solves a 1-edit task (executions > 0) and honours its hard
  ``max_executions`` cap (solved=False when capped to zero).
* ``beam_plan`` with the oracle scorer solves the same tasks; with a tiny beam /
  zero budget it reports failure honestly (no crash, solved=False).
* ``executions_saved`` math is correct, including the not-solved cases.
"""

import random

import pytest

from execwm.plan.goal_tasks import (Goal, GoalKind, GoalTask, make_goal_task,
                                     satisfies)
from execwm.plan.metrics import evaluate_planning, executions_saved
from execwm.plan.planner import (OracleScorer, beam_plan, cheap_scorer,
                                  oracle_scorer)
from execwm.plan.search_baseline import SearchResult, vm_search
from execwm.substrate.edits import apply_edit
from execwm.substrate.generators import GenSpec
from execwm.substrate.vm import run_traced

# Small, fast spec: short straight-line-ish programs, no heap (fewer traps).
_SPEC = GenSpec(num_vars=3, num_inputs=2, num_temps=8, max_depth=1, num_stmts=3,
                max_const=5, max_input_val=5, max_loop_count=3, use_heap=False)


def _task(seed: int, edit_budget: int) -> GoalTask:
    return make_goal_task(random.Random(seed), _SPEC, edit_budget=edit_budget)


# ---------------------------------------------------------------------------
# Goal-task construction
# ---------------------------------------------------------------------------

def test_make_goal_task_has_constructive_solution():
    for seed in range(12):
        task = _task(seed, edit_budget=2)
        # The base program does NOT satisfy the goal.
        base_trace = run_traced(task.base_bytecode, task.init_state,
                                max_steps=task.max_steps)
        assert not satisfies(task.goal, base_trace)
        # The recorded solution plan is within budget and DOES satisfy it.
        assert 1 <= len(task.solution_plan) <= task.edit_budget
        prog = list(task.base_bytecode)
        for edit in task.solution_plan:
            prog = apply_edit(prog, edit)
        solved_trace = run_traced(prog, task.init_state, max_steps=task.max_steps)
        assert satisfies(task.goal, solved_trace)


def test_satisfies_all_goal_kinds():
    task = _task(0, edit_budget=1)
    final = run_traced(task.base_bytecode, task.init_state,
                       max_steps=task.max_steps).final_state
    # HALTS_OK on a normally-terminating trace.
    trace = run_traced(task.base_bytecode, task.init_state, max_steps=task.max_steps)
    assert satisfies(Goal(GoalKind.HALTS_OK), trace)
    # REG_EQUALS / REG_COMPARE checked against the actual final state.
    reg = next(r for r, v in final.regs.items() if v is not None)
    val = final.regs[reg]
    assert satisfies(Goal(GoalKind.REG_EQUALS, reg=reg, value=val), final)
    assert not satisfies(Goal(GoalKind.REG_EQUALS, reg=reg, value=val + 1), final)
    assert satisfies(Goal(GoalKind.REG_COMPARE, reg=reg, op=">", value=val - 1), final)
    assert not satisfies(Goal(GoalKind.REG_COMPARE, reg=reg, op="<", value=val), final)


# ---------------------------------------------------------------------------
# No-world-model VM search baseline
# ---------------------------------------------------------------------------

def test_vm_search_solves_one_edit_task():
    task = _task(1, edit_budget=1)
    res = vm_search(task, max_executions=10_000, strategy="bfs")
    assert res.solved
    assert res.executions > 0
    assert res.plan is not None and len(res.plan) >= 1
    # The reported plan really solves the task on the VM.
    prog = list(task.base_bytecode)
    for edit in res.plan:
        prog = apply_edit(prog, edit)
    assert satisfies(task.goal, run_traced(prog, task.init_state,
                                           max_steps=task.max_steps))


def test_vm_search_respects_execution_cap():
    task = _task(1, edit_budget=1)
    res = vm_search(task, max_executions=0, strategy="bfs")
    assert res.solved is False
    assert res.executions == 0


def test_vm_search_best_first_also_solves():
    task = _task(2, edit_budget=2)
    res = vm_search(task, max_executions=50_000, strategy="best_first")
    assert res.solved
    assert res.executions > 0


# ---------------------------------------------------------------------------
# Re-encode beam planner
# ---------------------------------------------------------------------------

def test_beam_plan_oracle_solves():
    for seed in (1, 3, 5):
        task = _task(seed, edit_budget=1)
        res = beam_plan(task, scorer=OracleScorer(max_steps=task.max_steps),
                        beam_width=8, max_depth=task.edit_budget,
                        max_executions=10_000)
        assert res.solved, f"oracle beam failed on seed {seed}"
        assert res.executions > 0


def test_beam_plan_reports_failure_honestly():
    task = _task(1, edit_budget=2)
    # Zero execution budget -> cannot verify anything -> honest failure, no crash.
    res = beam_plan(task, scorer=cheap_scorer, beam_width=1, max_depth=2,
                    max_executions=0)
    assert isinstance(res, SearchResult)
    assert res.solved is False
    assert res.executions == 0
    assert res.plan is None


def test_beam_plan_cheap_scorer_runs_without_crash():
    # The cheap heuristic may or may not solve; it must never crash and must
    # report a well-formed result.
    for seed in range(6):
        task = _task(seed, edit_budget=2)
        res = beam_plan(task, scorer=cheap_scorer, beam_width=4, max_depth=2,
                        max_executions=200)
        assert isinstance(res, SearchResult)
        assert res.solved in (True, False)
        assert res.executions >= 0


def test_oracle_scorer_counts_its_vm_runs():
    task = _task(1, edit_budget=1)
    scorer = oracle_scorer(max_steps=task.max_steps)
    res = beam_plan(task, scorer=scorer, beam_width=4, max_depth=1,
                    max_executions=10_000)
    # The oracle scored many candidates by running the VM; those runs are folded
    # into the reported executions (so the oracle does NOT look free).
    assert scorer.executions > 0
    assert res.executions >= res.depth  # at least the verification runs counted


# ---------------------------------------------------------------------------
# executions_saved metric
# ---------------------------------------------------------------------------

def test_executions_saved_both_solved():
    base = SearchResult(solved=True, executions=100, plan=[], depth=2)
    plan = SearchResult(solved=True, executions=10, plan=[], depth=1)
    out = executions_saved(base, plan)
    assert out["both_solved"] is True
    assert out["baseline_execs"] == 100
    assert out["planned_execs"] == 10
    assert out["saved"] == 90
    assert out["saved_frac"] == pytest.approx(0.9)


def test_executions_saved_planner_failed():
    base = SearchResult(solved=True, executions=100, plan=[], depth=2)
    plan = SearchResult(solved=False, executions=50, plan=None, depth=2)
    out = executions_saved(base, plan)
    assert out["both_solved"] is False
    assert out["saved"] == 0
    assert out["saved_frac"] == 0.0


def test_executions_saved_baseline_failed():
    base = SearchResult(solved=False, executions=100, plan=None, depth=2)
    plan = SearchResult(solved=True, executions=10, plan=[], depth=1)
    out = executions_saved(base, plan)
    assert out["both_solved"] is False
    assert out["saved"] == 0
    assert out["saved_frac"] == 0.0


def test_evaluate_planning_aggregates_and_is_explicit():
    tasks = [_task(s, edit_budget=1) for s in range(6)]
    out = evaluate_planning(
        tasks,
        baseline_kw=dict(max_executions=10_000, strategy="bfs"),
        planner_kw=dict(scorer=cheap_scorer, beam_width=6, max_depth=1,
                        max_executions=200),
    )
    assert out["num_tasks"] == 6
    assert 0 <= out["planned_solved"] <= 6
    assert 0 <= out["baseline_solved"] <= 6
    assert out["num_both_solved"] <= min(out["baseline_solved"],
                                         out["planned_solved"])
    assert 0.0 <= out["mean_saved_frac"] <= 1.0
