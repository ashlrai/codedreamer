"""Tests for the M3 partial-program / missing-input task regime (regime #2).

Load-bearing properties:
* ``make_partial_task`` builds FORALL tasks the BASE program fails over the input
  distribution but whose ``solution_plan`` satisfies the quantified goal across
  ALL enumerated inputs — a solution exists by construction.
* ``enumerate_inputs`` returns the full Cartesian product of the input domain.
* ``evaluate_candidate_vm`` runs the VM exactly once per input assignment, so its
  reported execution count equals ``|input domain|``.
* ``vm_partial_search`` solves a 1-edit FORALL task, its executions are an exact
  multiple of the per-candidate enumeration cost, and it honours the hard cap.
"""

import math
import random

from execwm.plan.goal_tasks import satisfies
from execwm.plan.partial_search import vm_partial_search
from execwm.plan.partial_tasks import (PartialTask, Quantifier,
                                       enumerate_inputs, evaluate_candidate_vm,
                                       make_partial_task)
from execwm.plan.search_baseline import SearchResult
from execwm.substrate.edits import apply_edit
from execwm.substrate.generators import GenSpec
from execwm.substrate.vm import run_traced

# Small, fast spec: short straight-line-ish programs, no heap (fewer traps),
# arithmetic without DIV/MOD so candidate edits rarely trap on a zero input.
def _spec() -> GenSpec:
    from execwm.substrate.vm import Op
    return GenSpec(num_vars=3, num_inputs=2, num_temps=8, max_depth=1,
                   num_stmts=3, max_const=5, max_input_val=5, max_loop_count=3,
                   use_heap=False, arith_ops=(Op.ADD, Op.SUB, Op.MUL))


def _task(seed: int, edit_budget: int, domain_radius: int = 2) -> PartialTask:
    return make_partial_task(random.Random(seed), _spec(),
                             edit_budget=edit_budget,
                             domain_radius=domain_radius)


# ---------------------------------------------------------------------------
# Construction: base fails forall, solution_plan satisfies forall
# ---------------------------------------------------------------------------

def test_make_partial_task_forall_solution_by_construction():
    for seed in range(10):
        task = _task(seed, edit_budget=1)
        assert task.quantifier is Quantifier.FORALL
        inputs = enumerate_inputs(task)
        assert len(inputs) == task.domain_size

        # The edited (solution) program satisfies the goal on EVERY input.
        solved = list(task.base_bytecode)
        for edit in task.solution_plan:
            solved = apply_edit(solved, edit)
        edited_pass = [satisfies(task.goal,
                                 run_traced(solved, init, max_steps=task.max_steps))
                       for init in inputs]
        assert all(edited_pass), f"solution not forall on seed {seed}"

        # The base program FAILS forall (at least one input violates the goal).
        base_pass = [satisfies(task.goal,
                               run_traced(task.base_bytecode, init,
                                          max_steps=task.max_steps))
                     for init in inputs]
        assert not all(base_pass), f"base already forall on seed {seed}"

        assert 1 <= len(task.solution_plan) <= task.edit_budget


def test_enumerate_inputs_is_full_cartesian_product():
    task = _task(1, edit_budget=1, domain_radius=2)
    # 2 unbound inputs x 5 values each = 25 assignments.
    assert task.unbound_inputs == ("v0", "v1")
    assert task.domain_size == 25
    inputs = enumerate_inputs(task)
    assert len(inputs) == 25
    seen = {(st.regs["v0"], st.regs["v1"]) for st in inputs}
    expected = {(a, b) for a in range(-2, 3) for b in range(-2, 3)}
    assert seen == expected
    # The product size is the product of per-input domain sizes.
    prod = math.prod(len(task.input_domain[r]) for r in task.unbound_inputs)
    assert prod == task.domain_size


# ---------------------------------------------------------------------------
# evaluate_candidate_vm: one VM run per input assignment
# ---------------------------------------------------------------------------

def test_evaluate_candidate_vm_execution_count_equals_domain():
    task = _task(2, edit_budget=1)
    n = task.domain_size
    # The base program: scored over all inputs, executions == |input domain|.
    holds_base, execs_base = evaluate_candidate_vm(task.base_bytecode, task)
    assert execs_base == n
    assert holds_base is False  # base fails the quantified goal by construction

    # The solution program satisfies the quantified goal, same execution count.
    solved = list(task.base_bytecode)
    for edit in task.solution_plan:
        solved = apply_edit(solved, edit)
    holds_sol, execs_sol = evaluate_candidate_vm(solved, task)
    assert execs_sol == n
    assert holds_sol is True


# ---------------------------------------------------------------------------
# vm_partial_search: brute-force baseline
# ---------------------------------------------------------------------------

def test_vm_partial_search_solves_one_edit_forall_task():
    task = _task(1, edit_budget=1)
    n = task.domain_size
    res = vm_partial_search(task, max_executions=1_000_000)
    assert res.solved
    assert res.plan is not None and 1 <= len(res.plan) <= task.edit_budget
    # Every candidate cost is exactly |input domain|, so total is a multiple of it.
    assert res.executions > 0
    assert res.executions % n == 0
    # candidates_tried * inputs_per_candidate == reported executions.
    assert res.executions == (res.executions // n) * n
    # The reported plan really satisfies the quantified goal on the VM.
    holds, _ = evaluate_candidate_vm(
        _apply(task.base_bytecode, res.plan), task)
    assert holds


def test_vm_partial_search_respects_execution_cap():
    task = _task(1, edit_budget=1)
    # Cap below one candidate's enumeration cost -> nothing can be run.
    res = vm_partial_search(task, max_executions=task.domain_size - 1)
    assert isinstance(res, SearchResult)
    assert res.solved is False
    assert res.executions == 0
    assert res.plan is None


def test_vm_partial_search_executions_never_exceed_cap():
    task = _task(3, edit_budget=2)
    cap = 5 * task.domain_size + 1  # room for exactly 5 candidates
    res = vm_partial_search(task, max_executions=cap)
    assert res.executions <= cap
    assert res.executions % task.domain_size == 0


def _apply(program, plan):
    prog = list(program)
    for edit in plan:
        prog = apply_edit(prog, edit)
    return prog
