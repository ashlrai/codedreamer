"""Partial-program / missing-input tasks for the M3 planning harness (regime #2).

This is the "the VM literally cannot judge a candidate without enumerating
completions" regime from ``PLAN_M3.md`` section 4. A :class:`PartialTask` carries
a base bytecode program together with a set of **unbound input registers** whose
values are NOT fixed — each ranges over a small finite domain. The goal is a
*quantified* predicate over the whole input distribution (see :class:`Quantifier`):

  * ``FORALL`` — the per-input :class:`~execwm.plan.goal_tasks.Goal` must hold for
    **every** input assignment in the domain;
  * ``FRAC``   — it must hold for at least a target fraction ``p`` of assignments.

Because the predicate is quantified over inputs, the VM oracle cannot score a
candidate edit with a single run: it must enumerate every input assignment and
execute the candidate on each. That per-candidate enumeration cost is the whole
point of this regime — it is exactly the work a world model that reasons over the
input distribution in latent space could amortize. This module builds the
benchmark (task generator) and the no-WM brute-force baseline; it does NOT build
the world model.

Cost structure (the explicit R4 lever)
--------------------------------------
Let ``I = |input domain|`` = the number of enumerated input assignments
(``prod(len(domain[r]) for r in unbound_inputs)``), ``b`` = the per-step edit
branching factor, and ``d`` = the edit-sequence depth (``<= edit_budget``). Then:

  * per-candidate VM cost           = ``I`` executions (one full run per input);
  * brute-force VM cost (this file) = ``(#candidates) * I`` executions, with
    ``#candidates`` up to ``b + b^2 + ... + b^d`` (the enumerated edit tree).

A latent world model would score each candidate over the input distribution
WITHOUT running the VM, paying ``O(1)`` model calls instead of ``I`` VM runs per
candidate — i.e. the saving this benchmark is designed to later expose. Tasks are
constructed so a solution provably exists (recorded in ``solution_plan``) and the
base program does not satisfy the quantified goal; construction is symbolic only.
"""

from __future__ import annotations

import enum
import itertools
import random
from dataclasses import dataclass

from ..substrate.edits import (Edit, EditConfig, EditError, apply_edit,
                               sample_edit)
from ..substrate.generators import GenSpec, make_example
from ..substrate.vm import (Config, Instr, MachineState, Trace, VMError,
                            run_traced)
from .goal_tasks import Goal, GoalKind, satisfies


class Quantifier(enum.Enum):
    """How a per-input :class:`Goal` is lifted to the whole input distribution."""

    FORALL = 0  # the goal must hold for every enumerated input assignment
    FRAC = 1    # the goal must hold for >= ``target_frac`` of input assignments


@dataclass(frozen=True)
class PartialTask:
    """A partial-program planning task with unbound inputs and a quantified goal.

    The base program does NOT satisfy ``(goal, quantifier, target_frac)`` over the
    enumerated input domain, but ``solution_plan`` (length ``1..edit_budget``)
    applied to ``base_bytecode`` yields a program that DOES — a constructive proof
    the task is solvable within budget. ``unbound_inputs`` lists the input register
    names whose values are free; ``input_domain`` gives each one's finite candidate
    values. ``fixed_regs`` are the input registers held at a constant value and
    ``heap`` is the (shared) initial heap; together with one assignment of the
    unbound inputs they form a concrete :class:`MachineState` (see
    :func:`enumerate_inputs`).
    """

    base_bytecode: list[Instr]
    unbound_inputs: tuple[str, ...]
    input_domain: dict[str, tuple[int, ...]]
    fixed_regs: dict[str, int]
    heap: list[list[int]]
    goal: Goal
    quantifier: Quantifier
    target_frac: float
    config: Config
    edit_budget: int
    max_steps: int
    solution_plan: tuple[Edit, ...]
    edit_config: EditConfig | None = None

    @property
    def domain_size(self) -> int:
        """``I`` — the number of input assignments (per-candidate VM cost)."""
        size = 1
        for reg in self.unbound_inputs:
            size *= len(self.input_domain[reg])
        return size


def enumerate_inputs(task: PartialTask) -> list[MachineState]:
    """All init states over the unbound-input domain (the Cartesian product).

    Returns ``task.domain_size`` fresh :class:`MachineState` s — one per assignment
    of the unbound inputs — each combining ``fixed_regs`` and the assignment over a
    copy of ``heap``. Domains are kept small (e.g. 2 inputs x 5 values = 25) so the
    full product is tractable to enumerate in tests.
    """
    domains = [task.input_domain[r] for r in task.unbound_inputs]
    states: list[MachineState] = []
    for combo in itertools.product(*domains):
        regs = dict(task.fixed_regs)
        regs.update(dict(zip(task.unbound_inputs, combo)))
        states.append(task.config.initial_state(regs=regs, heap=task.heap))
    return states


# ---------------------------------------------------------------------------
# Quantified-goal evaluation
# ---------------------------------------------------------------------------


def _quantifier_holds(passes: list[bool], quantifier: Quantifier,
                      target_frac: float) -> bool:
    """Does the per-input pass/fail vector satisfy the quantifier?"""
    if not passes:
        return False
    if quantifier is Quantifier.FORALL:
        return all(passes)
    if quantifier is Quantifier.FRAC:
        return (sum(passes) / len(passes)) >= target_frac
    raise ValueError(f"unknown quantifier {quantifier!r}")


def _run(program: list[Instr], init: MachineState,
         max_steps: int) -> Trace | None:
    """Run the VM, returning ``None`` if the program traps on an undefined read."""
    try:
        return run_traced(program, init, max_steps=max_steps)
    except VMError:
        return None


def _input_passes(goal: Goal, program: list[Instr], init: MachineState,
                  max_steps: int) -> bool:
    """Does ``program`` satisfy ``goal`` on this single input assignment?

    A program that traps on an undefined read counts as NOT satisfying the goal.
    """
    trace = _run(program, init, max_steps)
    return trace is not None and satisfies(goal, trace)


def evaluate_candidate_vm(program: list[Instr], task: PartialTask,
                          config: Config | None = None) -> tuple[bool, int]:
    """Score one candidate against the quantified goal, counting VM executions.

    Enumerates the full input domain and runs ``program`` on EVERY assignment
    (no short-circuit), returning ``(satisfies_quantified_goal, num_executions)``
    where ``num_executions == task.domain_size`` — the per-candidate enumeration
    cost that makes this regime expensive for a no-WM baseline. ``config`` is
    accepted for symmetry with the rest of the planning API; the VM needs only the
    program and the enumerated init states.
    """
    inputs = enumerate_inputs(task)
    passes = [_input_passes(task.goal, program, init, task.max_steps)
              for init in inputs]
    holds = _quantifier_holds(passes, task.quantifier, task.target_frac)
    return holds, len(inputs)


# ---------------------------------------------------------------------------
# Task construction
# ---------------------------------------------------------------------------


def _domain_for(rng: random.Random, domain_radius: int) -> tuple[int, ...]:
    """A small symmetric finite domain ``[-domain_radius, domain_radius]``."""
    return tuple(range(-domain_radius, domain_radius + 1))


def _candidate_goals(rng: random.Random, config: Config,
                     edited_traces: list[Trace | None]) -> list[Goal]:
    """Quantified-goal candidates derived from the edited program's outcomes.

    Builds REG_EQUALS / REG_COMPARE goals around the register values the edited
    program produces across inputs (plus a HALTS_OK candidate). Each is only a
    *candidate* — the caller verifies which ones the edited program satisfies
    under the quantifier while the base program does not.
    """
    goals: list[Goal] = [Goal(GoalKind.HALTS_OK)]
    for reg in config.reg_names:
        vals: list[int] = []
        for tr in edited_traces:
            if tr is None:
                continue
            v = tr.final_state.regs.get(reg)
            if v is not None:
                vals.append(v)
        if not vals:
            continue
        for v in sorted(set(vals)):
            goals.append(Goal(GoalKind.REG_EQUALS, reg=reg, value=v))
        thresholds = sorted({v + d for v in vals for d in (-1, 0, 1)})
        for t in thresholds:
            goals.append(Goal(GoalKind.REG_COMPARE, reg=reg, op=">", value=t))
            goals.append(Goal(GoalKind.REG_COMPARE, reg=reg, op="<", value=t))
    rng.shuffle(goals)
    return goals


def make_partial_task(rng: random.Random, spec: GenSpec,
                      codec_cfg=None,
                      edit_budget: int = 2,
                      quantifier: Quantifier = Quantifier.FORALL,
                      target_frac: float = 1.0,
                      num_unbound: int | None = None,
                      domain_radius: int = 2,
                      edit_config: EditConfig | None = None,
                      max_attempts: int = 2000) -> PartialTask:
    """Build one solvable partial task by construction (see module docstring).

    Samples a base program with unbound inputs, applies ``k = randint(1,
    edit_budget)`` structurally-valid edits, enumerates the full input domain, runs
    the VM on the base and the edited program across all assignments, and derives a
    quantified goal that the EDITED program satisfies under ``quantifier`` /
    ``target_frac`` but the BASE program does not. The constructive proof is
    recorded in ``solution_plan``. Drops/retries samples where an edit cannot be
    sampled or no discriminating quantified goal exists. ``codec_cfg`` is accepted
    for API symmetry with :func:`~execwm.plan.goal_tasks.make_goal_task` (states
    here are not codec-encoded). Raises ``RuntimeError`` if no task is found within
    ``max_attempts``.
    """
    config = spec.config()
    n_unbound = spec.num_inputs if num_unbound is None else num_unbound
    if not (1 <= n_unbound <= spec.num_inputs):
        raise ValueError(
            f"num_unbound must be in [1, {spec.num_inputs}], got {n_unbound}")
    unbound = tuple(f"v{i}" for i in range(n_unbound))
    input_domain = {reg: _domain_for(rng, domain_radius) for reg in unbound}

    for _ in range(max_attempts):
        ex = make_example(rng, spec)
        base = ex.bytecode
        # Inputs NOT made unbound are held fixed at their sampled value.
        fixed_regs = {f"v{i}": ex.init_state.regs[f"v{i}"]
                      for i in range(n_unbound, spec.num_inputs)}
        heap = ex.init_state.heap

        # Apply k structurally-valid edits to obtain the target program.
        k = rng.randint(1, max(1, edit_budget))
        prog = base
        plan: list[Edit] = []
        ok = True
        for _ in range(k):
            edit = sample_edit(prog, config, rng, edit_config)
            if edit is None:
                ok = False
                break
            try:
                prog = apply_edit(prog, edit)
            except EditError:
                ok = False
                break
            plan.append(edit)
        if not ok or not plan:
            continue

        task_stub = PartialTask(
            base_bytecode=base, unbound_inputs=unbound,
            input_domain=input_domain, fixed_regs=fixed_regs, heap=heap,
            goal=Goal(GoalKind.HALTS_OK), quantifier=quantifier,
            target_frac=target_frac, config=config, edit_budget=edit_budget,
            max_steps=spec.max_steps, solution_plan=tuple(plan),
            edit_config=edit_config)
        inputs = enumerate_inputs(task_stub)
        base_traces = [_run(base, init, spec.max_steps) for init in inputs]
        edited_traces = [_run(prog, init, spec.max_steps) for init in inputs]

        for goal in _candidate_goals(rng, config, edited_traces):
            edited_pass = [tr is not None and satisfies(goal, tr)
                           for tr in edited_traces]
            if not _quantifier_holds(edited_pass, quantifier, target_frac):
                continue
            base_pass = [tr is not None and satisfies(goal, tr)
                         for tr in base_traces]
            if _quantifier_holds(base_pass, quantifier, target_frac):
                continue  # trivial: base already satisfies the quantified goal
            return PartialTask(
                base_bytecode=base, unbound_inputs=unbound,
                input_domain=input_domain, fixed_regs=fixed_regs, heap=heap,
                goal=goal, quantifier=quantifier, target_frac=target_frac,
                config=config, edit_budget=edit_budget,
                max_steps=spec.max_steps, solution_plan=tuple(plan),
                edit_config=edit_config)

    raise RuntimeError(
        f"could not construct a partial task in {max_attempts} attempts")
