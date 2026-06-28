"""Goal predicates and a goal-task generator for the M3 planning harness.

A :class:`Goal` is an exact predicate over the *outcome* of running a program
(its final machine state / trace), checked against the VM oracle. A
:class:`GoalTask` pairs a base program + inputs with a goal that the BASE program
does **not** satisfy but that is reachable by applying ``1..edit_budget`` valid
:class:`~execwm.substrate.edits.Edit` s. Tasks are built *constructively* — sample
a base program, apply ``k`` random valid edits, run the VM to read off the
resulting state, and set the goal to a property that the edited state achieves and
the base does not — so a solution provably exists (it is recorded as
``solution_plan``). This is symbolic only; no neural model is involved (see the
package docstring on R4 calibration).
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass

from ..data.state_codec import CodecConfig, EncodeError, StateCodec
from ..substrate.edits import (Edit, EditConfig, EditError, apply_edit,
                               sample_edit)
from ..substrate.generators import GenSpec, make_example
from ..substrate.vm import Config, Instr, MachineState, Trace, VMError, run_traced


class GoalKind(enum.Enum):
    """The supported goal predicates. Values are stable ids."""

    REG_EQUALS = 0    # regs[reg] == value
    REG_COMPARE = 1   # regs[reg] <op> value, op in {">", "<", "=="}
    HALTS_OK = 2      # program terminates (halt / fall-off-end) without a trap


@dataclass(frozen=True)
class Goal:
    """An exact, VM-checkable goal predicate.

    Only the fields relevant to ``kind`` are populated:
      REG_EQUALS   -> ``reg``, ``value``
      REG_COMPARE  -> ``reg``, ``op`` ('>'|'<'|'=='), ``value``
      HALTS_OK     -> (no fields)
    """

    kind: GoalKind
    reg: str | None = None
    value: int | None = None
    op: str | None = None


def _resolve(obj: Trace | MachineState) -> tuple[MachineState, bool, bool]:
    """Normalise a Trace or MachineState to ``(final_state, terminated, error)``."""
    if isinstance(obj, Trace):
        st = obj.final_state
        return st, obj.terminated, st.error
    return obj, obj.halted, obj.error


def goal_distance(goal: Goal, obj: Trace | MachineState) -> float:
    """A non-negative distance to ``goal`` (``0.0`` iff satisfied; lower = closer).

    Used both as the exact satisfaction check (``== 0``) and as a planning score.
    An undefined target register is ``inf`` (unreachable as scored).
    """
    state, terminated, error = _resolve(obj)
    if goal.kind is GoalKind.HALTS_OK:
        return 0.0 if (terminated and not error) else 1.0

    val = state.regs.get(goal.reg)
    if val is None:
        return float("inf")
    if goal.kind is GoalKind.REG_EQUALS:
        return float(abs(val - goal.value))
    # REG_COMPARE
    if goal.op == ">":
        return 0.0 if val > goal.value else float(goal.value - val + 1)
    if goal.op == "<":
        return 0.0 if val < goal.value else float(val - goal.value + 1)
    if goal.op == "==":
        return float(abs(val - goal.value))
    raise ValueError(f"unknown compare op {goal.op!r}")


def satisfies(goal: Goal, obj: Trace | MachineState) -> bool:
    """Exact check: does ``obj`` (a final state or a full trace) satisfy ``goal``?"""
    return goal_distance(goal, obj) == 0.0


@dataclass(frozen=True)
class GoalTask:
    """A planning task: hit ``goal`` by editing ``base_bytecode`` (from ``init_state``).

    The base program does NOT satisfy ``goal``; ``solution_plan`` is a known
    edit sequence (length ``<= edit_budget``) that, applied in order to
    ``base_bytecode``, yields a program that DOES — i.e. a constructive proof the
    task is solvable within budget. Searchers need only ``config`` / ``edit_config``
    / ``max_steps`` to reproduce the same edit space and VM behaviour.
    """

    base_bytecode: list[Instr]
    init_state: MachineState
    goal: Goal
    config: Config
    edit_budget: int
    max_steps: int
    solution_plan: tuple[Edit, ...]
    edit_config: EditConfig | None = None


def _derive_goal(rng: random.Random, base_trace: Trace,
                 edited_trace: Trace) -> Goal | None:
    """Pick a goal the edited trace satisfies and the base trace does not.

    Looks for a register that is INT-defined in the edited final state with a
    value differing from the base final state's value for that register
    (including the base being undefined), and builds a REG_EQUALS or (when the
    base value is defined) a REG_COMPARE goal around it.
    """
    base_final = base_trace.final_state
    edited_final = edited_trace.final_state
    cands: list[tuple[str, int, int | None]] = []
    for reg, v in edited_final.regs.items():
        if v is None:
            continue
        bv = base_final.regs.get(reg)
        if bv != v:
            cands.append((reg, v, bv))
    if not cands:
        return None

    reg, v, bv = rng.choice(cands)
    if bv is not None and rng.random() < 0.5:
        # A comparison the edited value passes and the base value fails.
        if v > bv:
            return Goal(GoalKind.REG_COMPARE, reg=reg, op=">", value=bv)
        return Goal(GoalKind.REG_COMPARE, reg=reg, op="<", value=bv)
    return Goal(GoalKind.REG_EQUALS, reg=reg, value=v)


def make_goal_task(rng: random.Random, spec: GenSpec,
                   codec_cfg: CodecConfig | None = None,
                   edit_budget: int = 2,
                   edit_config: EditConfig | None = None,
                   max_attempts: int = 500) -> GoalTask:
    """Build one solvable goal task by construction (see module docstring).

    Samples a base program + inputs, applies ``k = randint(1, edit_budget)``
    structurally-valid edits, runs the VM to read the achieved state, and derives
    a goal the edited program meets and the base does not. Drops/retries samples
    where: an edit could not be sampled, the edited program traps on an undefined
    read, no distinguishing goal exists (the edits changed nothing observable), or
    (when ``codec_cfg`` is given) a final state is not codec-encodable. Raises
    ``RuntimeError`` if no task is found within ``max_attempts``.
    """
    config = spec.config()
    scodec = StateCodec(config, codec_cfg) if codec_cfg is not None else None

    for _ in range(max_attempts):
        ex = make_example(rng, spec)
        base_trace = ex.trace

        k = rng.randint(1, max(1, edit_budget))
        prog = ex.bytecode
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

        try:
            edited_trace = run_traced(prog, ex.init_state, max_steps=spec.max_steps)
        except VMError:
            continue  # edited program reads an undefined register on these inputs

        goal = _derive_goal(rng, base_trace, edited_trace)
        if goal is None:
            continue
        # The task must be non-trivial (base fails) and solvable (edited passes).
        if satisfies(goal, base_trace) or not satisfies(goal, edited_trace):
            continue
        if scodec is not None:
            try:
                scodec.encode(base_trace.final_state)
                scodec.encode(edited_trace.final_state)
            except EncodeError:
                continue

        return GoalTask(
            base_bytecode=ex.bytecode,
            init_state=ex.init_state,
            goal=goal,
            config=config,
            edit_budget=edit_budget,
            max_steps=spec.max_steps,
            solution_plan=tuple(plan),
            edit_config=edit_config,
        )

    raise RuntimeError(
        f"could not construct a goal task in {max_attempts} attempts")
