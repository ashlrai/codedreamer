"""Re-encode beam planner with a pluggable scorer (the M3 step-2 "world model").

For step 2 the "world model" is a **symbolic stand-in**, not a neural net (the
learned edit-conditioned model is step 3). :func:`beam_plan` expands edit
sequences, scores partial candidates with a pluggable ``scorer`` (lower = closer
to the goal), keeps the top ``beam_width``, and spends real VM executions only to
*verify* the candidates it commits to — counting those verifications honestly.

Two scorers are provided so the harness can calibrate R4:

* :class:`OracleScorer` — runs the VM to score (an *upper bound* on planning
  quality). It is honest about cost: every scoring run is counted as an
  execution, so beam search with the oracle does NOT magically save executions.
* :func:`cheap_scorer` — a structural / partial-state heuristic that does NOT run
  the full VM (a single linear pass, no control flow / loops). It is cheap but
  weak; when it ranks well, ``beam_plan`` reaches the goal having run the VM only
  on a handful of verified candidates — that is the executions-saving we test.

Any VM executions a scorer performs are folded into the reported ``executions``
via the scorer's ``executions`` attribute, so no VM call is ever hidden.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..substrate.edits import EditConfig, EditError, apply_edit, enumerate_valid_edits
from ..substrate.vm import (ARITH_OPS, CMP_OPS, Config, Instr, MachineState, Op,
                            VMError, run_traced)
from .goal_tasks import Goal, GoalTask, goal_distance, satisfies
from .search_baseline import SearchResult

# A scorer: (candidate_program, init_state, goal) -> float (lower = closer).
# It MAY expose an ``executions`` int attribute counting any VM runs it performs;
# beam_plan reads that attribute to fold those costs into the reported total.


@dataclass
class OracleScorer:
    """Exact scorer that runs the VM — the upper bound on planning quality.

    Counts every scoring run in ``executions`` so beam search with the oracle
    pays the full VM cost (no hidden executions). A trapping candidate scores
    ``inf``.
    """

    max_steps: int = 256
    executions: int = 0

    def __call__(self, program: list[Instr], init_state: MachineState,
                 goal: Goal) -> float:
        self.executions += 1
        try:
            trace = run_traced(program, init_state, max_steps=self.max_steps)
        except VMError:
            return float("inf")
        return goal_distance(goal, trace)


def oracle_scorer(max_steps: int = 256) -> OracleScorer:
    """Convenience factory for a fresh :class:`OracleScorer`."""
    return OracleScorer(max_steps=max_steps)


def _aread(state: MachineState, operand) -> int | None:
    """Resolve an operand for the cheap linear pass; ``None`` if unknown/undefined."""
    if isinstance(operand, bool):  # defensive; bools are not VM operands
        return None
    if isinstance(operand, int):
        return operand
    if isinstance(operand, str):
        return state.regs.get(operand)
    return None


def _abstract_final_state(program: list[Instr],
                          init_state: MachineState) -> MachineState:
    """A cheap, single linear pass over the program ignoring control flow.

    Folds CONST/MOV/arith/comparison effects in index order from ``init_state``,
    skipping jumps, heap ops, halts, loops, and branch selection entirely. This is
    strictly cheaper than ``run_traced`` (no loop iteration, no branch following)
    and yields only an *estimate* of the final register values — deliberately weak
    on control-flow-heavy programs.
    """
    s = init_state.copy()
    for instr in program:
        op = instr.op
        if op is Op.CONST:
            s.regs[instr.dst] = int(instr.a)
        elif op is Op.MOV:
            s.regs[instr.dst] = _aread(s, instr.a)
        elif op in ARITH_OPS:
            x, y = _aread(s, instr.a), _aread(s, instr.b)
            if x is None or y is None:
                continue
            if op is Op.ADD:
                s.regs[instr.dst] = x + y
            elif op is Op.SUB:
                s.regs[instr.dst] = x - y
            elif op is Op.MUL:
                s.regs[instr.dst] = x * y
            elif op is Op.DIV:
                s.regs[instr.dst] = x // y if y != 0 else None
            elif op is Op.MOD:
                s.regs[instr.dst] = x % y if y != 0 else None
        elif op in CMP_OPS:
            x, y = _aread(s, instr.a), _aread(s, instr.b)
            if x is None or y is None:
                continue
            res = {
                Op.LT: x < y, Op.LE: x <= y, Op.EQ: x == y,
                Op.NE: x != y, Op.GT: x > y, Op.GE: x >= y,
            }[op]
            s.regs[instr.dst] = int(res)
        # JMP/JZ/JNZ/LOAD/STORE/HALT: ignored in the cheap estimate.
    return s


def cheap_scorer(program: list[Instr], init_state: MachineState,
                 goal: Goal) -> float:
    """Heuristic scorer that does NOT run the full VM (see ``_abstract_final_state``)."""
    est = _abstract_final_state(program, init_state)
    return goal_distance(goal, est)


def beam_plan(task: GoalTask, config: Config | None = None, *,
              scorer, beam_width: int, max_depth: int, max_executions: int,
              verify_k: int | None = None,
              rng: random.Random | None = None) -> SearchResult:
    """Beam search over edits, scoring with ``scorer`` and verifying on the VM.

    At each depth, every child candidate is scored (cheaply, by ``scorer``), the
    top ``beam_width`` are kept, and the best ``verify_k`` of those (default:
    ``beam_width``) are *verified* by a real VM run; a verified candidate that
    satisfies the goal ends the search. Reported ``executions`` = VM verification
    runs **plus** any VM runs the scorer performed (read from ``scorer.executions``).
    ``max_executions`` is a hard cap on that combined total.
    """
    config = config or task.config
    init = task.init_state
    goal = task.goal
    edit_cfg = task.edit_config
    max_steps = task.max_steps
    verify_k = beam_width if verify_k is None else verify_k

    scorer_base = getattr(scorer, "executions", 0)
    verifications = 0

    def total() -> int:
        return verifications + (getattr(scorer, "executions", 0) - scorer_base)

    frontier: list[tuple[list[Instr], tuple]] = [(list(task.base_bytecode), ())]
    seen: set[tuple[Instr, ...]] = {tuple(task.base_bytecode)}

    for depth in range(1, max_depth + 1):
        scored: list[tuple[float, list[Instr], tuple]] = []
        for prog, plan in frontier:
            for edit in enumerate_valid_edits(prog, config, rng, edit_cfg):
                try:
                    child = apply_edit(prog, edit)
                except EditError:
                    continue
                sig = tuple(child)
                if sig in seen:
                    continue
                seen.add(sig)
                if total() >= max_executions:
                    return SearchResult(False, total(), None, depth)
                s = scorer(child, init, goal)
                scored.append((s, child, plan + (edit,)))
        if not scored:
            break
        scored.sort(key=lambda x: x[0])
        beam = scored[:beam_width]

        for _, child, plan in beam[:verify_k]:
            if total() + 1 > max_executions:
                return SearchResult(False, total(), None, depth)
            verifications += 1
            try:
                trace = run_traced(child, init, max_steps=max_steps)
            except VMError:
                continue
            if satisfies(goal, trace):
                return SearchResult(True, total(), list(plan), depth)

        frontier = [(child, plan) for _, child, plan in beam]

    return SearchResult(False, total(), None, max_depth)
