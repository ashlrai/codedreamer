"""Register-based bytecode VM with a per-instruction tracer.

This is the ground-truth oracle for the whole project: every program executes
here, and the tracer emits the *full* machine state after each instruction. The
world model is trained and graded against these traces, so correctness and
determinism of this file are load-bearing.

Design notes
------------
* The machine state is fully observable and finite-dimensional: a fixed set of
  named scalar registers (each with a type tag) plus a fixed-shape integer heap
  (a small number of fixed-length lists) plus a program counter. This is exactly
  what makes "exact state match" well-defined (see ``execwm/data/state_codec.py``).
* Arithmetic is over Python ``int`` (unbounded) on purpose: it lets us define an
  out-of-distribution *magnitude* axis (train on small values, test on large)
  without the truncation a fixed-width machine would impose. Generation controls
  the magnitude; the codec controls how many digits are representable.
* Transitions are deterministic. There is therefore no need for a stochastic
  latent in the world model — a claimed simplification of the project.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import Union

Operand = Union[str, int]  # str -> register name, int -> immediate literal


class Op(enum.Enum):
    """Bytecode opcodes. Operand slots used are documented per-op."""

    # data movement
    CONST = enum.auto()   # dst = a (immediate)
    MOV = enum.auto()     # dst = reg(a)
    # integer arithmetic: dst = reg-or-imm(a) <op> reg-or-imm(b)
    ADD = enum.auto()
    SUB = enum.auto()
    MUL = enum.auto()
    DIV = enum.auto()     # floor division; div-by-zero -> error trap
    MOD = enum.auto()     # python modulo; mod-by-zero -> error trap
    # comparisons -> BOOL (0/1)
    LT = enum.auto()
    LE = enum.auto()
    EQ = enum.auto()
    NE = enum.auto()
    GT = enum.auto()
    GE = enum.auto()
    # control flow (target is an instruction index)
    JMP = enum.auto()     # pc = target
    JZ = enum.auto()      # if reg-or-imm(a) == 0: pc = target
    JNZ = enum.auto()     # if reg-or-imm(a) != 0: pc = target
    # fixed-shape heap (list_id, index) access
    LOAD = enum.auto()    # dst = heap[list_id][reg-or-imm(a)]
    STORE = enum.auto()   # heap[list_id][reg-or-imm(a)] = reg-or-imm(b)
    HALT = enum.auto()    # stop execution (normal termination)


# Opcode groups, handy for generators / encoders.
ARITH_OPS = (Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD)
CMP_OPS = (Op.LT, Op.LE, Op.EQ, Op.NE, Op.GT, Op.GE)
JUMP_OPS = (Op.JMP, Op.JZ, Op.JNZ)


class VType(enum.Enum):
    """Type tag for a scalar register slot. A grounding-head prediction target."""

    UNDEF = 0
    INT = 1
    BOOL = 2


@dataclass(frozen=True)
class Instr:
    """A single bytecode instruction.

    ``a`` and ``b`` are operands: a ``str`` names a register, an ``int`` is an
    immediate literal. ``dst`` is the destination register name (or ``None`` for
    stores / jumps / halt). ``list_id`` and ``target`` are used by heap and jump
    ops respectively.
    """

    op: Op
    dst: str | None = None
    a: Operand | None = None
    b: Operand | None = None
    list_id: int | None = None
    target: int | None = None


# ---------------------------------------------------------------------------
# Machine state
# ---------------------------------------------------------------------------


@dataclass
class MachineState:
    """Full, observable machine state at one instant.

    ``regs`` maps every declared register name to its value or ``None`` when
    undefined; ``types`` carries the parallel :class:`VType` tag. ``heap`` is a
    fixed list of fixed-length integer lists. The register *name set* and heap
    *shape* are fixed for a given configuration so the state has constant shape.
    """

    regs: dict[str, int | None]
    types: dict[str, VType]
    heap: list[list[int]]
    pc: int = 0
    halted: bool = False
    error: bool = False  # set on a trap (e.g. div/mod by zero, oob heap access)
    steps: int = 0

    def copy(self) -> "MachineState":
        return MachineState(
            regs=dict(self.regs),
            types=dict(self.types),
            heap=[list(cells) for cells in self.heap],
            pc=self.pc,
            halted=self.halted,
            error=self.error,
            steps=self.steps,
        )


@dataclass
class Config:
    """Shape of the machine: which registers exist and the heap geometry."""

    reg_names: tuple[str, ...]
    num_lists: int = 1
    list_len: int = 4
    max_steps: int = 256

    def initial_state(self, regs: dict[str, int] | None = None,
                      heap: list[list[int]] | None = None) -> MachineState:
        """Build a fresh state. Unspecified registers start UNDEF; the heap
        defaults to all-zero cells of the configured shape."""
        regs = regs or {}
        state_regs: dict[str, int | None] = {}
        state_types: dict[str, VType] = {}
        for name in self.reg_names:
            if name in regs:
                state_regs[name] = int(regs[name])
                state_types[name] = VType.INT
            else:
                state_regs[name] = None
                state_types[name] = VType.UNDEF
        if heap is None:
            heap = [[0] * self.list_len for _ in range(self.num_lists)]
        else:
            assert len(heap) == self.num_lists
            assert all(len(cells) == self.list_len for cells in heap)
            heap = [list(cells) for cells in heap]
        return MachineState(regs=state_regs, types=state_types, heap=heap)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


class VMError(Exception):
    """Raised on a malformed program (not a runtime trap — those set state.error)."""


def _read(state: MachineState, operand: Operand | None) -> int:
    """Resolve an operand to an int. Immediates pass through; register reads of
    an undefined register raise (generators must not emit such programs)."""
    if isinstance(operand, int):
        return operand
    if isinstance(operand, str):
        val = state.regs[operand]
        if val is None:
            raise VMError(f"read of undefined register {operand!r}")
        return val
    raise VMError(f"invalid operand {operand!r}")


def _truthy(state: MachineState, operand: Operand | None) -> bool:
    return _read(state, operand) != 0


def step(state: MachineState, instr: Instr) -> MachineState:
    """Execute one instruction, returning a *new* state (functional update).

    The input state is left untouched so callers can keep s_t and s_{t+1}
    independently for world-model training pairs.
    """
    s = state.copy()
    s.steps += 1
    op = instr.op

    def set_reg(name: str, value: int, vtype: VType) -> None:
        s.regs[name] = value
        s.types[name] = vtype

    advance = True  # whether to fall through to pc+1 (jumps override)

    if op is Op.HALT:
        s.halted = True
        advance = False

    elif op is Op.CONST:
        set_reg(instr.dst, int(instr.a), VType.INT)

    elif op is Op.MOV:
        src_val = _read(s, instr.a)
        src_type = state.types[instr.a] if isinstance(instr.a, str) else VType.INT
        set_reg(instr.dst, src_val, src_type)

    elif op in ARITH_OPS:
        x, y = _read(s, instr.a), _read(s, instr.b)
        if op is Op.ADD:
            set_reg(instr.dst, x + y, VType.INT)
        elif op is Op.SUB:
            set_reg(instr.dst, x - y, VType.INT)
        elif op is Op.MUL:
            set_reg(instr.dst, x * y, VType.INT)
        elif op is Op.DIV:
            if y == 0:
                s.error = True
                s.halted = True
                advance = False
            else:
                # floor division toward negative infinity (Python semantics)
                set_reg(instr.dst, x // y, VType.INT)
        elif op is Op.MOD:
            if y == 0:
                s.error = True
                s.halted = True
                advance = False
            else:
                set_reg(instr.dst, x % y, VType.INT)

    elif op in CMP_OPS:
        x, y = _read(s, instr.a), _read(s, instr.b)
        result = {
            Op.LT: x < y, Op.LE: x <= y, Op.EQ: x == y,
            Op.NE: x != y, Op.GT: x > y, Op.GE: x >= y,
        }[op]
        set_reg(instr.dst, int(result), VType.BOOL)

    elif op is Op.JMP:
        s.pc = instr.target
        advance = False

    elif op is Op.JZ:
        if not _truthy(s, instr.a):
            s.pc = instr.target
            advance = False

    elif op is Op.JNZ:
        if _truthy(s, instr.a):
            s.pc = instr.target
            advance = False

    elif op is Op.LOAD:
        idx = _read(s, instr.a)
        if not (0 <= idx < len(s.heap[instr.list_id])):
            s.error = True
            s.halted = True
            advance = False
        else:
            set_reg(instr.dst, s.heap[instr.list_id][idx], VType.INT)

    elif op is Op.STORE:
        idx = _read(s, instr.a)
        if not (0 <= idx < len(s.heap[instr.list_id])):
            s.error = True
            s.halted = True
            advance = False
        else:
            s.heap[instr.list_id][idx] = _read(s, instr.b)

    else:  # pragma: no cover - exhaustive above
        raise VMError(f"unhandled opcode {op}")

    if advance:
        s.pc += 1
    return s


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


@dataclass
class Trace:
    """An execution trace: ``states[t]`` is the machine state before step ``t``,
    ``actions[t]`` is the instruction executed at step ``t``, and
    ``states[t + 1]`` is the resulting state. ``len(states) == len(actions) + 1``.

    ``terminated`` is True if the program HALTed or trapped; False if it ran into
    the step budget (treated as non-terminating for that input).
    """

    program: list[Instr]
    states: list[MachineState]
    actions: list[Instr]
    terminated: bool

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def final_state(self) -> MachineState:
        return self.states[-1]


def run_traced(program: list[Instr], init_state: MachineState,
               max_steps: int | None = None) -> Trace:
    """Execute ``program`` from ``init_state``, recording the full state at every
    step. Stops on HALT/trap or when the step budget is exhausted.

    The instruction fetched at each step is the one at the current ``pc``. An
    empty program, or a pc that runs off the end, terminates as if HALTed.
    """
    if max_steps is None:
        max_steps = 256
    states: list[MachineState] = [init_state.copy()]
    actions: list[Instr] = []
    cur = init_state.copy()
    terminated = False

    while cur.steps < max_steps:
        if cur.halted:
            terminated = True
            break
        if not (0 <= cur.pc < len(program)):
            # falling off the end is normal termination
            terminated = True
            break
        instr = program[cur.pc]
        nxt = step(cur, instr)
        actions.append(instr)
        states.append(nxt)
        cur = nxt
        if cur.halted:
            terminated = True
            break

    return Trace(program=program, states=states, actions=actions,
                 terminated=terminated)
