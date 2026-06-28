"""Program edits as first-class actions (M3 foundation: "edit as action").

Where :mod:`action_codec` treats *one executed instruction* as the action, M3
treats *an edit to the program text* as the action: change a statement, re-run
from the same inputs, and observe how the whole execution trace changes. This
module defines the symbolic edit object and the structural machinery to apply
and sample edits — the pure-substrate counterpart to the world model that will
later learn to predict the trace delta an edit induces.

Design notes
------------
* An :class:`Edit` is purely structural: an instruction index plus a kind and the
  changed field(s). :func:`apply_edit` is functional — it returns a *new*
  program list and never mutates its input — mirroring :func:`vm.step`.
* Edits are constrained to keep the program *structurally valid* (a runnable
  :class:`~execwm.substrate.vm.Instr`). They are **not** guaranteed to keep it
  free of runtime traps or undefined-register reads — that is execution- and
  input-dependent, so the dataset builder (:mod:`execwm.data.edit_dataset`)
  validates by actually running and drops/retries bad samples.
* ``CHANGE_OP`` only swaps an opcode for another op in the *same arity class*
  (see :data:`_SWAP_CLASSES`): the binary-op class (arithmetic + comparison, all
  ``dst,a,b``) and the conditional-jump class (``JZ``/``JNZ``, both ``a,target``).
  Ops with a unique structure (``CONST``, ``MOV``, ``JMP``, ``LOAD``, ``STORE``,
  ``HALT``) have no same-arity peer and so admit no ``CHANGE_OP`` edit.
"""

from __future__ import annotations

import enum
import random
from dataclasses import dataclass, replace

from .vm import ARITH_OPS, CMP_OPS, Config, Instr, Op


class EditKind(enum.Enum):
    """The four kinds of single-field program edit. Values are stable codec ids."""

    CHANGE_OP = 0       # replace the opcode (within its arity class), keep operands
    CHANGE_DST = 1      # change the destination register
    CHANGE_OPERAND = 2  # change a source register operand (reg -> reg)
    CHANGE_IMM = 3      # change an immediate / constant operand


# Opcodes that may be swapped for one another by CHANGE_OP, grouped by structure.
_BINOP_CLASS = frozenset(ARITH_OPS + CMP_OPS)      # all take dst, a, b
_COND_JUMP_CLASS = frozenset((Op.JZ, Op.JNZ))      # both take a, target
_SWAP_CLASSES: tuple[frozenset[Op], ...] = (_BINOP_CLASS, _COND_JUMP_CLASS)


def _swap_class(op: Op) -> frozenset[Op] | None:
    """The arity class ``op`` belongs to (for CHANGE_OP), or ``None`` if unique."""
    for cls in _SWAP_CLASSES:
        if op in cls:
            return cls
    return None


@dataclass(frozen=True)
class EditConfig:
    """Bounds for proposing/encoding edits, decoupled from the numeric codec.

    ``max_program_len`` bounds the editable instruction index (the codec's index
    class count). It is kept separate from ``CodecConfig`` so the substrate layer
    has no dependency on the data layer.
    """

    max_program_len: int = 256


@dataclass(frozen=True)
class Edit:
    """A single structural edit to a bytecode program.

    Only the field(s) relevant to ``kind`` are populated; the rest stay ``None``
    so encode->decode round-trips to an identical object.

      CHANGE_OP        -> ``new_op``
      CHANGE_DST       -> ``new_dst``
      CHANGE_OPERAND   -> ``slot`` ('a'|'b'), ``new_reg``
      CHANGE_IMM       -> ``slot`` ('a'|'b'), ``new_imm``
    """

    index: int
    kind: EditKind
    new_op: Op | None = None
    new_dst: str | None = None
    slot: str | None = None
    new_reg: str | None = None
    new_imm: int | None = None


class EditError(ValueError):
    """Raised when an edit is not structurally applicable to its target."""


def apply_edit(program: list[Instr], edit: Edit) -> list[Instr]:
    """Apply ``edit`` to ``program``, returning a NEW list (input is untouched).

    Raises :class:`EditError` if the edit is not structurally applicable — e.g. a
    CHANGE_OP to an arity-incompatible opcode, or a CHANGE_IMM on a slot that does
    not currently hold an immediate. The result is always a structurally valid
    :class:`Instr` (register indices and operand kinds are respected), though it
    may still trap or read an undefined register at run time depending on inputs.
    """
    if not (0 <= edit.index < len(program)):
        raise EditError(f"edit index {edit.index} out of range [0, {len(program)})")
    instr = program[edit.index]
    new = list(program)

    if edit.kind is EditKind.CHANGE_OP:
        cls = _swap_class(instr.op)
        if cls is None or edit.new_op not in cls:
            raise EditError(
                f"CHANGE_OP: {edit.new_op} is not arity-compatible with {instr.op}")
        new[edit.index] = replace(instr, op=edit.new_op)

    elif edit.kind is EditKind.CHANGE_DST:
        if instr.dst is None:
            raise EditError(f"CHANGE_DST: {instr.op} has no destination register")
        if edit.new_dst is None:
            raise EditError("CHANGE_DST: new_dst is required")
        new[edit.index] = replace(instr, dst=edit.new_dst)

    elif edit.kind is EditKind.CHANGE_OPERAND:
        if edit.slot not in ("a", "b"):
            raise EditError(f"CHANGE_OPERAND: bad slot {edit.slot!r}")
        cur = getattr(instr, edit.slot)
        if not isinstance(cur, str):
            raise EditError(
                f"CHANGE_OPERAND: slot {edit.slot!r} is not a register operand")
        if edit.new_reg is None:
            raise EditError("CHANGE_OPERAND: new_reg is required")
        new[edit.index] = replace(instr, **{edit.slot: edit.new_reg})

    elif edit.kind is EditKind.CHANGE_IMM:
        if edit.slot not in ("a", "b"):
            raise EditError(f"CHANGE_IMM: bad slot {edit.slot!r}")
        cur = getattr(instr, edit.slot)
        if not isinstance(cur, int) or isinstance(cur, bool):
            raise EditError(
                f"CHANGE_IMM: slot {edit.slot!r} is not an immediate operand")
        if edit.new_imm is None:
            raise EditError("CHANGE_IMM: new_imm is required")
        new[edit.index] = replace(instr, **{edit.slot: edit.new_imm})

    else:  # pragma: no cover - exhaustive above
        raise EditError(f"unhandled edit kind {edit.kind}")

    return new


def _imm_candidates(val: int) -> list[int]:
    """A small set of alternative immediates near ``val`` (excluding ``val``)."""
    cands = {val + 1, val - 1, val + 2, val - 2, -val, 0}
    cands.discard(val)
    return sorted(cands)


def enumerate_valid_edits(program: list[Instr], config: Config,
                          rng: random.Random | None = None,
                          edit_config: EditConfig | None = None) -> list[Edit]:
    """All structurally-applicable edits to ``program`` over ``config``.

    "Applicable" means :func:`apply_edit` will succeed and produce a valid
    instruction; it does NOT guarantee the edited program avoids runtime traps or
    undefined reads (that depends on the inputs). Index is bounded by
    ``edit_config.max_program_len`` so every returned edit is encodable. Register
    replacements for CHANGE_OPERAND are biased toward registers written somewhere
    in the program (more likely to be defined when read).
    """
    edit_config = edit_config or EditConfig()
    reg_names = list(config.reg_names)
    written = sorted({ins.dst for ins in program if ins.dst is not None})
    operand_regs = written or reg_names

    edits: list[Edit] = []
    limit = min(len(program), edit_config.max_program_len)
    for i in range(limit):
        instr = program[i]

        # CHANGE_OP: swap within the opcode's arity class.
        cls = _swap_class(instr.op)
        if cls is not None:
            for op2 in cls:
                if op2 is not instr.op:
                    edits.append(Edit(i, EditKind.CHANGE_OP, new_op=op2))

        # CHANGE_DST: any other register.
        if instr.dst is not None:
            for r in reg_names:
                if r != instr.dst:
                    edits.append(Edit(i, EditKind.CHANGE_DST, new_dst=r))

        # Source operands: CHANGE_OPERAND (reg->reg) or CHANGE_IMM (int->int).
        for slot in ("a", "b"):
            cur = getattr(instr, slot)
            if isinstance(cur, str):
                for r in operand_regs:
                    if r != cur:
                        edits.append(
                            Edit(i, EditKind.CHANGE_OPERAND, slot=slot, new_reg=r))
            elif isinstance(cur, int) and not isinstance(cur, bool):
                for nv in _imm_candidates(cur):
                    edits.append(
                        Edit(i, EditKind.CHANGE_IMM, slot=slot, new_imm=nv))

    return edits


def sample_edit(program: list[Instr], config: Config, rng: random.Random,
                edit_config: EditConfig | None = None) -> Edit | None:
    """Sample one structurally-applicable edit, or ``None`` if none exist."""
    candidates = enumerate_valid_edits(program, config, rng, edit_config)
    if not candidates:
        return None
    return rng.choice(candidates)
