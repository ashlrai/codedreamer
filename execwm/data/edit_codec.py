"""Edit codec: encodes a program :class:`~execwm.substrate.edits.Edit` into
integer fields, the M3 analogue of :mod:`action_codec`.

An "edit-as-action" is encoded structurally so the world model receives it in
the same factored form as states and statement-actions: the edit kind, the
target instruction index, and the changed field — a new opcode id, a new
register id, or a new immediate (reusing the shared ``(sign, digits)`` encoding
from :mod:`state_codec`). Only the field relevant to the kind carries signal;
the rest take sentinel/zero values. Round-trip is bit-exact: ``encode`` then
``decode`` reconstructs the original :class:`Edit`, and re-encoding is identical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..substrate.edits import Edit, EditConfig, EditKind
from ..substrate.vm import Config
from .action_codec import ALL_OPS, _OP_INDEX
from .state_codec import CodecConfig, EncodeError, decode_int, encode_int

# operand-slot classes (which of an instruction's two source slots an edit hits)
SLOT_NONE, SLOT_A, SLOT_B = 0, 1, 2
_SLOT_TO_ID = {"a": SLOT_A, "b": SLOT_B}
_ID_TO_SLOT = {SLOT_A: "a", SLOT_B: "b"}


@dataclass
class EncodedEdit:
    """Integer fields for one edit (scalars unless noted).

      kind       in {0..len(EditKind)-1}
      index      target instruction index, in {0..max_program_len-1}
      op         new opcode id (valid iff kind==CHANGE_OP, else sentinel none_op)
      dst        new register id (valid iff kind==CHANGE_DST, else none_reg)
      slot       in {none, a, b} (valid iff kind in {CHANGE_OPERAND, CHANGE_IMM})
      reg        new register id (valid iff kind==CHANGE_OPERAND, else none_reg)
      imm_sign   new immediate sign (valid iff kind==CHANGE_IMM)
      imm_digits (max_digits,) new immediate magnitude (valid iff kind==CHANGE_IMM)
    """

    kind: int
    index: int
    op: int
    dst: int
    slot: int
    reg: int
    imm_sign: int
    imm_digits: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        scalar = lambda v: np.array(v, dtype=np.int64)
        return {
            "kind": scalar(self.kind), "index": scalar(self.index),
            "op": scalar(self.op), "dst": scalar(self.dst),
            "slot": scalar(self.slot), "reg": scalar(self.reg),
            "imm_sign": scalar(self.imm_sign), "imm_digits": self.imm_digits,
        }


class EditCodec:
    """Encodes/decodes edits for a fixed VM :class:`Config`."""

    def __init__(self, config: Config, edit_config: EditConfig | None = None,
                 codec: CodecConfig | None = None) -> None:
        self.config = config
        self.edit_config = edit_config or EditConfig()
        self.codec = codec or CodecConfig()
        self.reg_names = list(config.reg_names)
        self.reg_index = {n: i for i, n in enumerate(self.reg_names)}
        self.num_regs = len(self.reg_names)
        self.max_program_len = self.edit_config.max_program_len
        self.none_op = len(ALL_OPS)     # sentinel opcode id ("no op change")
        self.none_reg = self.num_regs   # sentinel register id ("no register")

    def encode(self, edit: Edit) -> EncodedEdit:
        D = self.codec.max_digits
        if not (0 <= edit.index < self.max_program_len):
            raise EncodeError(
                f"edit index {edit.index} exceeds max_program_len "
                f"{self.max_program_len}")

        op = self.none_op
        dst = self.none_reg
        slot = SLOT_NONE
        reg = self.none_reg
        imm_sign = 0
        imm_digits = np.zeros(D, dtype=np.int64)

        if edit.kind is EditKind.CHANGE_OP:
            if edit.new_op is None:
                raise EncodeError("CHANGE_OP edit missing new_op")
            op = _OP_INDEX[edit.new_op]
        elif edit.kind is EditKind.CHANGE_DST:
            if edit.new_dst is None:
                raise EncodeError("CHANGE_DST edit missing new_dst")
            dst = self.reg_index[edit.new_dst]
        elif edit.kind is EditKind.CHANGE_OPERAND:
            slot = _SLOT_TO_ID.get(edit.slot, SLOT_NONE)
            if slot == SLOT_NONE or edit.new_reg is None:
                raise EncodeError("CHANGE_OPERAND edit missing slot/new_reg")
            reg = self.reg_index[edit.new_reg]
        elif edit.kind is EditKind.CHANGE_IMM:
            slot = _SLOT_TO_ID.get(edit.slot, SLOT_NONE)
            if slot == SLOT_NONE or edit.new_imm is None:
                raise EncodeError("CHANGE_IMM edit missing slot/new_imm")
            imm_sign, imm_digits = encode_int(edit.new_imm, self.codec)
        else:  # pragma: no cover - exhaustive above
            raise EncodeError(f"unhandled edit kind {edit.kind}")

        return EncodedEdit(
            kind=edit.kind.value, index=edit.index, op=op, dst=dst,
            slot=slot, reg=reg, imm_sign=imm_sign, imm_digits=imm_digits,
        )

    def decode(self, enc: EncodedEdit) -> Edit:
        kind = EditKind(int(enc.kind))
        new_op = None
        new_dst = None
        slot = None
        new_reg = None
        new_imm = None

        if kind is EditKind.CHANGE_OP:
            new_op = ALL_OPS[int(enc.op)]
        elif kind is EditKind.CHANGE_DST:
            new_dst = self.reg_names[int(enc.dst)]
        elif kind is EditKind.CHANGE_OPERAND:
            slot = _ID_TO_SLOT[int(enc.slot)]
            new_reg = self.reg_names[int(enc.reg)]
        elif kind is EditKind.CHANGE_IMM:
            slot = _ID_TO_SLOT[int(enc.slot)]
            new_imm = decode_int(int(enc.imm_sign), enc.imm_digits, self.codec)

        return Edit(index=int(enc.index), kind=kind, new_op=new_op,
                    new_dst=new_dst, slot=slot, new_reg=new_reg, new_imm=new_imm)
