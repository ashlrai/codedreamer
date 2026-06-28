"""Action codec: encodes a bytecode :class:`~execwm.substrate.vm.Instr` into
integer fields the world model conditions its latent dynamics on.

An "action" at statement granularity is the instruction executed at a step. We
encode it structurally — opcode, destination register, two operands (each tagged
register / immediate / none), heap list id, and jump target — rather than as text,
so the dynamics predictor receives the action in the same factored form as the
state. Immediates reuse the shared digit encoding from :mod:`state_codec`, so the
magnitude out-of-distribution axis applies to literals in actions too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..substrate.vm import Config, Instr, Op
from .state_codec import CodecConfig, decode_int, encode_int

ALL_OPS: tuple[Op, ...] = tuple(Op)
_OP_INDEX: dict[Op, int] = {op: i for i, op in enumerate(ALL_OPS)}

# operand-kind classes
OPK_NONE, OPK_REG, OPK_IMM = 0, 1, 2


@dataclass
class EncodedAction:
    """Integer fields for one action (scalars unless noted).

      op        in {0..len(ALL_OPS)-1}
      dst       register index, or num_regs (sentinel = none)
      a_kind/b_kind  in {none, reg, imm}
      a_reg/b_reg    register index (valid iff kind==reg, else 0)
      a_sign/b_sign  immediate sign (valid iff kind==imm)
      a_digits/b_digits  (max_digits,) immediate magnitude (valid iff kind==imm)
      list_id   list index, or num_lists (sentinel = none)
      target    jump target pc, or max_pc (sentinel = none)
    """

    op: int
    dst: int
    a_kind: int
    a_reg: int
    a_sign: int
    a_digits: np.ndarray
    b_kind: int
    b_reg: int
    b_sign: int
    b_digits: np.ndarray
    list_id: int
    target: int

    def as_dict(self) -> dict[str, np.ndarray]:
        scalar = lambda v: np.array(v, dtype=np.int64)
        return {
            "op": scalar(self.op), "dst": scalar(self.dst),
            "a_kind": scalar(self.a_kind), "a_reg": scalar(self.a_reg),
            "a_sign": scalar(self.a_sign), "a_digits": self.a_digits,
            "b_kind": scalar(self.b_kind), "b_reg": scalar(self.b_reg),
            "b_sign": scalar(self.b_sign), "b_digits": self.b_digits,
            "list_id": scalar(self.list_id), "target": scalar(self.target),
        }


class ActionCodec:
    """Encodes/decodes instructions for a fixed VM :class:`Config`."""

    def __init__(self, config: Config, codec: CodecConfig | None = None) -> None:
        self.config = config
        self.codec = codec or CodecConfig()
        self.reg_names = list(config.reg_names)
        self.reg_index = {n: i for i, n in enumerate(self.reg_names)}
        self.num_regs = len(self.reg_names)
        self.num_lists = config.num_lists
        self.none_reg = self.num_regs       # sentinel index for "no register"
        self.none_list = self.num_lists     # sentinel for "no list"
        self.none_target = self.codec.max_pc  # sentinel for "no jump target"

    def _encode_operand(self, operand) -> tuple[int, int, int, np.ndarray]:
        D = self.codec.max_digits
        if operand is None:
            return OPK_NONE, 0, 0, np.zeros(D, dtype=np.int64)
        if isinstance(operand, str):
            return OPK_REG, self.reg_index[operand], 0, np.zeros(D, dtype=np.int64)
        if isinstance(operand, int):
            sign, digits = encode_int(operand, self.codec)
            return OPK_IMM, 0, sign, digits
        raise TypeError(f"bad operand {operand!r}")

    def encode(self, instr: Instr) -> EncodedAction:
        a_kind, a_reg, a_sign, a_digits = self._encode_operand(instr.a)
        b_kind, b_reg, b_sign, b_digits = self._encode_operand(instr.b)
        dst = self.reg_index[instr.dst] if instr.dst is not None else self.none_reg
        list_id = instr.list_id if instr.list_id is not None else self.none_list
        target = instr.target if instr.target is not None else self.none_target
        return EncodedAction(
            op=_OP_INDEX[instr.op], dst=dst,
            a_kind=a_kind, a_reg=a_reg, a_sign=a_sign, a_digits=a_digits,
            b_kind=b_kind, b_reg=b_reg, b_sign=b_sign, b_digits=b_digits,
            list_id=list_id, target=target,
        )

    def _decode_operand(self, kind, reg, sign, digits):
        if kind == OPK_NONE:
            return None
        if kind == OPK_REG:
            return self.reg_names[int(reg)]
        return decode_int(int(sign), digits, self.codec)

    def decode(self, enc: EncodedAction) -> Instr:
        dst = None if int(enc.dst) == self.none_reg else self.reg_names[int(enc.dst)]
        list_id = None if int(enc.list_id) == self.none_list else int(enc.list_id)
        target = None if int(enc.target) == self.none_target else int(enc.target)
        return Instr(
            op=ALL_OPS[int(enc.op)], dst=dst,
            a=self._decode_operand(enc.a_kind, enc.a_reg, enc.a_sign, enc.a_digits),
            b=self._decode_operand(enc.b_kind, enc.b_reg, enc.b_sign, enc.b_digits),
            list_id=list_id, target=target,
        )
