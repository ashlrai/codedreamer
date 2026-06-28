"""Lossless symbolic-state <-> tensor codec.

This module is the operational definition of "exact state match": a
:class:`~execwm.substrate.vm.MachineState` is encoded into a bundle of integer
*label* arrays, and two states are equal iff their encodings agree (with an
UNDEF-aware mask on register payloads). Those same label arrays are precisely the
targets the model's shallow grounding heads predict, so the codec ties together
data, training, and evaluation.

Integers are represented as ``(sign, digits)`` in a fixed base with a fixed digit
width. The width is a codec hyperparameter, deliberately wide enough to cover the
out-of-distribution *magnitude* axis (e.g. train values < 10, test values < 1000)
so generalization is testable without the codec ever truncating.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..substrate.vm import Config, MachineState, VType

# Register payload is meaningful only for these types; UNDEF masks it out.
_VALUED_TYPES = (VType.INT, VType.BOOL)


@dataclass(frozen=True)
class CodecConfig:
    """Fixed-width numeric encoding parameters.

    ``max_digits`` and ``base`` bound the representable magnitude
    (``base ** max_digits``); ``max_pc`` bounds the program-counter class count.
    """

    max_digits: int = 4
    base: int = 10
    max_pc: int = 256

    @property
    def max_magnitude(self) -> int:
        return self.base ** self.max_digits


class EncodeError(ValueError):
    """Raised when a value does not fit the codec's representable range."""


def encode_int(value: int, codec: "CodecConfig") -> tuple[int, np.ndarray]:
    """Encode an int as ``(sign, MSB-first digit array)`` in the codec's base.

    Shared by the state and action codecs so "what a number looks like to the
    model" is defined in exactly one place.
    """
    sign = 1 if value < 0 else 0
    mag = abs(int(value))
    if mag >= codec.max_magnitude:
        raise EncodeError(
            f"value {value} exceeds codec range "
            f"(base={codec.base}, digits={codec.max_digits})")
    digits = np.zeros(codec.max_digits, dtype=np.int64)
    for i in range(codec.max_digits - 1, -1, -1):
        digits[i] = mag % codec.base
        mag //= codec.base
    return sign, digits


def decode_int(sign: int, digits: np.ndarray, codec: "CodecConfig") -> int:
    mag = 0
    for d in digits:
        mag = mag * codec.base + int(d)
    return -mag if sign else mag


@dataclass
class EncodedState:
    """Integer label arrays for one state. All arrays hold class indices.

    Shapes (R = #registers, C = #heap cells = num_lists * list_len):
      reg_type   (R,)            in {0,1,2}  (VType value)
      reg_sign   (R,)            in {0,1}
      reg_digits (R, max_digits) in {0..base-1}
      heap_sign  (C,)            in {0,1}
      heap_digits(C, max_digits) in {0..base-1}
      pc         ()              in {0..max_pc}
      halted     ()              in {0,1}
      error      ()              in {0,1}
    """

    reg_type: np.ndarray
    reg_sign: np.ndarray
    reg_digits: np.ndarray
    heap_sign: np.ndarray
    heap_digits: np.ndarray
    pc: np.ndarray
    halted: np.ndarray
    error: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "reg_type": self.reg_type, "reg_sign": self.reg_sign,
            "reg_digits": self.reg_digits, "heap_sign": self.heap_sign,
            "heap_digits": self.heap_digits, "pc": self.pc,
            "halted": self.halted, "error": self.error,
        }


class StateCodec:
    """Encodes/decodes states for a fixed VM :class:`Config`."""

    def __init__(self, config: Config, codec: CodecConfig | None = None) -> None:
        self.config = config
        self.codec = codec or CodecConfig()
        self.reg_names = list(config.reg_names)
        self.num_regs = len(self.reg_names)
        self.num_cells = config.num_lists * config.list_len

    # -- integer <-> (sign, digits) ------------------------------------------

    def _encode_int(self, value: int) -> tuple[int, np.ndarray]:
        return encode_int(value, self.codec)

    def _decode_int(self, sign: int, digits: np.ndarray) -> int:
        return decode_int(sign, digits, self.codec)

    # -- state <-> tensors ----------------------------------------------------

    def encode(self, state: MachineState) -> EncodedState:
        R, D = self.num_regs, self.codec.max_digits
        reg_type = np.zeros(R, dtype=np.int64)
        reg_sign = np.zeros(R, dtype=np.int64)
        reg_digits = np.zeros((R, D), dtype=np.int64)
        for i, name in enumerate(self.reg_names):
            vtype = state.types[name]
            reg_type[i] = vtype.value
            if vtype in _VALUED_TYPES:
                s, dg = self._encode_int(state.regs[name])
                reg_sign[i] = s
                reg_digits[i] = dg

        C = self.num_cells
        heap_sign = np.zeros(C, dtype=np.int64)
        heap_digits = np.zeros((C, D), dtype=np.int64)
        flat = [v for cells in state.heap for v in cells]
        for i, v in enumerate(flat):
            s, dg = self._encode_int(v)
            heap_sign[i] = s
            heap_digits[i] = dg

        if not (0 <= state.pc <= self.codec.max_pc):
            raise EncodeError(f"pc {state.pc} exceeds max_pc {self.codec.max_pc}")

        return EncodedState(
            reg_type=reg_type, reg_sign=reg_sign, reg_digits=reg_digits,
            heap_sign=heap_sign, heap_digits=heap_digits,
            pc=np.array(state.pc, dtype=np.int64),
            halted=np.array(int(state.halted), dtype=np.int64),
            error=np.array(int(state.error), dtype=np.int64),
        )

    def decode(self, enc: EncodedState) -> MachineState:
        regs: dict[str, int | None] = {}
        types: dict[str, VType] = {}
        for i, name in enumerate(self.reg_names):
            vtype = VType(int(enc.reg_type[i]))
            types[name] = vtype
            if vtype in _VALUED_TYPES:
                regs[name] = self._decode_int(int(enc.reg_sign[i]), enc.reg_digits[i])
            else:
                regs[name] = None

        flat = [self._decode_int(int(enc.heap_sign[i]), enc.heap_digits[i])
                for i in range(self.num_cells)]
        heap = [flat[j * self.config.list_len:(j + 1) * self.config.list_len]
                for j in range(self.config.num_lists)]

        return MachineState(
            regs=regs, types=types, heap=heap, pc=int(enc.pc),
            halted=bool(enc.halted), error=bool(enc.error),
        )

    # -- exact match ----------------------------------------------------------

    def exact_match(self, a: EncodedState, b: EncodedState) -> bool:
        """True iff the two encoded states are identical, ignoring the numeric
        payload of any register that is UNDEF in *both* (its value is junk)."""
        if int(a.pc) != int(b.pc):
            return False
        if int(a.halted) != int(b.halted) or int(a.error) != int(b.error):
            return False
        if not np.array_equal(a.reg_type, b.reg_type):
            return False
        if not (np.array_equal(a.heap_sign, b.heap_sign)
                and np.array_equal(a.heap_digits, b.heap_digits)):
            return False
        # register payloads: only compare where the (agreed) type is valued
        valued = np.isin(a.reg_type, [t.value for t in _VALUED_TYPES])
        if not np.array_equal(a.reg_sign[valued], b.reg_sign[valued]):
            return False
        if not np.array_equal(a.reg_digits[valued], b.reg_digits[valued]):
            return False
        return True
