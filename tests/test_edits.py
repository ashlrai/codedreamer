"""Tests for the M3 "edit as action" data substrate.

Four load-bearing properties:
* :func:`apply_edit` is functional (new list, input untouched) and yields a
  runnable program.
* Each :class:`EditKind` can actually change execution (edited trace differs).
* :class:`EditCodec` round-trips every edit kind bit-exactly.
* :func:`make_edit_example` yields divergent base/edited traces over seeds.
"""

import random

import numpy as np
import pytest

from execwm.data.edit_codec import EditCodec
from execwm.data.edit_dataset import make_edit_example, traces_equivalent
from execwm.data.state_codec import CodecConfig, EncodeError
from execwm.substrate.dsl import make_config
from execwm.substrate.edits import (Edit, EditConfig, EditError, EditKind,
                                     apply_edit, enumerate_valid_edits,
                                     sample_edit)
from execwm.substrate.generators import GenSpec
from execwm.substrate.vm import Instr, Op, run_traced

# A small fixed config + a handcrafted program with a known trace, used to give
# each edit kind a deterministic divergence. v0/v1 are set by CONST so no inputs
# are needed; t0/t1 are temps.
_CONFIG = make_config(num_vars=2, num_temps=2)
_CODEC = CodecConfig(max_digits=4, base=10, max_pc=128)
_EDIT_CFG = EditConfig(max_program_len=64)


def _base_program() -> list[Instr]:
    return [
        Instr(Op.CONST, dst="v0", a=5),            # 0: v0 = 5
        Instr(Op.CONST, dst="v1", a=3),            # 1: v1 = 3
        Instr(Op.ADD, dst="t0", a="v0", b="v1"),   # 2: t0 = v0 + v1 = 8
        Instr(Op.MOV, dst="t1", a="t0"),           # 3: t1 = t0 = 8
        Instr(Op.HALT),                            # 4
    ]


def _trace(program: list[Instr]):
    init = _CONFIG.initial_state()
    return run_traced(program, init, max_steps=256)


# One edit per kind that is known to diverge from the base trace above.
_EDITS_BY_KIND = {
    EditKind.CHANGE_OP: Edit(2, EditKind.CHANGE_OP, new_op=Op.SUB),       # 8 -> 2
    EditKind.CHANGE_DST: Edit(3, EditKind.CHANGE_DST, new_dst="v0"),      # v0 8 not 5
    EditKind.CHANGE_OPERAND: Edit(2, EditKind.CHANGE_OPERAND, slot="a",
                                  new_reg="v1"),                          # 6 not 8
    EditKind.CHANGE_IMM: Edit(0, EditKind.CHANGE_IMM, slot="a", new_imm=7),
}


def test_apply_edit_is_functional_and_runnable():
    for kind, edit in _EDITS_BY_KIND.items():
        program = _base_program()
        snapshot = list(program)
        edited = apply_edit(program, edit)
        # new list object, input untouched (same elements, same order)
        assert edited is not program
        assert program == snapshot, f"{kind} mutated the input program"
        assert len(edited) == len(program)
        # exactly the targeted index changed
        diff = [i for i in range(len(program)) if edited[i] != program[i]]
        assert diff == [edit.index]
        # edited program is runnable end to end
        tr = _trace(edited)
        assert tr.terminated


def test_apply_edit_rejects_inapplicable():
    program = _base_program()
    # CHANGE_OP on a non-swappable op (CONST has no arity peer)
    with pytest.raises(EditError):
        apply_edit(program, Edit(0, EditKind.CHANGE_OP, new_op=Op.MOV))
    # CHANGE_OP to an op outside the arity class (ADD -> JZ)
    with pytest.raises(EditError):
        apply_edit(program, Edit(2, EditKind.CHANGE_OP, new_op=Op.JZ))
    # CHANGE_IMM on a register operand slot
    with pytest.raises(EditError):
        apply_edit(program, Edit(2, EditKind.CHANGE_IMM, slot="a", new_imm=1))
    # CHANGE_OPERAND on an immediate slot
    with pytest.raises(EditError):
        apply_edit(program, Edit(0, EditKind.CHANGE_OPERAND, slot="a", new_reg="v1"))
    # index out of range
    with pytest.raises(EditError):
        apply_edit(program, Edit(99, EditKind.CHANGE_DST, new_dst="v0"))


def test_each_editkind_changes_execution():
    base = _trace(_base_program())
    for kind, edit in _EDITS_BY_KIND.items():
        edited = _trace(apply_edit(_base_program(), edit))
        assert not traces_equivalent(base, edited), (
            f"{kind} did not change the trace")


def _edit_eq_arrays(a, b) -> bool:
    da, db = a.as_dict(), b.as_dict()
    assert set(da) == set(db)
    return all(np.array_equal(da[k], db[k]) for k in da)


def test_editcodec_roundtrip_bit_exact():
    codec = EditCodec(_CONFIG, _EDIT_CFG, _CODEC)
    for edit in _EDITS_BY_KIND.values():
        enc = codec.encode(edit)
        dec = codec.decode(enc)
        # decode reconstructs the exact Edit...
        assert dec == edit
        # ...and re-encoding is bit-identical.
        assert _edit_eq_arrays(codec.encode(dec), enc)
        # all fields are fixed-shape int64
        for v in enc.as_dict().values():
            assert v.dtype == np.int64


def test_editcodec_roundtrip_over_sampled_edits():
    rng = random.Random(0)
    spec = GenSpec(num_vars=3, num_temps=14, max_depth=2, num_stmts=4,
                   max_const=5, max_input_val=5, max_loop_count=3)
    config = spec.config()
    codec = EditCodec(config, _EDIT_CFG, CodecConfig(max_digits=6, max_pc=256))
    seen = set()
    for s in range(40):
        ex_rng = random.Random(s)
        program = make_edit_example(
            ex_rng, spec, CodecConfig(max_digits=6, max_pc=256), _EDIT_CFG).edit
        enc = codec.encode(program)
        dec = codec.decode(enc)
        assert dec == program
        assert _edit_eq_arrays(codec.encode(dec), enc)
        seen.add(program.kind)
    # the sampler exercises more than one edit kind
    assert len(seen) >= 2


def test_editcodec_out_of_range_raises():
    codec = EditCodec(_CONFIG, EditConfig(max_program_len=4),
                      CodecConfig(max_digits=2, base=10))  # |imm| < 100
    with pytest.raises(EncodeError):  # immediate out of codec range
        codec.encode(Edit(0, EditKind.CHANGE_IMM, slot="a", new_imm=100))
    with pytest.raises(EncodeError):  # index >= max_program_len
        codec.encode(Edit(4, EditKind.CHANGE_DST, new_dst="v0"))


def test_make_edit_example_diverges_over_seeds():
    spec = GenSpec(num_vars=3, num_temps=14, max_depth=2, num_stmts=4,
                   max_const=5, max_input_val=5, max_loop_count=3)
    codec_cfg = CodecConfig(max_digits=6, max_pc=256)
    for seed in range(8):
        rng = random.Random(seed)
        ex = make_edit_example(rng, spec, codec_cfg, _EDIT_CFG)
        # traces share the same starting state but diverge
        assert ex.base_trace.states[0].regs == ex.init_state.regs
        assert ex.edited_trace.states[0].regs == ex.init_state.regs
        assert not traces_equivalent(ex.base_trace, ex.edited_trace)
        # the recorded edit reproduces the edited program
        assert apply_edit(ex.base_bytecode, ex.edit) == ex.edited_bytecode


def test_enumerate_only_applicable_edits():
    program = _base_program()
    edits = enumerate_valid_edits(program, _CONFIG, random.Random(0), _EDIT_CFG)
    assert edits
    # every enumerated edit must apply without raising
    for edit in edits:
        apply_edit(program, edit)
    # sampler returns one of them
    e = sample_edit(program, _CONFIG, random.Random(1), _EDIT_CFG)
    assert e is not None
    apply_edit(program, e)
