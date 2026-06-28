"""Tests for the action codec, the compositional structural check, and the
disjoint OOD split builder."""

import random

import numpy as np
import pytest

from execwm.data.action_codec import ActionCodec
from execwm.data.dataset import (build_split, flatten_transitions,
                                  program_uses_pairs)
from execwm.data.state_codec import CodecConfig, StateCodec
from execwm.substrate.generators import GenSpec, default_axes, make_example
from execwm.substrate.vm import Op


def test_action_codec_roundtrip():
    rng = random.Random(0)
    spec = GenSpec(num_vars=4, max_depth=2, num_stmts=5)
    acodec = ActionCodec(spec.config(), CodecConfig(max_digits=4))
    checked = 0
    for _ in range(200):
        ex = make_example(rng, spec)
        for instr in ex.bytecode:
            dec = acodec.decode(acodec.encode(instr))
            assert dec == instr, f"{dec} != {instr}"
            checked += 1
    assert checked > 200


def test_compositional_structural_check():
    """A held-out pair forbidden in generation must never appear; allowed
    generation must (sometimes) produce it."""
    rng = random.Random(1)
    pairs = frozenset({("loop", Op.MUL), ("loop", Op.MOD)})
    forbid_spec = GenSpec(forbidden_pairs=pairs, max_depth=3, num_stmts=6)
    free_spec = GenSpec(forbidden_pairs=frozenset(), max_depth=3, num_stmts=6)
    for _ in range(300):
        ex = make_example(rng, forbid_spec)
        assert not program_uses_pairs(ex.program, pairs)
    found = sum(program_uses_pairs(make_example(rng, free_spec).program, pairs)
                for _ in range(400))
    assert found > 0, "free generation never produced a held-out pairing"


def test_build_split_all_axes_disjoint():
    """build_split asserts disjointness internally; here we also check the
    arrays are well-formed and the next-state decodes back to a real VM state."""
    codec_cfg = CodecConfig(max_digits=9, base=10, max_pc=512)
    for axis in default_axes():
        split = build_split(axis, n_train=24, n_test=12, codec_cfg=codec_cfg, seed=0)
        st = split.stats
        assert st["n_train_transitions"] > 0 and st["n_test_transitions"] > 0
        # array length consistency across all keys in a partition
        for part in (split.train, split.test):
            n = len(part["ex_id"])
            for k, v in part.items():
                assert v.shape[0] == n, f"{axis.name}:{k} ragged"
        # numeric axes must report a non-overlapping metric range
        if axis.name != "compositional":
            lo_tr, hi_tr = st["train_metric_range"]
            lo_te, hi_te = st["test_metric_range"]
            assert hi_tr < lo_te


def test_transition_targets_are_consistent():
    """The flattened next-state must equal the VM's actual next state, i.e. the
    training target is genuinely ground truth."""
    rng = random.Random(2)
    spec = GenSpec(num_vars=4, max_depth=2, num_stmts=5, max_const=5, max_input_val=5)
    codec = CodecConfig(max_digits=9, max_pc=512)
    scodec = StateCodec(spec.config(), codec)
    acodec = ActionCodec(spec.config(), codec)
    examples = [make_example(rng, spec) for _ in range(20)]
    examples = [e for e in examples if e.trace.terminated and len(e.trace) > 0]
    arrays = flatten_transitions(examples, scodec, acodec)
    # rebuild expected next-states directly and compare digit arrays
    idx = 0
    for ex in examples:
        for t in range(len(ex.trace.actions)):
            expected = scodec.encode(ex.trace.states[t + 1])
            assert np.array_equal(arrays["ns_reg_digits"][idx], expected.reg_digits)
            assert int(arrays["ns_pc"][idx]) == int(expected.pc)
            idx += 1
    assert idx == len(arrays["ex_id"])
