"""Round-trip and exact-match tests for the state codec.

The codec is the operational definition of "exact match", so two properties are
load-bearing: encode->decode is the identity on representable states, and
``exact_match`` is true for a state vs itself and false under any single change.
"""

import random

import numpy as np
import pytest

from execwm.data.state_codec import CodecConfig, EncodeError, StateCodec
from execwm.substrate.dsl import make_config
from execwm.substrate.generators import GenSpec, make_example
from execwm.substrate.vm import Config, MachineState, VType


def _random_state(rng: random.Random, cfg: Config, vrange: int) -> MachineState:
    regs, types = {}, {}
    for name in cfg.reg_names:
        roll = rng.random()
        if roll < 0.2:
            regs[name], types[name] = None, VType.UNDEF
        elif roll < 0.6:
            regs[name], types[name] = rng.randint(-vrange, vrange), VType.INT
        else:
            regs[name], types[name] = rng.randint(0, 1), VType.BOOL
    heap = [[rng.randint(-vrange, vrange) for _ in range(cfg.list_len)]
            for _ in range(cfg.num_lists)]
    pc = rng.randint(0, 200)
    return MachineState(regs=regs, types=types, heap=heap, pc=pc,
                        halted=bool(rng.getrandbits(1)),
                        error=bool(rng.getrandbits(1)))


def test_int_roundtrip():
    codec = StateCodec(make_config(2), CodecConfig(max_digits=6, base=10))
    for v in [0, 1, -1, 9, -9, 12345, -999999, 524287, -524287]:
        sign, digits = codec._encode_int(v)
        assert codec._decode_int(sign, digits) == v


def test_state_roundtrip_bit_exact():
    rng = random.Random(0)
    cfg = make_config(num_vars=5, num_temps=4, num_lists=2, list_len=4)
    codec = StateCodec(cfg, CodecConfig(max_digits=7, base=10, max_pc=256))
    for _ in range(2000):
        st = _random_state(rng, cfg, vrange=900_000)
        dec = codec.decode(codec.encode(st))
        # types and pc/flags exact
        assert dec.types == st.types
        assert dec.pc == st.pc and dec.halted == st.halted and dec.error == st.error
        assert dec.heap == st.heap
        # valued registers exact; UNDEF registers decode to None
        for name in cfg.reg_names:
            if st.types[name] in (VType.INT, VType.BOOL):
                assert dec.regs[name] == st.regs[name]
            else:
                assert dec.regs[name] is None


def test_exact_match_self_and_perturbation():
    rng = random.Random(1)
    cfg = make_config(num_vars=4, num_temps=2)
    codec = StateCodec(cfg, CodecConfig(max_digits=6))
    for _ in range(300):
        st = _random_state(rng, cfg, vrange=5000)
        e = codec.encode(st)
        assert codec.exact_match(e, e)
        # perturb one defined register's value -> must break exact match
        defined = [n for n in cfg.reg_names if st.types[n] in (VType.INT, VType.BOOL)]
        if defined:
            st2 = st.copy()
            name = defined[0]
            st2.regs[name] = (st2.regs[name] or 0) + 1
            st2.types[name] = VType.INT
            assert not codec.exact_match(e, codec.encode(st2))
        # perturb pc -> must break
        st3 = st.copy(); st3.pc = st.pc + 1
        assert not codec.exact_match(e, codec.encode(st3))


def test_out_of_range_raises():
    codec = StateCodec(make_config(1), CodecConfig(max_digits=2, base=10))  # max 99
    with pytest.raises(EncodeError):
        codec._encode_int(100)


def test_encode_real_trace_states():
    """Every state of a generated trace must be encodable with a wide codec."""
    rng = random.Random(2)
    spec = GenSpec(num_vars=4, max_depth=2, num_stmts=4, max_const=5,
                   max_input_val=5, max_loop_count=3)
    codec = StateCodec(spec.config(), CodecConfig(max_digits=12, max_pc=spec.max_steps))
    for _ in range(200):
        ex = make_example(rng, spec)
        for st in ex.trace.states:
            enc = codec.encode(st)
            assert codec.exact_match(enc, codec.encode(st))
