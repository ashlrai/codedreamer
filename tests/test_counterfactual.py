"""Tests for the counterfactual intervention metric (M2 centerpiece).

Two kinds of test:
* A model-free *correctness* test: the intervention helpers must produce real
  counterfactuals — the VM ``step`` result must actually differ from the
  un-intervened transition — and must never trap on an undefined register read.
* A fast smoke test with a tiny UNTRAINED model: ``evaluate_counterfactual``
  returns a well-formed dict with metrics in ``[0, 1]``. No training happens here.
"""

import random

import torch

from execwm.data.state_codec import CodecConfig
from execwm.eval.counterfactual import (_read_regs, _state_equal,
                                        evaluate_counterfactual,
                                        identity_baseline, intervene_action,
                                        intervene_register, make_action_pairs,
                                        make_register_pairs,
                                        sample_base_transitions)
from execwm.substrate.generators import GenSpec
from execwm.substrate.vm import VMError, step
from execwm.train.train_m1 import build

# Small but representative spec/codec used across the tests.
_SPEC = GenSpec(num_vars=3, num_temps=6, max_depth=1, num_stmts=3,
                max_const=4, max_input_val=4, max_loop_count=2)
_CODEC = CodecConfig(max_digits=4, base=10, max_pc=128)


def _step_no_undefined_read(state, instr):
    """step() raises VMError only on a read of an undefined register; assert it
    doesn't (interventions must keep all reads defined)."""
    try:
        return step(state, instr)
    except VMError as exc:  # pragma: no cover - the assertion is the point
        raise AssertionError(f"intervention trapped on undefined read: {exc}")


def test_register_intervention_is_a_real_counterfactual():
    rng = random.Random(0)
    base = sample_base_transitions(_SPEC, 60, seed=1, codec_cfg=_CODEC)
    assert base, "expected some base transitions"

    produced = 0
    for state, instr in base:
        res = intervene_register(state, instr, rng, value_range=(-12, 12))
        if res is None:
            continue
        produced += 1
        mod_state, kept_instr = res
        assert kept_instr is instr, "register intervention must keep the instruction"
        # No undefined-read trap, and the counterfactual genuinely differs.
        orig_next = _step_no_undefined_read(state, instr)
        cf_next = _step_no_undefined_read(mod_state, kept_instr)
        assert not _state_equal(cf_next, orig_next), (
            "register intervention produced an identical next state")
        # Exactly one register's value was changed vs the original state.
        changed = [n for n in state.regs
                   if mod_state.regs[n] != state.regs[n]]
        assert len(changed) == 1
        assert mod_state.regs[changed[0]] != state.regs[changed[0]]

    assert produced >= 5, f"too few register interventions produced ({produced})"


def test_action_intervention_is_a_real_counterfactual():
    rng = random.Random(2)
    base = sample_base_transitions(_SPEC, 60, seed=3, codec_cfg=_CODEC)
    assert base

    produced = 0
    for state, instr in base:
        res = intervene_action(state, instr, rng)
        if res is None:
            continue
        produced += 1
        kept_state, new_instr = res
        assert kept_state is state, "action intervention must keep the state"
        assert new_instr != instr, "action intervention must change the instruction"
        orig_next = _step_no_undefined_read(state, instr)
        cf_next = _step_no_undefined_read(state, new_instr)
        assert not cf_next.error, "safe action swap should never trap"
        assert not _state_equal(cf_next, orig_next), (
            "action intervention produced an identical next state")

    assert produced >= 5, f"too few action interventions produced ({produced})"


def test_read_regs_matches_operands():
    base = sample_base_transitions(_SPEC, 40, seed=5, codec_cfg=_CODEC)
    for state, instr in base:
        reads = _read_regs(instr)
        # Every "read" register name must be defined in the state where the
        # instruction actually executed (else the real trace would have trapped).
        for name in reads:
            assert state.regs[name] is not None


def test_identity_baseline_in_range():
    rng = random.Random(7)
    base = sample_base_transitions(_SPEC, 40, seed=9, codec_cfg=_CODEC)
    reg_pairs = make_register_pairs(base, rng, value_range=(-9, 9))
    act_pairs = make_action_pairs(base, rng)
    for pairs in (reg_pairs, act_pairs):
        b = identity_baseline(pairs)
        assert 0.0 <= b <= 1.0
    assert identity_baseline([]) == 0.0


def test_evaluate_counterfactual_smoke_untrained():
    """Tiny untrained model: the eval returns a well-formed dict in [0, 1]. No
    training — just exercise the encode/predict/grade plumbing end to end."""
    model, scodec, acodec = build(_SPEC, _CODEC, d_model=32, n_heads=2,
                                  enc_layers=1, dyn_layers=1)
    device = torch.device("cpu")
    model.to(device)

    rng = random.Random(11)
    base = sample_base_transitions(_SPEC, 24, seed=13, codec_cfg=_CODEC)
    reg_pairs = make_register_pairs(base, rng, value_range=(-9, 9))
    act_pairs = make_action_pairs(base, rng)
    assert reg_pairs and act_pairs

    for pairs in (reg_pairs, act_pairs):
        res = evaluate_counterfactual(model, scodec, acodec, pairs, device)
        assert set(res) == {"n", "exact_match", "per_var", "n_skipped"}
        assert res["n"] + res["n_skipped"] == len(pairs)
        assert res["n"] > 0
        assert 0.0 <= res["exact_match"] <= 1.0
        assert 0.0 <= res["per_var"] <= 1.0

    # Empty input is handled gracefully.
    empty = evaluate_counterfactual(model, scodec, acodec, [], device)
    assert empty == {"n": 0, "exact_match": 0.0, "per_var": 0.0, "n_skipped": 0}
