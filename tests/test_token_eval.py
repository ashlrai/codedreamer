"""Fast plumbing tests for the token-space causal + OOD eval path.

These assert the *contract*, not accuracy: a tiny UNTRAINED TokenBaseline on CPU
suffices because we only exercise the encode -> greedy-decode -> grade wiring.

Two load-bearing claims:
  * ``evaluate_counterfactual_token`` returns the same
    ``{n, exact_match, per_var, n_skipped}`` dict shape as the latent
    ``evaluate_counterfactual``, with metrics in [0, 1] and correct n_skipped
    accounting, AND it grades the SAME pairs to the SAME ``n``/``n_skipped`` as
    the latent path (the pair set is model-agnostic, so a caller can compare
    latent vs token on identical pairs).
  * ``evaluate_ood_token`` returns a per-axis report dict of the right shape, and
    register-shape-changing axes are marked skipped (a single model is tied to
    one register shape) — matching ``ood_eval``'s skip rule.
"""

from __future__ import annotations

import random

import torch

from execwm.data.action_codec import ActionCodec
from execwm.data.state_codec import CodecConfig, StateCodec
from execwm.eval.counterfactual import (evaluate_counterfactual,
                                        make_action_pairs, make_register_pairs,
                                        sample_base_transitions)
from execwm.eval.token_eval import (evaluate_counterfactual_token,
                                    evaluate_ood_token)
from execwm.model.token_baseline import TokenSerializer, build_token_baseline
from execwm.substrate.generators import GenSpec
from execwm.train.train_m1 import build as build_latent

# Tiny but representative spec/codec used across the tests.
_SPEC = GenSpec(num_vars=2, num_temps=4, max_depth=1, num_stmts=3,
                max_const=4, max_input_val=4, max_loop_count=2)
_CODEC = CodecConfig(max_digits=2, base=10, max_pc=64)
_DEVICE = torch.device("cpu")


def _tiny_token_model():
    cfg = _SPEC.config()
    scodec = StateCodec(cfg, _CODEC)
    acodec = ActionCodec(cfg, _CODEC)
    serializer = TokenSerializer(scodec, acodec)
    model = build_token_baseline(serializer, d_model=32, n_layers=1, n_heads=2)
    model.to(_DEVICE)
    return model, serializer, scodec, acodec


def _make_pairs():
    rng = random.Random(11)
    base = sample_base_transitions(_SPEC, 24, seed=13, codec_cfg=_CODEC)
    reg_pairs = make_register_pairs(base, rng, value_range=(-9, 9))
    act_pairs = make_action_pairs(base, rng)
    assert reg_pairs and act_pairs
    return reg_pairs, act_pairs


def test_counterfactual_token_contract_and_ranges():
    """Well-formed dict, metrics in [0, 1], n + n_skipped == len(pairs)."""
    model, serializer, scodec, acodec = _tiny_token_model()
    reg_pairs, act_pairs = _make_pairs()

    for pairs in (reg_pairs, act_pairs):
        res = evaluate_counterfactual_token(model, serializer, scodec, acodec,
                                            pairs, _DEVICE, chunk=4)
        assert set(res) == {"n", "exact_match", "per_var", "n_skipped"}
        assert res["n"] + res["n_skipped"] == len(pairs)
        assert res["n"] > 0
        assert 0.0 <= res["exact_match"] <= 1.0
        assert 0.0 <= res["per_var"] <= 1.0

    # Empty input is handled gracefully (same as the latent path).
    empty = evaluate_counterfactual_token(model, serializer, scodec, acodec,
                                          [], _DEVICE)
    assert empty == {"n": 0, "exact_match": 0.0, "per_var": 0.0, "n_skipped": 0}


def test_counterfactual_token_grades_same_pairs_as_latent():
    """The pair set is model-agnostic, so token and latent paths must skip/keep
    exactly the same pairs (identical n and n_skipped)."""
    token_model, serializer, scodec, acodec = _tiny_token_model()
    # A latent model built on the identical spec/codec -> identical codecs.
    latent_model, lscodec, lacodec = build_latent(_SPEC, _CODEC, d_model=32,
                                                  n_heads=2, enc_layers=1,
                                                  dyn_layers=1)
    latent_model.to(_DEVICE)
    reg_pairs, act_pairs = _make_pairs()

    for pairs in (reg_pairs, act_pairs):
        tok = evaluate_counterfactual_token(token_model, serializer, scodec,
                                            acodec, pairs, _DEVICE, chunk=4)
        lat = evaluate_counterfactual(latent_model, lscodec, lacodec, pairs,
                                      _DEVICE)
        assert tok["n"] == lat["n"]
        assert tok["n_skipped"] == lat["n_skipped"]
        assert tok["n"] + tok["n_skipped"] == len(pairs)


def test_counterfactual_token_chunking_is_invariant():
    """Chunk size only bounds memory; exact_match must not depend on it."""
    model, serializer, scodec, acodec = _tiny_token_model()
    reg_pairs, _ = _make_pairs()
    r1 = evaluate_counterfactual_token(model, serializer, scodec, acodec,
                                       reg_pairs, _DEVICE, chunk=1)
    r8 = evaluate_counterfactual_token(model, serializer, scodec, acodec,
                                       reg_pairs, _DEVICE, chunk=8)
    assert r1["n"] == r8["n"]
    assert r1["exact_match"] == r8["exact_match"]


def test_ood_token_report_shape_and_skips():
    """Per-axis dict shape; shape-changing axes are skipped, evaluable axes carry
    {exact_match, per_var} for indist/ood plus a numeric delta."""
    # Generation-safe spec for the OOD axes: the trace-length / magnitude axes
    # deepen programs, so we use the default temp pool (num_temps=14) instead of
    # the tiny counterfactual spec, mirroring tests/test_ood_eval.py's setup.
    ood_spec = GenSpec(num_vars=3, num_inputs=2, max_depth=1, num_stmts=3,
                       max_const=4, max_input_val=4, max_loop_count=2)
    ood_codec = CodecConfig(max_digits=4, base=10, max_pc=128)
    cfg = ood_spec.config()
    scodec = StateCodec(cfg, ood_codec)
    acodec = ActionCodec(cfg, ood_codec)
    serializer = TokenSerializer(scodec, acodec)
    model = build_token_baseline(serializer, d_model=32, n_layers=1, n_heads=2)
    model.to(_DEVICE)

    reports = evaluate_ood_token(model, serializer, scodec, acodec, ood_spec,
                                 ood_codec, _DEVICE, n=6, seed=0)
    assert reports, "expected at least one axis"

    n_skipped = 0
    n_evaluated = 0
    for name, rep in reports.items():
        assert set(rep) == {"skipped", "reason", "indist", "ood",
                            "delta_exact_match"}, name
        if rep["skipped"]:
            n_skipped += 1
            assert isinstance(rep["reason"], str) and rep["reason"]
            assert rep["indist"] is None
            assert rep["ood"] is None
            assert rep["delta_exact_match"] is None
        else:
            n_evaluated += 1
            assert rep["reason"] is None
            for side in ("indist", "ood"):
                assert set(rep[side]) == {"exact_match", "per_var"}
                assert 0.0 <= rep[side]["exact_match"] <= 1.0
                assert 0.0 <= rep[side]["per_var"] <= 1.0
            assert isinstance(rep["delta_exact_match"], float)

    # At least one canonical axis widens the register shape (nesting / program
    # size), so a single fixed-shape model must skip it.
    assert n_skipped >= 1, "expected at least one shape-changing axis to skip"
    assert n_evaluated >= 1, "expected at least one shape-preserving axis"
