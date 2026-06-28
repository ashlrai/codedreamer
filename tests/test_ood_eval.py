"""Fast smoke test for execwm.eval.ood_eval.

Builds a TINY UNTRANED model and runs ``evaluate_split`` on a handful of
generated examples — no training (that would be too slow). Asserts the returned
report has the right keys, that the scalar metrics live in [0, 1], and that the
rollout-horizon curve is a list of valid entries. Untrained accuracy is
essentially random, so we only check ranges, not quality.
"""

from __future__ import annotations

import math

import torch

from execwm.data.dataset import collect_examples
from execwm.data.state_codec import CodecConfig
from execwm.eval.ood_eval import (compare_indist_vs_ood, evaluate_split,
                                   gather_ood_examples, model_reg_shape,
                                   spec_reg_shape)
from execwm.substrate.generators import GenSpec, default_axes
from execwm.train.train_m1 import build


def _tiny_setup():
    spec = GenSpec(num_vars=3, num_inputs=2, max_depth=1, num_stmts=3,
                   max_const=4, max_input_val=4, max_loop_count=2)
    codec_cfg = CodecConfig(max_digits=4, base=10, max_pc=128)
    model, scodec, acodec = build(spec, codec_cfg, d_model=16, n_heads=2,
                                  enc_layers=1, dyn_layers=1)
    model.to(torch.device("cpu"))
    return spec, codec_cfg, model, scodec, acodec


def _in_unit_interval(x: float) -> bool:
    return math.isnan(x) or (0.0 <= x <= 1.0 + 1e-6)


def test_evaluate_split_shape_and_ranges():
    spec, _cfg, model, scodec, acodec = _tiny_setup()
    examples, _ = collect_examples(spec, 6, lambda ex: True, seed=0,
                                   scodec=scodec, acodec=acodec)

    rollout_k = 6
    report = evaluate_split(model, scodec, acodec, examples,
                            device=torch.device("cpu"), max_len=10,
                            rollout_k=rollout_k)

    # right keys
    for key in ("step_exact_match", "per_var_acc", "rollout_horizon", "n",
                "n_episodes"):
        assert key in report, f"missing key {key}"

    # scalar metrics in [0, 1]
    assert _in_unit_interval(report["step_exact_match"])
    assert _in_unit_interval(report["per_var_acc"])

    # rollout horizon is a list of valid entries
    horizon = report["rollout_horizon"]
    assert isinstance(horizon, list)
    assert len(horizon) == rollout_k
    assert all(_in_unit_interval(v) for v in horizon)

    assert report["n"] >= 0
    assert report["n_episodes"] >= 1


def test_shape_helpers_agree():
    spec, _cfg, _model, scodec, _acodec = _tiny_setup()
    assert model_reg_shape(scodec) == spec_reg_shape(spec)


def test_gather_ood_examples_metric_threshold():
    # The magnitude axis keeps the base register shape; build a model/codec for
    # its train_spec and confirm gathered OOD examples clear the test threshold.
    base = GenSpec(num_vars=3, num_inputs=2, max_depth=1, num_stmts=3,
                   max_loop_count=2)
    codec_cfg = CodecConfig(max_digits=9, base=10, max_pc=256)
    axis = {a.name: a for a in default_axes(base)}["magnitude"]
    _model, scodec, acodec = build(axis.train_spec, codec_cfg, d_model=16,
                                   n_heads=2, enc_layers=1, dyn_layers=1)

    from execwm.substrate.generators import realized_metrics
    ood = gather_ood_examples(axis, scodec, acodec, n=4, seed=1)
    assert len(ood) == 4
    assert all(realized_metrics(ex)[axis.metric] >= axis.test_min for ex in ood)


def test_compare_skips_on_shape_mismatch():
    # A model built for a 3-var base cannot evaluate the nesting axis, which
    # widens num_vars to 8 -> different register shape -> skip record.
    base = GenSpec(num_vars=3, num_inputs=2, max_depth=1, num_stmts=3)
    codec_cfg = CodecConfig(max_digits=4, base=10, max_pc=128)
    model, scodec, acodec = build(base, codec_cfg, d_model=16, n_heads=2,
                                  enc_layers=1, dyn_layers=1)
    axis = {a.name: a for a in default_axes(base)}["nesting_depth"]
    report = compare_indist_vs_ood(model, scodec, acodec, axis, n=2,
                                   device=torch.device("cpu"))
    assert report["skipped"] is True
    assert "mismatch" in report["reason"]
