"""Tests for the neurosymbolic readout analysis + executor (the M3.5 spike plumbing).

Contract/invariant tests on a tiny untrained model on CPU -- they assert the metrics
are well-formed and that the key invariants hold, NOT that the untrained model is
accurate (accuracy is what `scripts/neurosym_spike.py` measures on the trained model):

* `field_breakdown` returns the expected keys, all accuracies in [0, 1], and the
  invariant `em_digits_oracle >= em_learned` (oracling the digit payload can only help);
* `neurosym_execute` produces one StepRecord per executed step with consistent flags;
* `evaluate_executor` returns the documented aggregate keys in range.
"""

import random

import torch

from execwm.data.action_codec import ActionCodec
from execwm.data.dataset import collect_examples
from execwm.data.state_codec import CodecConfig, StateCodec
from execwm.eval.neurosym import field_breakdown
from execwm.eval.neurosym_exec import demo_trace, evaluate_executor, neurosym_execute
from execwm.eval.demo_backend import render_trace_html, summary_md
from execwm.model.world_model import GroundedLatentWM, ModelConfig
from execwm.substrate.generators import GenSpec
from execwm.substrate.vm import Op

_SPEC = GenSpec(num_vars=3, num_inputs=2, num_temps=8, max_depth=2, num_stmts=4,
                max_const=4, max_input_val=4, max_loop_count=2,
                arith_ops=(Op.ADD, Op.SUB), use_heap=True, num_lists=1, list_len=4,
                max_steps=64)
_CODEC = CodecConfig(max_digits=4, base=10, max_pc=64)
_DEVICE = torch.device("cpu")


def _build():
    cfg = _SPEC.config()
    scodec = StateCodec(cfg, _CODEC)
    acodec = ActionCodec(cfg, _CODEC)
    mcfg = ModelConfig.from_codec(len(cfg.reg_names), scodec.num_cells,
                                  cfg.num_lists, _CODEC,
                                  d_model=64, n_heads=4, enc_layers=2, dyn_layers=2)
    torch.manual_seed(0)
    model = GroundedLatentWM(mcfg).to(_DEVICE).eval()
    return model, scodec, acodec


def _examples(n=12, seed=0):
    _, scodec, acodec = _build()
    ex, _ = collect_examples(_SPEC, n, lambda e: True, seed, scodec, acodec)
    return ex


def test_field_breakdown_keys_and_oracle_invariant():
    model, scodec, acodec = _build()
    ex = _examples(12, seed=1)
    out = field_breakdown(model, ex, scodec, acodec, _DEVICE, max_len=16, batch_size=8)
    for k in ("em_learned", "em_digits_oracle", "pc", "written_digits", "n"):
        assert k in out
    assert out["n"] > 0
    # all reported accuracies live in [0, 1] (NaN allowed for empty op-families)
    for k, v in out.items():
        if k == "n" or v != v:  # skip count and NaN
            continue
        assert 0.0 <= v <= 1.0, f"{k}={v} out of range"
    # the core invariant: a perfect-ALU digit readout can only raise exact-match
    assert out["em_digits_oracle"] >= out["em_learned"] - 1e-9


def test_neurosym_execute_records_consistent():
    model, scodec, acodec = _build()
    ex = _examples(6, seed=2)[0]
    recs, nsteps, full = neurosym_execute(model, scodec, acodec, ex, _DEVICE)
    assert nsteps == len(recs)
    assert isinstance(full, bool)
    for r in recs:
        assert isinstance(r.control_ok, bool)
        assert isinstance(r.state_exact, bool)
        # a state can only be exact if control picked the right next pc
        if r.state_exact:
            assert r.control_ok
    # full == every step exact
    assert full == (nsteps > 0 and all(r.state_exact for r in recs))


def test_evaluate_executor_aggregate_keys():
    model, scodec, acodec = _build()
    ex = _examples(10, seed=3)
    agg = evaluate_executor(model, scodec, acodec, ex, _DEVICE)
    for k in ("full_trajectory_success", "per_step_state_exact", "control_accuracy",
              "mean_exact_horizon", "n_programs", "n_steps"):
        assert k in agg
    assert agg["n_programs"] > 0
    assert 0.0 <= agg["per_step_state_exact"] <= 1.0
    assert 0.0 <= agg["control_accuracy"] <= 1.0


def test_demo_trace_and_render():
    model, scodec, acodec = _build()
    ex = _examples(6, seed=4)[0]
    d = demo_trace(model, scodec, acodec, ex, _DEVICE)
    assert set(d) >= {"reg_names", "init", "steps", "summary"}
    assert 0.0 <= d["summary"]["pure_net_exact_frac"] <= 1.0
    assert 0.0 <= d["summary"]["neurosym_exact_frac"] <= 1.0
    for s in d["steps"]:
        assert set(s) >= {"instr", "ground_truth", "pure_net", "neurosym",
                          "pure_exact", "neurosym_exact"}
    html = render_trace_html(d)
    assert "<table" in html and "neurosym" in html
    assert "exact-match" in summary_md(400, {"pure_net": 0.0, "neurosym": 0.9})


def test_demo_engine_intervention():
    import os
    from execwm.eval.demo_backend import DemoEngine
    if not os.path.exists("artifacts/neurosym_model.pt"):
        return  # checkpoint optional in CI; skip if absent
    eng = DemoEngine()
    target, original, d = eng.intervened_trace(20, 0, 7)
    assert target is not None and d is not None
    assert 0.0 <= d["summary"]["neurosym_exact_frac"] <= 1.0
