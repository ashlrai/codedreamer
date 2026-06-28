"""Fast smoke tests for the frozen-encoder linear probes + causal intervention.

These use a TINY, UNTRAINED model: we only check that the probing machinery runs
end-to-end and returns well-formed values (accuracies in [0, 1], a flip-rate in
[0, 1]). We deliberately do NOT train the world model here -- the >=95% accuracy
claim is exercised by the demo in ``execwm/eval/probes.py``, not the unit test.
"""

import random

import torch

from execwm.data.state_codec import CodecConfig
from execwm.data.torch_data import _STATE_KEYS
from execwm.eval.probes import (LinearProbes, causal_intervention,
                                collect_state_tensors, fit_linear_probes,
                                heads_accuracy, probe_accuracy)
from execwm.substrate.generators import GenSpec, make_example
from execwm.train.train_m1 import build


def _setup(seed=0, n=8):
    spec = GenSpec(num_vars=3, num_temps=6, max_depth=1, num_stmts=3,
                   max_const=4, max_input_val=4, max_loop_count=2)
    codec = CodecConfig(max_digits=4, base=10, max_pc=128)
    # tiny untrained model -- fast to build and encode
    model, scodec, _acodec = build(spec, codec, d_model=64, n_heads=4,
                                   enc_layers=1, dyn_layers=1)
    device = torch.device("cpu")
    model.to(device)
    rng = random.Random(seed)
    examples = []
    while len(examples) < n:
        e = make_example(rng, spec)
        if len(e.trace) > 0:
            examples.append(e)
    return model, scodec, examples, device


def _valid_acc_dict(acc):
    assert isinstance(acc, dict) and acc
    for k, v in acc.items():
        assert isinstance(v, float), f"{k} -> {type(v)}"
        assert 0.0 <= v <= 1.0, f"{k} = {v} out of [0,1]"


def test_collect_state_tensors():
    _model, scodec, examples, device = _setup()
    state = collect_state_tensors(examples, scodec, max_states=200, device=device)
    assert set(state) == set(_STATE_KEYS)
    n = next(iter(state.values())).shape[0]
    assert 0 < n <= 200
    for k in _STATE_KEYS:
        assert state[k].shape[0] == n
        assert state[k].dtype == torch.int64


def test_fit_and_accuracy():
    model, scodec, examples, device = _setup()
    state = collect_state_tensors(examples, scodec, max_states=300, device=device)

    probes = fit_linear_probes(model, state, device, epochs=20)
    assert isinstance(probes, LinearProbes)
    assert set(probes.probes) == set(_STATE_KEYS)

    acc = probe_accuracy(model, probes, state, device)
    _valid_acc_dict(acc)
    assert "reg_composite" in acc

    # heads baseline produces the same well-formed metric dict
    h_acc = heads_accuracy(model, state, device)
    _valid_acc_dict(h_acc)
    assert set(h_acc) == set(acc)


def test_causal_intervention():
    model, scodec, examples, device = _setup(seed=1)
    state = collect_state_tensors(examples, scodec, max_states=200, device=device)
    probes = fit_linear_probes(model, state, device, epochs=20)

    ci = causal_intervention(model, probes, state, device, max_examples=64)
    assert isinstance(ci, dict)
    assert 0.0 <= ci["flip_rate"] <= 1.0
    assert 0.0 <= ci["others_stable_rate"] <= 1.0
    assert ci["field"] == "reg_sign"
    assert isinstance(ci["n"], int)

    # decoding with the probe itself should also yield a valid flip-rate
    ci_p = causal_intervention(model, probes, state, device, use_heads=False)
    assert 0.0 <= ci_p["flip_rate"] <= 1.0
    assert ci_p["decoder"] == "probe"
