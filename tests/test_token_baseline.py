"""Fast unit tests for the token-space baseline (no real training).

Covers the three load-bearing contracts:
  * the serializer round-trips (state_to_tokens -> tokens_to_state_labels is the
    identity on valid label dicts),
  * a tiny untrained TokenBaseline forwards + greedy-decodes into correctly
    shaped, parseable next-state labels,
  * evaluate_token_baseline returns metrics in [0, 1] on a few real examples.
"""

from __future__ import annotations

import random

import torch

from execwm.data.action_codec import ActionCodec
from execwm.data.dataset import collect_examples
from execwm.data.state_codec import CodecConfig, StateCodec
from execwm.data.torch_data import _STATE_KEYS
from execwm.model.delta import exact_match_labels
from execwm.model.token_baseline import (TokenSerializer, build_token_baseline,
                                         predict_next_labels)
from execwm.substrate.generators import GenSpec
from execwm.substrate.vm import VType
from execwm.train.train_token import evaluate_token_baseline


def _tiny_setup():
    spec = GenSpec(num_vars=2, num_temps=4, max_depth=1, num_stmts=3,
                   max_const=4, max_input_val=4, max_loop_count=2)
    codec_cfg = CodecConfig(max_digits=2, base=10, max_pc=64)
    cfg = spec.config()
    scodec = StateCodec(cfg, codec_cfg)
    acodec = ActionCodec(cfg, codec_cfg)
    serializer = TokenSerializer(scodec, acodec)
    return spec, codec_cfg, scodec, acodec, serializer


def _random_state_labels(serializer: TokenSerializer, N: int, gen: torch.Generator):
    R, C, D = serializer.R, serializer.C, serializer.D
    ri = lambda hi, *shape: torch.randint(0, hi, shape, generator=gen)
    return {
        "reg_type": ri(len(VType), N, R),
        "reg_sign": ri(2, N, R),
        "reg_digits": ri(serializer.base, N, R, D),
        "heap_sign": ri(2, N, C),
        "heap_digits": ri(serializer.base, N, C, D),
        "pc": ri(serializer.max_pc + 1, N),
        "halted": ri(2, N),
        "error": ri(2, N),
    }


def test_serializer_roundtrip():
    _, _, _, _, serializer = _tiny_setup()
    gen = torch.Generator().manual_seed(0)
    s = _random_state_labels(serializer, 16, gen)

    tokens = serializer.state_to_tokens(s)
    assert tokens.shape == (16, serializer.T_state)
    assert tokens.min().item() >= 0 and tokens.max().item() < serializer.vocab_size

    back = serializer.tokens_to_state_labels(tokens)
    for k in _STATE_KEYS:
        assert torch.equal(back[k], s[k]), f"round-trip mismatch on {k}"


def test_forward_and_greedy_decode():
    _, _, _, _, serializer = _tiny_setup()
    model = build_token_baseline(serializer, d_model=32, n_layers=1, n_heads=2)

    gen = torch.Generator().manual_seed(1)
    cur = _random_state_labels(serializer, 4, gen)
    # a trivially valid action label dict (all zeros parses fine via codec ranges)
    act = {
        "op": torch.zeros(4, dtype=torch.long), "dst": torch.zeros(4, dtype=torch.long),
        "a_kind": torch.zeros(4, dtype=torch.long), "a_reg": torch.zeros(4, dtype=torch.long),
        "a_sign": torch.zeros(4, dtype=torch.long),
        "a_digits": torch.zeros(4, serializer.D, dtype=torch.long),
        "b_kind": torch.zeros(4, dtype=torch.long), "b_reg": torch.zeros(4, dtype=torch.long),
        "b_sign": torch.zeros(4, dtype=torch.long),
        "b_digits": torch.zeros(4, serializer.D, dtype=torch.long),
        "list_id": torch.zeros(4, dtype=torch.long), "target": torch.zeros(4, dtype=torch.long),
    }

    # teacher-forced forward produces vocab logits over the full sequence
    input_ids, labels = serializer.build_training_sequence(cur, act, cur)
    logits = model(input_ids)
    assert logits.shape == (4, serializer.T_full - 1, serializer.vocab_size)

    # greedy decode -> parseable, correctly-shaped next-state labels
    pred = predict_next_labels(model, serializer, cur, act, torch.device("cpu"))
    assert pred["reg_type"].shape == (4, serializer.R)
    assert pred["reg_digits"].shape == (4, serializer.R, serializer.D)
    assert pred["pc"].shape == (4,)
    # exact_match_labels accepts the decoded dict (grades without error)
    em = exact_match_labels(pred, cur)
    assert em.shape == (4,) and em.dtype == torch.bool


def test_evaluate_returns_metrics():
    spec, _, scodec, acodec, serializer = _tiny_setup()
    examples, _ = collect_examples(spec, 6, lambda e: True, 0, scodec, acodec)
    model = build_token_baseline(serializer, d_model=32, n_layers=1, n_heads=2)

    out = evaluate_token_baseline(model, serializer, scodec, acodec,
                                  examples, torch.device("cpu"))
    assert set(out) == {"step_exact_match", "per_var_acc", "n"}
    assert out["n"] > 0
    assert 0.0 <= out["step_exact_match"] <= 1.0
    assert 0.0 <= out["per_var_acc"] <= 1.0
