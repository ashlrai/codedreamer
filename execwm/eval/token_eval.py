"""Token-space evaluation path for the causal (counterfactual) and OOD axes.

The grounded-latent world model is graded on counterfactual interventions
(``execwm.eval.counterfactual``) and OOD axes (``execwm.eval.ood_eval``). To make
the project's core CAUSAL-axis comparison apples-to-apples, the token-space
baseline (``execwm.model.token_baseline``) needs the *same* metrics on the *same*
pairs/splits. This module provides that path:

* :func:`evaluate_counterfactual_token` mirrors
  :func:`execwm.eval.counterfactual.evaluate_counterfactual` exactly — same
  skip-on-:class:`EncodeError` logic, same ``{n, exact_match, per_var,
  n_skipped}`` return contract — but grades the TOKEN model by greedy-decoding
  the next-state token sequence and scoring decoded label dicts with the shared
  ``exact_match_labels`` / ``per_var_accuracy_labels`` rules. Callers reuse
  ``sample_base_transitions`` / ``make_register_pairs`` / ``make_action_pairs``
  from :mod:`execwm.eval.counterfactual`, so the latent and token models are
  graded on identical pair sets.
* :func:`evaluate_ood_token` mirrors
  :func:`execwm.eval.ood_eval.evaluate_all_axes` — same per-axis splits, same
  register-shape skip logic — but grades the token model via
  :func:`execwm.train.train_token.evaluate_token_baseline`.

Honesty / performance note
--------------------------
The token baseline's greedy decode is autoregressive and has no KV cache, so it
is slow and fragments (MPS) unified memory. Both functions therefore decode in
small chunks (``chunk`` / ``batch_size`` <= 8 recommended) and release the MPS
cache between chunks. Nothing here sub-samples the pairs/examples — every
provided pair and every collected example is graded; the chunking only bounds
peak memory, it does not drop work. ``per_var`` is accumulated as an
example-count-weighted mean across chunks (the same aggregation
:func:`evaluate_token_baseline` uses), which is exact when ``chunk`` covers all
example in one pass and a close approximation otherwise.

Token-vocabulary limit on program-lengthening axes
---------------------------------------------------
Unlike the grounded latent (whose action encoder embeds ``pc``/``target`` with a
generous table), the token baseline serializes *every* field value through a
single shared value-token block whose size is fixed at construction to
``serializer.max_value`` (dominated by the codec's ``max_pc``). An OOD axis that
lengthens programs (e.g. ``trace_length``) can emit jump ``target`` immediates
that exceed ``max_pc``; those values have no token id and would index past the
embedding table. Rather than crash or silently clamp the target to ``max_pc``
(which would feed the model a *wrong* action and corrupt the metric),
:func:`evaluate_ood_token` detects such axes and SKIPS them with an explicit
reason — the same honest "this model cannot represent these inputs" stance that
:func:`execwm.eval.ood_eval.compare_indist_vs_ood` takes for register-shape
mismatches. (The latent OOD path does not hit this because its embeddings are
not sized by ``max_pc``.)
"""
from __future__ import annotations

import torch

from ..data.action_codec import ActionCodec
from ..data.dataset import flatten_transitions
from ..data.state_codec import CodecConfig, EncodeError, StateCodec
from ..data.torch_data import _ACTION_KEYS, _STATE_KEYS
from ..model.delta import exact_match_labels
from ..model.token_baseline import (TokenBaseline, TokenSerializer,
                                     per_var_accuracy_labels,
                                     predict_next_labels)
from ..substrate.generators import GenSpec, default_axes
from ..substrate.vm import Instr, MachineState, step
from ..train.train_token import evaluate_token_baseline
from .counterfactual import _stack
from .ood_eval import (gather_indist_examples, gather_ood_examples,
                       model_reg_shape, spec_reg_shape)

__all__ = [
    "evaluate_counterfactual_token",
    "evaluate_ood_token",
]


# ---------------------------------------------------------------------------
# Counterfactual interventions, token-model grading
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_counterfactual_token(model: TokenBaseline, serializer: TokenSerializer,
                                  scodec: StateCodec, acodec: ActionCodec,
                                  pairs: list[tuple[MachineState, Instr]], device,
                                  chunk: int = 8) -> dict:
    """Grade the TOKEN model on intervened ``(state, action)`` pairs vs VM truth.

    The token-space twin of
    :func:`execwm.eval.counterfactual.evaluate_counterfactual`: same skip logic,
    same ``{n, exact_match, per_var, n_skipped}`` contract, so a caller can grade
    the latent and token models on the *same* pair set with near-identical call
    sites. For each pair the true next state is ``step(state, action)``; the
    model predicts it by greedy-decoding the next-state token sequence from
    ``[state + action]``. Pairs whose state, action, or true-next fall outside
    the codecs' representable range are skipped (:class:`EncodeError`).

    Greedy decode is autoregressive and memory-heavy, so prediction runs in
    chunks of ``chunk`` (<= 8 recommended) with the MPS cache released between
    chunks. Returns metrics in ``[0, 1]``.
    """
    enc_s: list[dict] = []
    enc_a: list[dict] = []
    enc_t: list[dict] = []
    n_skipped = 0
    for state, instr in pairs:
        try:
            true_next = step(state, instr)
            s = scodec.encode(state).as_dict()
            a = acodec.encode(instr).as_dict()
            t = scodec.encode(true_next).as_dict()
        except EncodeError:
            n_skipped += 1
            continue
        enc_s.append(s)
        enc_a.append(a)
        enc_t.append(t)

    n = len(enc_s)
    if n == 0:
        return {"n": 0, "exact_match": 0.0, "per_var": 0.0, "n_skipped": n_skipped}

    was_training = model.training
    model.eval()
    s_dict = _stack(enc_s, device)
    a_dict = _stack(enc_a, device)
    t_dict = _stack(enc_t, device)

    em_sum = 0.0
    pv_sum = 0.0
    for i in range(0, n, chunk):
        sl = slice(i, min(i + chunk, n))
        s_b = {k: v[sl] for k, v in s_dict.items()}
        a_b = {k: v[sl] for k, v in a_dict.items()}
        t_b = {k: v[sl] for k, v in t_dict.items()}
        pred = predict_next_labels(model, serializer, s_b, a_b, device)
        bsz = t_b["pc"].shape[0]
        em_sum += exact_match_labels(pred, t_b).float().sum().item()
        pv_sum += per_var_accuracy_labels(pred, t_b).item() * bsz
        # autoregressive greedy_decode (no KV cache) fragments MPS unified memory;
        # release the cache each chunk so it can't accumulate to an OOM.
        if getattr(device, "type", None) == "mps":
            torch.mps.empty_cache()

    if was_training:
        model.train()
    return {"n": n, "exact_match": em_sum / n, "per_var": pv_sum / n,
            "n_skipped": n_skipped}


# ---------------------------------------------------------------------------
# OOD axes, token-model grading
# ---------------------------------------------------------------------------


def _tokens_representable(serializer: TokenSerializer, scodec: StateCodec,
                          acodec: ActionCodec, examples: list) -> bool:
    """True iff every (s_t, a_t) -> s_{t+1} of ``examples`` serializes into token
    ids strictly below ``serializer.vocab_size``.

    The serializer's shared value block is sized to ``serializer.max_value`` at
    construction; an OOD field value beyond it (e.g. a jump ``target`` past
    ``max_pc`` in a long program) would index past the embedding table. This is
    the token model's analogue of a register-shape mismatch."""
    examples = [e for e in examples if len(e.trace) > 0]
    if not examples:
        return True
    flat = flatten_transitions(examples, scodec, acodec)
    cur = {k: torch.from_numpy(flat[f"s_{k}"]) for k in _STATE_KEYS}
    act = {k: torch.from_numpy(flat[f"a_{k}"]) for k in _ACTION_KEYS}
    nxt = {k: torch.from_numpy(flat[f"ns_{k}"]) for k in _STATE_KEYS}
    vocab = serializer.vocab_size
    return bool(serializer.state_to_tokens(cur).max().item() < vocab
                and serializer.action_to_tokens(act).max().item() < vocab
                and serializer.state_to_tokens(nxt).max().item() < vocab)


@torch.no_grad()
def evaluate_ood_token(model: TokenBaseline, serializer: TokenSerializer,
                       scodec: StateCodec, acodec: ActionCodec, spec: GenSpec,
                       codec_cfg: CodecConfig, device, n: int = 120,
                       seed: int = 0) -> dict:
    """Grade the TOKEN model in-distribution vs OOD along every canonical axis.

    The token-space twin of :func:`execwm.eval.ood_eval.evaluate_all_axes`. For
    each axis in :func:`default_axes(spec) <execwm.substrate.generators.default_axes>`
    it gathers in-distribution (``train_spec``, below ``train_max``) and OOD
    (``test_spec``, at/above ``test_min``) example splits the same way
    ``ood_eval`` does — encoded with the model's own ``scodec`` — and grades both
    with :func:`evaluate_token_baseline` (greedy single-step exact match +
    per-variable accuracy, ``batch_size`` <= 8 to bound greedy-decode memory).

    A model is tied to one ``(num_regs, num_cells)`` register shape, so axes
    whose ``test_spec`` widens that shape (e.g. nesting depth, program size)
    cannot be evaluated by this model and are SKIPPED — exactly the skip rule
    :func:`execwm.eval.ood_eval.compare_indist_vs_ood` applies. Returns a dict
    mapping ``axis_name -> {indist, ood, delta_exact_match, skipped, reason}``;
    for evaluated axes ``indist``/``ood`` are ``{exact_match, per_var}`` and
    ``reason`` is ``None``; for skipped axes ``indist``/``ood``/
    ``delta_exact_match`` are ``None`` and ``reason`` explains the mismatch.

    ``codec_cfg`` is accepted for signature parity with the latent OOD path and
    documents the codec the splits are encoded under (the live ``scodec`` already
    carries it); ``n`` is kept small because greedy decode is slow.
    """
    m_shape = model_reg_shape(scodec)
    reports: dict[str, dict] = {}
    for axis in default_axes(spec):
        a_shape = spec_reg_shape(axis.test_spec)
        if m_shape != a_shape:
            reports[axis.name] = {
                "skipped": True,
                "reason": (f"register shape mismatch: model {m_shape} != axis "
                           f"{a_shape}; train a token model on this axis' spec "
                           f"to evaluate it"),
                "indist": None,
                "ood": None,
                "delta_exact_match": None,
            }
            continue

        indist_ex = gather_indist_examples(axis, scodec, acodec, n, seed)
        ood_ex = gather_ood_examples(axis, scodec, acodec, n, seed + 1)

        if not (_tokens_representable(serializer, scodec, acodec, indist_ex)
                and _tokens_representable(serializer, scodec, acodec, ood_ex)):
            reports[axis.name] = {
                "skipped": True,
                "reason": (f"token vocab cannot represent axis values: a field "
                           f"(e.g. jump target) exceeds the serializer's "
                           f"max_value={serializer.max_value} (vocab "
                           f"{serializer.vocab_size}); raise the codec's max_pc "
                           f"to evaluate this axis with the token baseline"),
                "indist": None,
                "ood": None,
                "delta_exact_match": None,
            }
            continue

        indist = evaluate_token_baseline(model, serializer, scodec, acodec,
                                         indist_ex, device, batch_size=8)
        ood = evaluate_token_baseline(model, serializer, scodec, acodec,
                                      ood_ex, device, batch_size=8)

        reports[axis.name] = {
            "skipped": False,
            "reason": None,
            "indist": {"exact_match": indist["step_exact_match"],
                       "per_var": indist["per_var_acc"]},
            "ood": {"exact_match": ood["step_exact_match"],
                    "per_var": ood["per_var_acc"]},
            "delta_exact_match": (indist["step_exact_match"]
                                  - ood["step_exact_match"]),
        }
    return reports
