"""Tests for the divergence-aware planner (M3 step-3 payoff plumbing).

These use a TINY, UNTRAINED ``EditConditionedWM`` on CPU — they prove the
plumbing (zero VM calls, no latent rollout, end-to-end search, label round-trip),
NOT that an untrained model solves goals. Load-bearing properties:

* ``EditConditionedWMScorer`` returns a finite float and keeps ``executions == 0``.
* The grounding-head label -> ``MachineState`` reconstruction round-trips a state.
* ``divergence_beam_plan`` runs end-to-end on a ``make_goal_task`` task without
  error (solved or not — the untrained model's accuracy is not asserted), and the
  scorer still reports zero VM executions afterward.
"""

import random

import torch

from execwm.data.state_codec import CodecConfig
from execwm.plan.divergence_planner import (EditConditionedWMScorer,
                                            decode_logits_to_state,
                                            divergence_beam_plan,
                                            state_from_labels)
from execwm.plan.goal_tasks import goal_distance, make_goal_task
from execwm.plan.search_baseline import SearchResult
from execwm.substrate.edits import EditConfig
from execwm.substrate.generators import GenSpec
from execwm.substrate.vm import run_traced
from execwm.train.train_edit import build

# Small, fast spec: short straight-line-ish programs, no heap (fewer traps).
_SPEC = GenSpec(num_vars=3, num_inputs=2, num_temps=6, max_depth=1, num_stmts=3,
                max_const=4, max_input_val=4, max_loop_count=2, use_heap=False)
_CODEC = CodecConfig(max_digits=4, base=10, max_pc=128)
_EDIT_CFG = EditConfig(max_program_len=128)
_DEVICE = torch.device("cpu")


def _tiny_scorer():
    """A tiny untrained EditConditionedWM wrapped in the scorer (+ its codecs)."""
    model, scodec, ecodec = build(_SPEC, _CODEC, _EDIT_CFG, d_model=32, n_heads=4,
                                  enc_layers=1, dyn_layers=1)
    scorer = EditConditionedWMScorer(model, scodec, ecodec, device=_DEVICE)
    return scorer, scodec, ecodec


def _task(seed: int, edit_budget: int = 2):
    return make_goal_task(random.Random(seed), _SPEC, codec_cfg=_CODEC,
                          edit_budget=edit_budget)


# ---------------------------------------------------------------------------
# Scorer: finite float, zero executions
# ---------------------------------------------------------------------------

def test_scorer_returns_finite_float_zero_executions():
    scorer, _, _ = _tiny_scorer()
    task = _task(0)
    score = scorer(task.base_bytecode, task.init_state, task.goal)
    assert isinstance(score, float)
    # finite OR the well-defined +inf (undefined target reg) — never NaN.
    assert score == score  # not NaN
    assert score >= 0.0
    assert scorer.executions == 0


def test_score_edit_zero_executions_and_caches():
    scorer, _, _ = _tiny_scorer()
    task = _task(1)
    root = list(task.base_bytecode)
    root_sig = tuple(root)
    # Seed the root with its true executed states (as the planner would).
    trace = run_traced(root, task.init_state, max_steps=task.max_steps)
    scorer.seed_base_states(root_sig, list(trace.states))

    from execwm.substrate.edits import enumerate_valid_edits, apply_edit
    edit = enumerate_valid_edits(root, task.config, None, task.edit_config)[0]
    child = apply_edit(root, edit)
    s = scorer.score_edit(root_sig, edit, child, task.init_state, task.goal)
    assert isinstance(s, float) and s == s
    assert scorer.executions == 0
    # The child's predicted states are cached for its own children.
    assert tuple(child) in scorer._cache
    assert len(scorer._cache[tuple(child)]) >= 1


# ---------------------------------------------------------------------------
# Label -> MachineState reconstruction round-trips a known state
# ---------------------------------------------------------------------------

def test_state_reconstruction_round_trips():
    _, scodec, _ = _tiny_scorer()
    task = _task(2)
    # A concrete, codec-encodable state from a real trace.
    trace = run_traced(task.base_bytecode, task.init_state, max_steps=task.max_steps)
    state = trace.final_state

    labels = scodec.encode(state).as_dict()
    recon = state_from_labels(scodec, labels)

    # Exact codec match (the operational notion of state equality).
    assert scodec.exact_match(scodec.encode(state), scodec.encode(recon))
    # And goal distance is preserved (the planner reads it via goal_distance).
    assert goal_distance(task.goal, recon) == goal_distance(task.goal, state)


def test_decode_logits_argmax_recovers_state():
    """A one-hot 'logits' tensor at the true labels argmaxes back to the state."""
    _, scodec, _ = _tiny_scorer()
    task = _task(3)
    state = run_traced(task.base_bytecode, task.init_state,
                       max_steps=task.max_steps).final_state
    enc = scodec.encode(state).as_dict()

    # Build a fake (N=1) logits dict whose argmax equals the true label.
    def onehot(arr, n_classes):
        t = torch.as_tensor(arr).long()
        return torch.nn.functional.one_hot(t, n_classes).float().unsqueeze(0)

    base = scodec.codec.base
    logits = {
        "reg_type": onehot(enc["reg_type"], 3),
        "reg_sign": onehot(enc["reg_sign"], 2),
        "reg_digits": onehot(enc["reg_digits"], base),
        "heap_sign": onehot(enc["heap_sign"], 2),
        "heap_digits": onehot(enc["heap_digits"], base),
        "pc": onehot(enc["pc"], scodec.codec.max_pc + 1),
        "halted": onehot(enc["halted"], 2),
        "error": onehot(enc["error"], 2),
    }
    recon = decode_logits_to_state(scodec, logits, 0)
    assert scodec.exact_match(scodec.encode(state), scodec.encode(recon))


# ---------------------------------------------------------------------------
# End-to-end: divergence_beam_plan runs without error, scorer stays at 0 VM
# ---------------------------------------------------------------------------

def test_divergence_beam_plan_runs_end_to_end():
    for seed in range(4):
        scorer, _, _ = _tiny_scorer()
        task = _task(seed, edit_budget=2)
        res = divergence_beam_plan(task, scorer=scorer, beam_width=4,
                                   max_depth=2, max_executions=200)
        assert isinstance(res, SearchResult)
        assert res.solved in (True, False)
        assert res.executions >= 0
        # The scorer itself ran ZERO VM executions (no hidden interpreter cost).
        assert scorer.executions == 0
        # If it claims a solution, that plan must really satisfy the goal.
        if res.solved and res.plan:
            from execwm.substrate.edits import apply_edit
            from execwm.plan.goal_tasks import satisfies
            prog = list(task.base_bytecode)
            for e in res.plan:
                prog = apply_edit(prog, e)
            assert satisfies(task.goal, run_traced(prog, task.init_state,
                                                   max_steps=task.max_steps))


def test_divergence_beam_plan_respects_execution_cap():
    scorer, _, _ = _tiny_scorer()
    task = _task(1, edit_budget=2)
    # Zero budget: not even the root base run is allowed -> honest failure.
    res = divergence_beam_plan(task, scorer=scorer, beam_width=2, max_depth=2,
                               max_executions=0)
    assert isinstance(res, SearchResult)
    assert res.solved is False
    assert res.executions == 0
    assert scorer.executions == 0
