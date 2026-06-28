"""Tests for the learned-world-model planning scorer (M3 payoff plumbing).

These are plumbing/contract tests on a tiny untrained model on CPU -- they assert
the scorer is well-formed and never touches the VM, NOT that it scores accurately
(accuracy is what ``scripts/wm_plan_eval.py`` measures on the trained checkpoint):

* the scorer returns a finite float and keeps ``executions == 0`` (no VM calls);
* ``goal_distance`` is exactly ``0`` when the goal matches the decoded final state;
* ``beam_plan`` runs end-to-end with the WM scorer without crashing (solved or not).
"""

import math
import random

import torch

from execwm.data.action_codec import ActionCodec
from execwm.data.state_codec import CodecConfig, StateCodec
from execwm.model.world_model import GroundedLatentWM, ModelConfig
from execwm.plan.goal_tasks import (Goal, GoalKind, GoalTask, make_goal_task,
                                     satisfies)
from execwm.plan.planner import beam_plan
from execwm.plan.search_baseline import SearchResult
from execwm.plan.wm_scorer import WorldModelScorer
from execwm.substrate.generators import GenSpec

_SPEC = GenSpec(num_vars=3, num_inputs=2, num_temps=8, max_depth=1, num_stmts=3,
                max_const=4, max_input_val=4, max_loop_count=2, use_heap=False)
_CODEC = CodecConfig(max_digits=4, base=10, max_pc=128)
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
    scorer = WorldModelScorer(model, scodec, acodec, _DEVICE, max_steps=16)
    return scorer


def _task(seed: int, edit_budget: int = 2) -> GoalTask:
    return make_goal_task(random.Random(seed), _SPEC, codec_cfg=_CODEC,
                          edit_budget=edit_budget)


def test_scorer_returns_finite_float_and_no_executions():
    scorer = _build()
    task = _task(1)
    # A HALTS_OK goal scores 0.0/1.0 from any decoded state, so this asserts the
    # float contract independent of the untrained model's (random) register decode.
    score = scorer(task.base_bytecode, task.init_state, Goal(GoalKind.HALTS_OK))
    assert isinstance(score, float)
    assert math.isfinite(score)
    # A register goal is still a valid float (may be inf if the target decodes UNDEF).
    reg_score = scorer(task.base_bytecode, task.init_state, task.goal)
    assert isinstance(reg_score, float)
    assert scorer.executions == 0  # never ran the VM


def test_goal_distance_zero_when_decoded_state_satisfies():
    scorer = _build()
    task = _task(2)
    # Decode the scorer's own simulated final state, then build a goal it meets.
    final = scorer.simulate(task.base_bytecode, task.init_state)
    reg = next((r for r, v in final.regs.items() if v is not None), None)
    assert reg is not None, "tiny model decoded no INT register to target"
    goal = Goal(GoalKind.REG_EQUALS, reg=reg, value=final.regs[reg])
    assert satisfies(goal, final)
    # The scorer is deterministic, so re-scoring that matching goal yields 0.0.
    assert scorer(task.base_bytecode, task.init_state, goal) == 0.0
    assert scorer.executions == 0


def test_beam_plan_runs_end_to_end_with_wm_scorer():
    scorer = _build()
    for seed in range(4):
        task = _task(seed)
        res = beam_plan(task, scorer=scorer, beam_width=2,
                        max_depth=task.edit_budget, max_executions=50)
        assert isinstance(res, SearchResult)
        assert res.solved in (True, False)
        assert res.executions >= 0
    # The scorer contributed zero VM executions to every plan.
    assert scorer.executions == 0
