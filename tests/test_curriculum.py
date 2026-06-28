"""Tests for the magnitude curriculum (mechanism + a tiny CPU plumbing smoke).

These are fast: the schedule-math tests are pure, and the end-to-end smoke trains
~10 steps on a tiny model/spec on CPU. It is a plumbing test (right dict shape,
metrics in [0,1]), NOT an accuracy test.
"""

import dataclasses

import torch

from execwm.data.state_codec import CodecConfig
from execwm.substrate.generators import GenSpec
from execwm.train.curriculum import (CurriculumStage, MagnitudeCurriculum,
                                      linear_magnitude_curriculum,
                                      spec_for_step, stage_index_for_step)
from execwm.train.train_arith_curriculum import train_arith_curriculum
from execwm.train.train_m1 import TrainConfig


# --- schedule math --------------------------------------------------------


def test_linear_curriculum_stage_math():
    target = GenSpec(max_const=100, max_input_val=50)
    curr = linear_magnitude_curriculum(target, n_stages=4, start_max=1)
    assert curr.n_stages == 4
    # equal fractions summing to ~1.0
    assert abs(curr.total_fraction - 1.0) < 1e-9
    assert all(abs(s.fraction - 0.25) < 1e-9 for s in curr.stages)
    # starts at start_max, ends exactly at the target magnitude
    assert curr.stages[0].max_const == 1 and curr.stages[0].max_input_val == 1
    assert curr.stages[-1].max_const == 100 and curr.stages[-1].max_input_val == 50
    # monotonic non-decreasing magnitude
    cs = [s.max_const for s in curr.stages]
    vs = [s.max_input_val for s in curr.stages]
    assert cs == sorted(cs) and vs == sorted(vs)


def test_linear_curriculum_single_stage_is_target():
    target = GenSpec(max_const=7, max_input_val=9)
    curr = linear_magnitude_curriculum(target, n_stages=1, start_max=1)
    assert curr.n_stages == 1
    assert curr.stages[0].max_const == 7 and curr.stages[0].max_input_val == 9


def test_spec_for_step_increases_and_does_not_mutate():
    base = GenSpec(num_vars=4, max_const=64, max_input_val=64)
    base_snapshot = dataclasses.asdict(base)
    curr = linear_magnitude_curriculum(base, n_stages=4, start_max=1)
    total = 100

    mags = [spec_for_step(curr, base, s, total).max_const
            for s in range(0, total, 5)]
    # non-decreasing across the run, and strictly grows overall
    assert mags == sorted(mags)
    assert mags[0] < mags[-1]

    # base_spec is never mutated; replace() yields a fresh object
    assert dataclasses.asdict(base) == base_snapshot
    s_early = spec_for_step(curr, base, 0, total)
    assert s_early is not base

    # final stage == full target magnitude; other GenSpec fields untouched
    s_last = spec_for_step(curr, base, total - 1, total)
    assert s_last.max_const == base.max_const
    assert s_last.max_input_val == base.max_input_val
    assert s_last.num_vars == base.num_vars
    # steps at/beyond total map to the hardest stage
    assert stage_index_for_step(curr, total, total) == curr.n_stages - 1
    assert stage_index_for_step(curr, total + 50, total) == curr.n_stages - 1


def test_stage_index_partitions_step_budget():
    curr = MagnitudeCurriculum(stages=(
        CurriculumStage(0.5, 1, 1),
        CurriculumStage(0.5, 10, 10),
    ))
    total = 10
    idx = [stage_index_for_step(curr, s, total) for s in range(total)]
    assert idx[:5] == [0] * 5
    assert idx[5:] == [1] * 5


# --- tiny end-to-end CPU smoke (plumbing, not accuracy) -------------------


def test_train_arith_curriculum_smoke_cpu():
    torch.manual_seed(0)
    base = GenSpec(num_vars=3, num_inputs=2, num_temps=14, max_depth=1,
                   max_expr_depth=2, num_stmts=3, max_const=8, max_input_val=8,
                   max_loop_count=2, num_lists=1, list_len=2)
    codec = CodecConfig(max_digits=4, base=10, max_pc=64)
    tc = TrainConfig(steps=10, batch_size=4, max_len=8, rollout_warmup=5,
                     rollout_grow_every=5, rollout_max_k=2)
    curr = linear_magnitude_curriculum(base, n_stages=2, start_max=1)

    out = train_arith_curriculum(
        base, codec, tc, curr, n_train=24, n_eval=12, seed=0,
        device=torch.device("cpu"), log_every=5,
        d_model=32, n_heads=2, enc_layers=1, dyn_layers=1)

    assert set(out) >= {"model", "eval", "rollout_horizon", "scodec",
                        "acodec", "curriculum"}
    ev = out["eval"]
    assert 0.0 <= ev["step_exact_match"] <= 1.0
    assert 0.0 <= ev["per_var_acc"] <= 1.0
    assert ev["n"] > 0
    assert len(out["rollout_horizon"]) == tc.max_len
    assert out["curriculum"] is curr
