"""Magnitude curriculum for the carry-aware arithmetic head.

M1.5/M1.6 established that the binding constraint on rollout horizon is the
*single-step arithmetic error*: rollout@k decays geometrically in the single-step
exact-match. Multi-digit arithmetic is the hard part because carries propagate
low->high, and the literature (Abacus 2405.17399, Learning-to-Execute 1410.4615)
recommends *ramping the value magnitude over training* so carries are learned
progressively (small magnitudes first, then larger).

The codec ``max_digits`` is FIXED (it defines the tensor shape and never
truncates). What this module ramps is the DATA's value magnitude — the GenSpec
fields ``max_const`` (literal constants) and ``max_input_val`` (input register /
heap magnitudes) — which together control the effective operand size and hence
how many carries a single arithmetic step must get right.

A :class:`MagnitudeCurriculum` is a list of stages, each a fraction of total
training steps plus the (max_const, max_input_val) magnitude active in that
stage. :func:`spec_for_step` maps a global step to the GenSpec for its stage via
``dataclasses.replace`` (never mutating the base spec). :func:`stage_examples`
resamples the training pool at a stage's magnitude — call it only on stage
boundaries, not every step.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..data.action_codec import ActionCodec
from ..data.dataset import collect_examples
from ..data.state_codec import StateCodec
from ..substrate.generators import Example, GenSpec


@dataclass(frozen=True)
class CurriculumStage:
    """One curriculum stage: ``fraction`` of total steps at this magnitude."""

    fraction: float       # share of total training steps (stages sum to ~1.0)
    max_const: int        # GenSpec.max_const for this stage
    max_input_val: int    # GenSpec.max_input_val for this stage


@dataclass(frozen=True)
class MagnitudeCurriculum:
    """An ordered schedule of magnitude stages, small -> large.

    ``stages`` are consumed in order; ``fraction`` fields are interpreted as
    consecutive shares of the total step budget and are normalized at lookup
    time, so they need only be *proportional* (they need not sum to exactly 1).
    """

    stages: tuple[CurriculumStage, ...]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("MagnitudeCurriculum needs at least one stage")
        if any(s.fraction <= 0 for s in self.stages):
            raise ValueError("stage fractions must be positive")

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    @property
    def total_fraction(self) -> float:
        return sum(s.fraction for s in self.stages)


def linear_magnitude_curriculum(
    target_spec: GenSpec, n_stages: int, start_max: int = 1
) -> MagnitudeCurriculum:
    """Build an ``n_stages`` curriculum that linearly ramps magnitude from
    ``start_max`` up to ``target_spec``'s (max_const, max_input_val).

    Each stage gets an equal ``1 / n_stages`` share of the step budget. Stage 0
    sits at ``start_max`` for both fields; the final stage equals the target
    magnitude exactly. Values are rounded to ints and clamped to ``>= 1``.
    """
    if n_stages < 1:
        raise ValueError("n_stages must be >= 1")
    tgt_c, tgt_v = target_spec.max_const, target_spec.max_input_val
    frac = 1.0 / n_stages

    def lerp(lo: int, hi: int, t: float) -> int:
        return max(1, round(lo + t * (hi - lo)))

    stages: list[CurriculumStage] = []
    for i in range(n_stages):
        t = i / (n_stages - 1) if n_stages > 1 else 1.0
        stages.append(CurriculumStage(
            fraction=frac,
            max_const=lerp(start_max, tgt_c, t),
            max_input_val=lerp(start_max, tgt_v, t),
        ))
    # pin the final stage to the exact target magnitude (guard against rounding)
    stages[-1] = CurriculumStage(fraction=frac, max_const=tgt_c, max_input_val=tgt_v)
    return MagnitudeCurriculum(stages=tuple(stages))


def stage_index_for_step(curriculum: MagnitudeCurriculum, step: int, total_steps: int) -> int:
    """Index of the active stage for a 0-based global ``step``.

    Steps at or beyond ``total_steps`` map to the final (hardest) stage.
    """
    if total_steps <= 0:
        return curriculum.n_stages - 1
    frac = step / total_steps
    if frac >= 1.0:
        return curriculum.n_stages - 1
    total = curriculum.total_fraction
    cum = 0.0
    for idx, st in enumerate(curriculum.stages):
        cum += st.fraction / total
        if frac < cum:
            return idx
    return curriculum.n_stages - 1


def spec_for_step(
    curriculum: MagnitudeCurriculum, base_spec: GenSpec, step: int, total_steps: int
) -> GenSpec:
    """GenSpec for ``step``'s stage: ``base_spec`` with the stage's magnitude
    fields. Never mutates ``base_spec`` (uses ``dataclasses.replace``)."""
    st = curriculum.stages[stage_index_for_step(curriculum, step, total_steps)]
    return replace(base_spec, max_const=st.max_const, max_input_val=st.max_input_val)


def stage_examples(
    spec: GenSpec, n: int, seed: int, scodec: StateCodec, acodec: ActionCodec
) -> list[Example]:
    """Sample ``n`` terminating, encodable training examples at ``spec``'s
    magnitude. Thin wrapper over :func:`collect_examples`; call once per stage
    boundary, not per step."""
    examples, _ = collect_examples(spec, n, lambda e: True, seed, scodec, acodec)
    return examples
