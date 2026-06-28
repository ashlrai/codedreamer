"""Edit-example generator: base program + edit + the two traces it induces.

This is the data substrate for M3 ("edit as action"). Each example pairs a base
program's trace with the trace of an *edited* copy run from the SAME inputs, so
the learning signal is exactly how an edit perturbs execution. Only *informative*
samples are kept: the edit must actually change the trace and both traces must be
encodable (so downstream state/edit codecs never choke). Uninformative or
non-encodable / trapping samples are dropped and resampled.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..substrate.edits import Edit, EditConfig, apply_edit, sample_edit
from ..substrate.generators import GenSpec, make_example
from ..substrate.vm import Instr, MachineState, Trace, VMError, run_traced
from .edit_codec import EditCodec
from .state_codec import CodecConfig, EncodeError, StateCodec


@dataclass
class EditExample:
    """A base program, an edit, and the two traces run from identical inputs.

    ``base_trace`` and ``edited_trace`` start from the same ``init_state`` and are
    guaranteed (by :func:`make_edit_example`) to differ — that divergence is the
    point of the example.
    """

    base_bytecode: list[Instr]
    edited_bytecode: list[Instr]
    edit: Edit
    init_state: MachineState
    base_trace: Trace
    edited_trace: Trace


def traces_equivalent(a: Trace, b: Trace) -> bool:
    """True iff two traces are observationally identical (same actions and same
    full state sequence). Used to reject edits that do not change execution."""
    if len(a.states) != len(b.states):
        return False
    if a.actions != b.actions:
        return False
    return a.states == b.states


def _all_encodable(trace: Trace, scodec: StateCodec) -> bool:
    try:
        for st in trace.states:
            scodec.encode(st)
    except EncodeError:
        return False
    return True


def make_edit_example(rng: random.Random, spec: GenSpec,
                      codec_cfg: CodecConfig | None = None,
                      edit_config: EditConfig | None = None,
                      max_attempts: int = 400) -> EditExample:
    """Generate one informative edit example.

    Repeatedly: synthesize a base program + inputs, sample a structurally-valid
    edit, apply it, and re-run from the same ``init_state``. Keep the first sample
    whose edited trace differs from the base trace and where the edit and both
    traces are encodable; drop samples that trap on an undefined read, leave the
    trace unchanged, or fall outside codec range. Raises ``RuntimeError`` if no
    such example is found within ``max_attempts`` (should not happen for sane
    specs).
    """
    codec_cfg = codec_cfg or CodecConfig()
    config = spec.config()
    scodec = StateCodec(config, codec_cfg)
    ecodec = EditCodec(config, edit_config, codec_cfg)

    for _ in range(max_attempts):
        ex = make_example(rng, spec)
        edit = sample_edit(ex.bytecode, config, rng, edit_config)
        if edit is None:
            continue
        try:
            edited_bytecode = apply_edit(ex.bytecode, edit)
        except ValueError:
            continue
        try:
            edited_trace = run_traced(edited_bytecode, ex.init_state,
                                      max_steps=spec.max_steps)
        except VMError:
            # edited program read an undefined register; not runnable on these
            # inputs -> drop and resample.
            continue
        if traces_equivalent(ex.trace, edited_trace):
            continue  # edit had no observable effect
        try:
            ecodec.encode(edit)
        except EncodeError:
            continue
        if not (_all_encodable(ex.trace, scodec)
                and _all_encodable(edited_trace, scodec)):
            continue
        return EditExample(
            base_bytecode=ex.bytecode, edited_bytecode=edited_bytecode,
            edit=edit, init_state=ex.init_state,
            base_trace=ex.trace, edited_trace=edited_trace,
        )

    raise RuntimeError(
        f"could not generate a divergent edit example in {max_attempts} attempts")
