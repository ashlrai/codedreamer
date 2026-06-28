"""Dataset builder: turn the generators into provably-disjoint OOD train/test
splits of encoded ``(s_t, a_t) -> s_{t+1}`` transitions.

For threshold axes (trace length, magnitude, nesting, size) disjointness is
enforced numerically: every kept train example sits at or below the axis'
``train_max`` and every kept test example at or above its ``test_min``, and the
builder asserts ``max(train) < min(test)``. For the compositional axis it is
*structural*: training programs are generated with the held-out operator/context
pairings forbidden, and only test programs that actually contain such a pairing
are kept — so the two sets are disjoint by construction in program structure.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from ..substrate.dsl import (Assign, BinOp, For, If, ListLoad, ListStore,
                             Program)
from ..substrate.generators import (Example, GenSpec, OODAxis, make_example,
                                     realized_metrics)
from ..substrate.vm import Op
from .action_codec import ActionCodec
from .state_codec import CodecConfig, EncodeError, StateCodec


# ---------------------------------------------------------------------------
# Structural check for the compositional axis
# ---------------------------------------------------------------------------

def _expr_uses(expr, context: str, pairs: frozenset[tuple[str, Op]]) -> bool:
    if isinstance(expr, BinOp):
        if (context, expr.op) in pairs:
            return True
        return (_expr_uses(expr.left, context, pairs)
                or _expr_uses(expr.right, context, pairs))
    return False


def program_uses_pairs(program: Program, pairs: frozenset[tuple[str, Op]]) -> bool:
    """True iff ``program`` uses any held-out (context, op) pairing. Mirrors how
    the generator assigns block contexts, so it is the exact structural inverse
    of the ``forbidden_pairs`` gate."""
    if not pairs:
        return False

    def block(body, context: str) -> bool:
        for s in body:
            if isinstance(s, Assign):
                if _expr_uses(s.expr, context, pairs):
                    return True
            elif isinstance(s, ListStore):
                if (_expr_uses(s.index, context, pairs)
                        or _expr_uses(s.value, context, pairs)):
                    return True
            elif isinstance(s, ListLoad):
                if _expr_uses(s.index, context, pairs):
                    return True
            elif isinstance(s, If):
                if (_expr_uses(s.cond, "if", pairs)
                        or block(s.then, "if") or block(s.orelse, "if")):
                    return True
            elif isinstance(s, For):
                if block(s.body, "loop"):
                    return True
        return False

    return block(program.body, "top")


# ---------------------------------------------------------------------------
# Example collection with filtering
# ---------------------------------------------------------------------------


def _encodable(ex: Example, scodec: StateCodec, acodec: ActionCodec) -> bool:
    """Cheap check that every state/action will encode in range — uses realized
    metrics instead of actually encoding (the full encode happens once later in
    the dataset). A state is in range iff its largest |value| fits the digit width
    and its pc fits ``max_pc``; action immediates are bounded by program constants,
    which are far smaller, so they need no separate check."""
    codec = scodec.codec
    m = realized_metrics(ex)
    if m["max_magnitude"] >= codec.max_magnitude:
        return False
    if max(st.pc for st in ex.trace.states) > codec.max_pc:
        return False
    return True


def collect_examples(spec: GenSpec, n: int, predicate, seed: int,
                     scodec: StateCodec, acodec: ActionCodec,
                     max_attempts: int | None = None) -> tuple[list[Example], int]:
    """Generate until ``n`` examples pass ``predicate`` (and are terminating and
    encodable). Returns (examples, attempts)."""
    rng = random.Random(seed)
    max_attempts = max_attempts or n * 200
    out: list[Example] = []
    attempts = 0
    while len(out) < n and attempts < max_attempts:
        attempts += 1
        ex = make_example(rng, spec)
        if not ex.trace.terminated or len(ex.trace) == 0:
            continue
        if not predicate(ex):
            continue
        if not _encodable(ex, scodec, acodec):
            continue
        out.append(ex)
    if len(out) < n:
        raise RuntimeError(
            f"only collected {len(out)}/{n} examples in {attempts} attempts")
    return out, attempts


# ---------------------------------------------------------------------------
# Flattening traces -> stacked transition arrays
# ---------------------------------------------------------------------------


def flatten_transitions(examples: list[Example], scodec: StateCodec,
                        acodec: ActionCodec) -> dict[str, np.ndarray]:
    """Flatten every (s_t, a_t, s_{t+1}) step of every example into stacked
    arrays. Keys are prefixed ``s_`` (current state), ``a_`` (action), ``ns_``
    (next state); an extra ``ex_id`` array tags which example each row came from
    (needed later for multi-step rollouts)."""
    s_rows: list[dict] = []
    a_rows: list[dict] = []
    ns_rows: list[dict] = []
    ex_ids: list[int] = []
    for ex_id, ex in enumerate(examples):
        states, actions = ex.trace.states, ex.trace.actions
        for t in range(len(actions)):
            s_rows.append(scodec.encode(states[t]).as_dict())
            a_rows.append(acodec.encode(actions[t]).as_dict())
            ns_rows.append(scodec.encode(states[t + 1]).as_dict())
            ex_ids.append(ex_id)

    def stack(rows: list[dict], prefix: str) -> dict[str, np.ndarray]:
        if not rows:
            return {}
        return {f"{prefix}{k}": np.stack([r[k] for r in rows]) for k in rows[0]}

    out: dict[str, np.ndarray] = {}
    out.update(stack(s_rows, "s_"))
    out.update(stack(a_rows, "a_"))
    out.update(stack(ns_rows, "ns_"))
    out["ex_id"] = np.array(ex_ids, dtype=np.int64)
    return out


# ---------------------------------------------------------------------------
# Split building
# ---------------------------------------------------------------------------


@dataclass
class Split:
    axis: str
    train: dict[str, np.ndarray]
    test: dict[str, np.ndarray]
    stats: dict[str, object]


def build_split(axis: OODAxis, n_train: int, n_test: int, *,
                codec_cfg: CodecConfig, seed: int = 0) -> Split:
    """Build a disjoint train/test transition split for one OOD axis."""
    scfg = axis.train_spec.config()  # train/test specs share register shape
    scodec = StateCodec(scfg, codec_cfg)
    acodec = ActionCodec(scfg, codec_cfg)

    if axis.name == "compositional":
        pairs = axis.train_spec.forbidden_pairs
        train_pred = lambda ex: not program_uses_pairs(ex.program, pairs)
        test_pred = lambda ex: program_uses_pairs(ex.program, pairs)
    else:
        m = axis.metric
        train_pred = lambda ex: realized_metrics(ex)[m] <= axis.train_max
        test_pred = lambda ex: realized_metrics(ex)[m] >= axis.test_min

    train_ex, train_att = collect_examples(
        axis.train_spec, n_train, train_pred, seed, scodec, acodec)
    test_ex, test_att = collect_examples(
        axis.test_spec, n_test, test_pred, seed + 1, scodec, acodec)

    # --- disjointness assertions ---
    if axis.name == "compositional":
        pairs = axis.train_spec.forbidden_pairs
        assert all(not program_uses_pairs(e.program, pairs) for e in train_ex)
        assert all(program_uses_pairs(e.program, pairs) for e in test_ex)
        train_metric_range = test_metric_range = None
    else:
        m = axis.metric
        tr = [realized_metrics(e)[m] for e in train_ex]
        te = [realized_metrics(e)[m] for e in test_ex]
        assert max(tr) < min(te), (
            f"{axis.name}: splits overlap (train max {max(tr)} >= test min {min(te)})")
        train_metric_range = (min(tr), max(tr))
        test_metric_range = (min(te), max(te))

    train = flatten_transitions(train_ex, scodec, acodec)
    test = flatten_transitions(test_ex, scodec, acodec)
    stats = {
        "axis": axis.name,
        "n_train_examples": len(train_ex),
        "n_test_examples": len(test_ex),
        "n_train_transitions": int(len(train.get("ex_id", []))),
        "n_test_transitions": int(len(test.get("ex_id", []))),
        "train_attempts": train_att,
        "test_attempts": test_att,
        "train_metric_range": train_metric_range,
        "test_metric_range": test_metric_range,
    }
    return Split(axis=axis.name, train=train, test=test, stats=stats)
