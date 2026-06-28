"""Out-of-distribution generalization evaluation for the grounded latent world
model.

This module probes the project's core thesis: does the learned latent *execute*
the substrate's semantics and therefore generalize across the five OOD axes
(trace length, numeric magnitude, nesting depth, compositional novelty, program
size), or does it merely case-match the training distribution? For each axis we
build an in-distribution split (from the axis' ``train_spec``, below
``train_max``) and an OOD test split (from the axis' ``test_spec``, at or above
``test_min`` — or, for the compositional axis, programs that actually use the
held-out operator/context pairings) and score the *same* trained model on both:

  * single-step exact-match            (does s_{t+1} decode exactly?)
  * per-variable accuracy              (partial credit across state fields)
  * rollout-horizon curve              (exact-match after k pure-latent steps)

The heavy lifting (the eval loops) is reused verbatim from
:mod:`execwm.train.train_m1` — this module only builds the right example splits
and wires them through :func:`evaluate` and :func:`rollout_horizon`.

Shape caveat
------------
A model is tied to one ``(num_regs, num_cells)`` register shape via its
``ModelConfig``. Several OOD axes (nesting, program size) widen ``num_vars`` and
therefore change the register shape, so a model trained at one shape *cannot* be
evaluated on a differently-shaped axis. :func:`compare_indist_vs_ood` detects a
shape mismatch and returns a skip record rather than crashing. The magnitude and
trace-length axes keep the base register shape, so they are always evaluable.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from ..data.dataset import collect_examples, program_uses_pairs
from ..data.state_codec import CodecConfig, EncodeError, StateCodec
from ..data.torch_data import EpisodeDataset, collate_episodes
from ..substrate.generators import (Example, GenSpec, OODAxis, default_axes,
                                     realized_metrics)
from ..train.train_m1 import (build, evaluate, pick_device, rollout_horizon)

__all__ = [
    "evaluate_split",
    "gather_split_examples",
    "gather_ood_examples",
    "gather_indist_examples",
    "compare_indist_vs_ood",
    "evaluate_all_axes",
    "spec_reg_shape",
    "model_reg_shape",
]


# ---------------------------------------------------------------------------
# Register-shape helpers
# ---------------------------------------------------------------------------


def spec_reg_shape(spec: GenSpec) -> tuple[int, int]:
    """``(num_regs, num_cells)`` implied by a GenSpec's VM config."""
    cfg = spec.config()
    num_regs = len(cfg.reg_names)
    num_cells = cfg.num_lists * cfg.list_len
    return num_regs, num_cells


def model_reg_shape(scodec: StateCodec) -> tuple[int, int]:
    """``(num_regs, num_cells)`` the model was built for, read off its codec."""
    return scodec.num_regs, scodec.num_cells


# ---------------------------------------------------------------------------
# Split predicates + example gathering
# ---------------------------------------------------------------------------


def _split_predicate(axis: OODAxis, scodec: StateCodec, split: str):
    """Build a ``predicate(ex) -> bool`` selecting the requested OOD split.

    ``split`` is ``"test"`` (OOD side) or ``"train"`` (in-distribution side).
    The predicate enforces the axis condition (metric threshold, or the
    structural held-out-pairing check for the compositional axis) and guards
    that every state in the trace encodes with ``scodec`` (a tighter check than
    ``collect_examples``' cheap realized-metric gate, and the explicit contract
    that the chosen codec can represent the example)."""
    if axis.name == "compositional":
        pairs = axis.train_spec.forbidden_pairs
        if split == "test":
            cond = lambda ex: program_uses_pairs(ex.program, pairs)
        else:
            cond = lambda ex: not program_uses_pairs(ex.program, pairs)
    else:
        metric = axis.metric
        if split == "test":
            cond = lambda ex: realized_metrics(ex)[metric] >= axis.test_min
        else:
            cond = lambda ex: realized_metrics(ex)[metric] <= axis.train_max

    def predicate(ex: Example) -> bool:
        if not cond(ex):
            return False
        try:
            for state in ex.trace.states:
                scodec.encode(state)
        except EncodeError:
            return False
        return True

    return predicate


def gather_split_examples(axis: OODAxis, scodec: StateCodec, acodec,
                          n: int, seed: int, split: str) -> list[Example]:
    """Collect ``n`` terminating, encodable examples for one side of an axis.

    ``split`` selects ``"train"`` (in-distribution, from ``axis.train_spec``) or
    ``"test"`` (OOD, from ``axis.test_spec``). Disjointness from the other side
    is guaranteed by the predicate (metric thresholds with ``test_min >
    train_max``, or the structural held-out-pairing check)."""
    spec = axis.train_spec if split == "train" else axis.test_spec
    predicate = _split_predicate(axis, scodec, split)
    examples, _attempts = collect_examples(spec, n, predicate, seed, scodec, acodec)
    return examples


def gather_ood_examples(axis: OODAxis, scodec: StateCodec, acodec,
                        n: int = 300, seed: int = 0) -> list[Example]:
    """Collect ``n`` OOD test examples for ``axis`` (the ``test_spec`` side)."""
    return gather_split_examples(axis, scodec, acodec, n, seed, "test")


def gather_indist_examples(axis: OODAxis, scodec: StateCodec, acodec,
                           n: int = 300, seed: int = 0) -> list[Example]:
    """Collect ``n`` in-distribution examples for ``axis`` (the ``train_spec``
    side, below ``train_max``)."""
    return gather_split_examples(axis, scodec, acodec, n, seed, "train")


# ---------------------------------------------------------------------------
# Scoring one set of examples
# ---------------------------------------------------------------------------


def evaluate_split(model, scodec: StateCodec, acodec, examples: list[Example],
                   device=None, max_len: int = 24, rollout_k: int = 24,
                   batch_size: int = 32) -> dict:
    """Score ``model`` on a list of examples.

    Builds an :class:`EpisodeDataset` + ``DataLoader`` (the same plumbing the
    trainer uses) and delegates to the existing :func:`evaluate` (single-step
    exact-match + per-variable accuracy) and :func:`rollout_horizon` (the
    k-step compounding-error curve). Returns
    ``{step_exact_match, per_var_acc, rollout_horizon, n, n_episodes}``."""
    device = device or pick_device()
    ds = EpisodeDataset(examples, scodec, acodec, max_len=max_len)
    if len(ds) == 0:
        return {"step_exact_match": float("nan"), "per_var_acc": float("nan"),
                "rollout_horizon": [float("nan")] * rollout_k, "n": 0,
                "n_episodes": 0}
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_episodes)
    model.to(device)
    ev = evaluate(model, loader, device)
    horizon = rollout_horizon(model, loader, device, max_k=rollout_k)
    return {
        "step_exact_match": ev["step_exact_match"],
        "per_var_acc": ev["per_var_acc"],
        "rollout_horizon": horizon,
        "n": ev["n"],
        "n_episodes": len(ds),
    }


# ---------------------------------------------------------------------------
# In-distribution vs OOD comparison on a single model
# ---------------------------------------------------------------------------


def compare_indist_vs_ood(model, scodec: StateCodec, acodec, axis: OODAxis,
                          n: int = 300, device=None, max_len: int = 24,
                          seed: int = 0) -> dict:
    """Compare the SAME model in-distribution vs OOD along one axis.

    Only valid when the axis' register shape matches the model's (the model is
    tied to one ``(num_regs, num_cells)``). If it differs, returns a skip
    record. Otherwise gathers in-dist (``train_spec``, below ``train_max``) and
    OOD (``test_spec``, at/above ``test_min``) examples — encoded with the
    model's own codec so the inputs match what the model was trained on — and
    scores both. ``delta_step_exact_match`` is the in-dist minus OOD drop: large
    positive means the latent case-matched rather than generalized."""
    device = device or pick_device()
    m_shape = model_reg_shape(scodec)
    a_shape = spec_reg_shape(axis.test_spec)

    if m_shape != a_shape:
        return {
            "axis": axis.name,
            "skipped": True,
            "reason": (f"register shape mismatch: model {m_shape} != axis "
                       f"{a_shape}; train a model on this axis' spec to evaluate it"),
            "model_shape": m_shape,
            "axis_shape": a_shape,
        }

    indist_ex = gather_indist_examples(axis, scodec, acodec, n, seed)
    ood_ex = gather_ood_examples(axis, scodec, acodec, n, seed + 1)

    indist = evaluate_split(model, scodec, acodec, indist_ex, device,
                            max_len=max_len, rollout_k=max_len)
    ood = evaluate_split(model, scodec, acodec, ood_ex, device,
                         max_len=max_len, rollout_k=max_len)

    return {
        "axis": axis.name,
        "skipped": False,
        "metric": axis.metric,
        "model_shape": m_shape,
        "n_indist": indist["n_episodes"],
        "n_ood": ood["n_episodes"],
        "indist": indist,
        "ood": ood,
        "delta_step_exact_match": (indist["step_exact_match"]
                                   - ood["step_exact_match"]),
        "delta_per_var_acc": indist["per_var_acc"] - ood["per_var_acc"],
    }


def evaluate_all_axes(model, scodec: StateCodec, acodec,
                      base: GenSpec | None = None, n: int = 300,
                      device=None, max_len: int = 24, seed: int = 0) -> dict:
    """Run :func:`compare_indist_vs_ood` for every canonical axis on one model.

    Axes whose register shape differs from the model's are skipped with a note
    (a single model cannot span multiple register shapes); the returned dict
    maps axis name -> report (evaluated or skipped)."""
    reports: dict[str, dict] = {}
    for axis in default_axes(base):
        reports[axis.name] = compare_indist_vs_ood(
            model, scodec, acodec, axis, n=n, device=device,
            max_len=max_len, seed=seed)
    return reports


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------


def _fmt_horizon(horizon: list[float], k: int = 8) -> str:
    return "  ".join(f"k{i + 1}:{v:.2f}" for i, v in enumerate(horizon[:k]))


def format_report(report: dict) -> str:
    if report.get("skipped"):
        return f"[{report['axis']}] SKIPPED - {report['reason']}"
    lines = [
        f"[{report['axis']}] axis-metric={report['metric']}  "
        f"reg-shape={report['model_shape']}",
        f"  in-dist (n={report['n_indist']:3d}): "
        f"step-EM {report['indist']['step_exact_match']:.4f}  "
        f"per-var {report['indist']['per_var_acc']:.4f}",
        f"      OOD (n={report['n_ood']:3d}): "
        f"step-EM {report['ood']['step_exact_match']:.4f}  "
        f"per-var {report['ood']['per_var_acc']:.4f}",
        f"  drop: step-EM {report['delta_step_exact_match']:+.4f}  "
        f"per-var {report['delta_per_var_acc']:+.4f}",
        f"  rollout in-dist: {_fmt_horizon(report['indist']['rollout_horizon'])}",
        f"  rollout    OOD: {_fmt_horizon(report['ood']['rollout_horizon'])}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Demo: train a tiny model and compare in-dist vs OOD on the magnitude axis
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from ..train.train_m1 import TrainConfig, train

    # Base spec the model trains on. The magnitude axis only widens the numeric
    # range (max_const / max_input_val), keeping this register shape, so it is
    # the natural single-model OOD demo.
    base = GenSpec(num_vars=4, max_depth=2, num_stmts=5,
                   max_const=5, max_input_val=5, max_loop_count=3)
    codec_cfg = CodecConfig(max_digits=9, base=10, max_pc=512)
    tc = TrainConfig(steps=300, batch_size=32)

    out = train(spec=base, codec_cfg=codec_cfg, tc=tc, n_train=600, n_eval=200,
                d_model=64, n_heads=4, enc_layers=2, dyn_layers=2)
    model = out["model"]
    scodec, acodec, device = out["scodec"], out["acodec"], out["device"]

    axes = {a.name: a for a in default_axes(base)}
    print("\n=== OOD comparison: magnitude axis (in-dist vs OOD) ===")
    report = compare_indist_vs_ood(model, scodec, acodec, axes["magnitude"],
                                   n=150, device=device, max_len=tc.max_len)
    print(format_report(report))
