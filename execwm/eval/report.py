"""ExecWM-Bench report / scorecard layer.

This module is the *reporting* head of the evaluation suite. The individual
metric families live elsewhere:

* :mod:`execwm.eval.ood_eval`        — single-step / rollout exact-match in-dist
                                        vs each OOD axis
* :mod:`execwm.eval.probes`          — frozen-encoder linear probes +
                                        causal-intervention flip-rate
* :mod:`execwm.eval.counterfactual`  — do(reg)/do(action) causal accuracy

Those modules each produce dicts of numbers. :class:`BenchReport` is the single
structured container that holds *all* of them for one model, serializes to / from
JSON, renders a readable markdown scorecard, grades the run against the project's
M1/M2 targets, and — the headline — puts a latent world model side by side with a
token-space baseline.

It is deliberately **pure Python**: no torch, no model, no I/O beyond an optional
JSON file. Every field is a plain dict / list / scalar so the whole thing is
JSON round-trippable, and every metric field defaults to empty so a *partial*
report (e.g. only the core numbers computed so far) is still valid and renders.

The caller is responsible for actually computing the numbers (and for stamping a
timestamp into ``meta`` — this module never calls ``datetime`` itself, to keep it
deterministic and testable).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "BenchReport",
    "scorecard",
    "scorecard_markdown",
    "compare_reports",
]


# ---------------------------------------------------------------------------
# Targets (the project's M1/M2 acceptance bars)
# ---------------------------------------------------------------------------

# single-step exact-match is reported but informational (it is the known-hard
# arithmetic frontier — see FINDINGS_M1); the binding M1 bars are per-var,
# frozen-probe interpretability, and the M2 causal counterfactual.
TARGET_SINGLE_STEP_EM = 0.99
TARGET_PER_VAR = 0.99
TARGET_REG_COMPOSITE = 0.95
TARGET_FLIP_RATE = 0.90


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------


def _fmt(x: Any, nd: int = 4) -> str:
    """Format a metric for a table cell, tolerating None / non-numbers."""
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "✅" if x else "❌"
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)


def _get(d: Any, *path: str, default: Any = None) -> Any:
    """Safe nested lookup; returns ``default`` if any link is missing."""
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


@dataclass
class BenchReport:
    """All ExecWM-Bench metric families for a single model.

    Every metric field defaults empty so a partially-filled report is valid.

    Schema
    ------
    core: {
        "single_step_exact_match": float,
        "per_var_acc": float,
        "rollout_horizon": list[float],   # exact-match at k=1,2,3,...
        "n": int,                         # #examples graded
    }
    ood: {
        axis_name: {
            "indist": {"exact_match": float, "per_var": float},
            "ood":    {"exact_match": float, "per_var": float},
            "delta_exact_match": float,    # ood - indist
            "skipped": bool,
            "reason": str | None,
        }, ...
    }
    interpretability: {
        "probe_accuracy": {field: float},
        "reg_composite": float,            # frozen-probe register field score
        "intervention_flip_rate": float,
    }
    counterfactual: {
        "register_do": {"exact_match": float, "per_var": float},
        "action_swap": {"exact_match": float, "per_var": float},
        "identity_baseline": float,        # "predict no change" lower bound
    }
    planning: {                            # R4 calibration; symbolic scorer, NO world model
        "baseline_success_rate": float,    # brute-force VM search
        "planned_success_rate": float,     # cheap-scorer beam planner
        "baseline_mean_execs": float,      # real VM executions / task
        "planned_mean_execs": float,
        "mean_saved_frac": float,          # over tasks BOTH solved
        "n_tasks": int,
        "n_both_solved": int,
        ...
    }
    meta: free-form (params, steps, device, spec/codec summary, timestamp...)
    """

    model_name: str
    core: dict = field(default_factory=dict)
    ood: dict = field(default_factory=dict)
    interpretability: dict = field(default_factory=dict)
    counterfactual: dict = field(default_factory=dict)
    # planning: R4 calibration — cheap symbolic scorer vs brute-force VM search.
    # Model-INDEPENDENT for now (no neural world model); a learned scorer comes later.
    planning: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict:
        """Plain JSON-serializable dict (a deep copy via dataclass asdict)."""
        return asdict(self)

    def to_json(self, path: str | None = None, *, indent: int = 2) -> str:
        """Serialize to a JSON string; also write to ``path`` if given."""
        s = json.dumps(self.to_dict(), indent=indent, sort_keys=True)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(s)
        return s

    @classmethod
    def from_json(cls, s_or_path: str) -> "BenchReport":
        """Rebuild from a JSON string *or* a path to a ``.json`` file.

        The argument is treated as a file path when it does not look like JSON
        (i.e. does not start with ``{``) and that file exists.
        """
        text = s_or_path
        stripped = s_or_path.lstrip()
        if not stripped.startswith("{"):
            import os

            if os.path.exists(s_or_path):
                with open(s_or_path, "r", encoding="utf-8") as fh:
                    text = fh.read()
        data = json.loads(text)
        return cls(
            model_name=data.get("model_name", "unknown"),
            core=data.get("core", {}) or {},
            ood=data.get("ood", {}) or {},
            interpretability=data.get("interpretability", {}) or {},
            counterfactual=data.get("counterfactual", {}) or {},
            # backward-compatible: old reports have no "planning" key -> empty dict
            planning=data.get("planning", {}) or {},
            meta=data.get("meta", {}) or {},
        )

    # -- markdown rendering -------------------------------------------------

    def _rollout_line(self) -> str:
        horizon = self.core.get("rollout_horizon") or []
        if not horizon:
            return "_(not measured)_"
        return " ".join(f"k{i + 1}:{v:.2f}" for i, v in enumerate(horizon))

    def to_markdown(self) -> str:
        """A clean sectioned markdown scorecard of every metric family."""
        L: list[str] = []
        L.append(f"# ExecWM-Bench Report — `{self.model_name}`")
        L.append("")

        # Core ---------------------------------------------------------------
        L.append("## Core")
        L.append("")
        L.append("| Metric | Value |")
        L.append("| --- | --- |")
        L.append(f"| single-step exact-match | {_fmt(self.core.get('single_step_exact_match'))} |")
        L.append(f"| per-variable accuracy | {_fmt(self.core.get('per_var_acc'))} |")
        L.append(f"| n (examples) | {_fmt(self.core.get('n'), nd=0)} |")
        L.append("")
        L.append(f"rollout-horizon (exact-match @ k): {self._rollout_line()}")
        L.append("")

        # OOD ----------------------------------------------------------------
        L.append("## OOD axes")
        L.append("")
        if not self.ood:
            L.append("_(no OOD axes evaluated)_")
        else:
            L.append("| Axis | in-dist EM | OOD EM | Δ EM | in-dist per-var | OOD per-var | status |")
            L.append("| --- | --- | --- | --- | --- | --- | --- |")
            for axis, rec in self.ood.items():
                if rec.get("skipped"):
                    reason = rec.get("reason") or "skipped"
                    L.append(f"| {axis} | — | — | — | — | — | skipped: {reason} |")
                    continue
                ind_em = _get(rec, "indist", "exact_match")
                ood_em = _get(rec, "ood", "exact_match")
                ind_pv = _get(rec, "indist", "per_var")
                ood_pv = _get(rec, "ood", "per_var")
                delta = rec.get("delta_exact_match")
                L.append(
                    f"| {axis} | {_fmt(ind_em)} | {_fmt(ood_em)} | {_fmt(delta)} "
                    f"| {_fmt(ind_pv)} | {_fmt(ood_pv)} | ok |"
                )
        L.append("")

        # Interpretability ---------------------------------------------------
        L.append("## Interpretability (frozen probes)")
        L.append("")
        L.append("| Metric | Value |")
        L.append("| --- | --- |")
        L.append(f"| reg_composite (frozen-probe) | {_fmt(self.interpretability.get('reg_composite'))} |")
        L.append(f"| intervention flip-rate | {_fmt(self.interpretability.get('intervention_flip_rate'))} |")
        probe = self.interpretability.get("probe_accuracy") or {}
        for fld, acc in probe.items():
            L.append(f"| probe[{fld}] | {_fmt(acc)} |")
        L.append("")

        # Counterfactual -----------------------------------------------------
        L.append("## Counterfactual (causal)")
        L.append("")
        L.append("| Intervention | exact-match | per-var |")
        L.append("| --- | --- | --- |")
        L.append(
            f"| do(register) | {_fmt(_get(self.counterfactual, 'register_do', 'exact_match'))} "
            f"| {_fmt(_get(self.counterfactual, 'register_do', 'per_var'))} |"
        )
        L.append(
            f"| do(action swap) | {_fmt(_get(self.counterfactual, 'action_swap', 'exact_match'))} "
            f"| {_fmt(_get(self.counterfactual, 'action_swap', 'per_var'))} |"
        )
        L.append(
            f"| identity baseline | {_fmt(self.counterfactual.get('identity_baseline'))} | — |"
        )
        L.append("")

        # Planning -----------------------------------------------------------
        # R4 calibration: cheap symbolic scorer (NO world model) vs brute-force
        # VM search. This is the no-WM-baseline planning result; a learned-scorer
        # variant will be reported separately.
        L.append("## Planning (real-VM executions saved)")
        L.append("")
        L.append("_no-WM baseline: cheap symbolic scorer vs brute-force VM search._")
        L.append("")
        if not self.planning:
            L.append("_(no planning evaluated)_")
        else:
            p = self.planning
            L.append("| Method | success rate | mean VM executions |")
            L.append("| --- | --- | --- |")
            L.append(
                f"| brute-force VM search (baseline) | "
                f"{_fmt(p.get('baseline_success_rate'))} | "
                f"{_fmt(p.get('baseline_mean_execs'), nd=2)} |"
            )
            L.append(
                f"| cheap-scorer beam planner | "
                f"{_fmt(p.get('planned_success_rate'))} | "
                f"{_fmt(p.get('planned_mean_execs'), nd=2)} |"
            )
            L.append("")
            L.append(
                f"mean saved-fraction (over {_fmt(p.get('n_both_solved'), nd=0)} "
                f"tasks both solved, of {_fmt(p.get('n_tasks'), nd=0)}): "
                f"**{_fmt(p.get('mean_saved_frac'))}**"
            )
        L.append("")

        # Meta ---------------------------------------------------------------
        if self.meta:
            L.append("## Meta")
            L.append("")
            for k in sorted(self.meta):
                L.append(f"- **{k}**: {self.meta[k]}")
            L.append("")

        return "\n".join(L)


# ---------------------------------------------------------------------------
# Scorecard: grade a report against the M1/M2 targets
# ---------------------------------------------------------------------------


def scorecard(report: BenchReport) -> list[dict]:
    """Evaluate ``report`` against the project targets.

    Returns one row per criterion::

        {"criterion": str, "value": float, "target": str, "pass": bool}

    Missing metrics produce a row with ``value=None`` and ``pass=False``.
    """
    rows: list[dict] = []

    # single-step exact-match — informational (known-hard arithmetic frontier)
    ss = _get(report.core, "single_step_exact_match")
    rows.append({
        "criterion": "single-step exact-match (informational)",
        "value": ss,
        "target": f"≥{TARGET_SINGLE_STEP_EM}",
        "pass": ss is not None and ss >= TARGET_SINGLE_STEP_EM,
    })

    # per-variable accuracy
    pv = _get(report.core, "per_var_acc")
    rows.append({
        "criterion": "per-variable accuracy",
        "value": pv,
        "target": f"≥{TARGET_PER_VAR}",
        "pass": pv is not None and pv >= TARGET_PER_VAR,
    })

    # frozen-probe reg_composite
    rc = _get(report.interpretability, "reg_composite")
    rows.append({
        "criterion": "frozen-probe reg_composite",
        "value": rc,
        "target": f"≥{TARGET_REG_COMPOSITE}",
        "pass": rc is not None and rc >= TARGET_REG_COMPOSITE,
    })

    # counterfactual register_do exact-match must beat identity AND be > 0
    cf = _get(report.counterfactual, "register_do", "exact_match")
    ident = _get(report.counterfactual, "identity_baseline")
    cf_pass = (
        cf is not None
        and ident is not None
        and cf > ident
        and cf > 0.0
    )
    rows.append({
        "criterion": "counterfactual register_do exact-match",
        "value": cf,
        "target": (
            f">identity ({_fmt(ident, 4) if ident is not None else '—'}) and >0"
        ),
        "pass": cf_pass,
    })

    # interpretability intervention flip-rate
    fr = _get(report.interpretability, "intervention_flip_rate")
    rows.append({
        "criterion": "intervention flip-rate",
        "value": fr,
        "target": f"≥{TARGET_FLIP_RATE}",
        "pass": fr is not None and fr >= TARGET_FLIP_RATE,
    })

    # planning — INFORMATIONAL only (does not affect the binding M1/M2 bars).
    # Appended only when planning was evaluated, so reports without a planning
    # family keep exactly the binding criteria (backward-compatible row count).
    # This is the R4 no-WM calibration: cheap symbolic scorer vs brute-force VM
    # search — it does NOT measure the world model.
    if report.planning:
        saved = _get(report.planning, "mean_saved_frac")
        rows.append({
            "criterion": "planner saves VM executions at matched success (info)",
            "value": saved,
            "target": ">0 (informational)",
            "pass": "info",
        })

    return rows


def scorecard_markdown(report: BenchReport) -> str:
    """Render :func:`scorecard` as a markdown table with ✅/❌."""
    rows = scorecard(report)
    # "info" rows (e.g. planning) are not pass/fail and excluded from the tally.
    binding = [r for r in rows if r["pass"] != "info"]
    n_pass = sum(1 for r in binding if r["pass"])
    L: list[str] = []
    L.append(f"## Scorecard — `{report.model_name}`")
    L.append("")
    L.append(f"**{n_pass}/{len(binding)} criteria pass**")
    L.append("")
    L.append("| | Criterion | Value | Target | Pass |")
    L.append("| --- | --- | --- | --- | --- |")
    for r in rows:
        mark = "ℹ️" if r["pass"] == "info" else ("✅" if r["pass"] else "❌")
        L.append(
            f"| {mark} | {r['criterion']} | {_fmt(r['value'])} "
            f"| {r['target']} | {mark} |"
        )
    L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# The headline comparison: latent world model vs token-space baseline
# ---------------------------------------------------------------------------


def _delta(a: Any, b: Any) -> Any:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a - b
    return None


def compare_reports(latent: BenchReport, baseline: BenchReport) -> str:
    """Side-by-side markdown of a latent model vs a token-space baseline.

    Lines up the core, counterfactual, and in-distribution OOD numbers with a
    ``Δ (latent − baseline)`` column, then states the headline verdict: does the
    latent model beat token-space on single-step exact-match *and* on
    counterfactual (register_do) accuracy — the project's central thesis.
    """
    L: list[str] = []
    L.append("# Latent vs Token-space — ExecWM-Bench")
    L.append("")
    L.append(f"**Latent:** `{latent.model_name}`  **Baseline:** `{baseline.model_name}`")
    L.append("")
    L.append("| Metric | Latent | Baseline | Δ (latent − baseline) |")
    L.append("| --- | --- | --- | --- |")

    def row(label: str, a: Any, b: Any) -> None:
        L.append(f"| {label} | {_fmt(a)} | {_fmt(b)} | {_fmt(_delta(a, b))} |")

    # Core
    row(
        "single-step exact-match",
        _get(latent.core, "single_step_exact_match"),
        _get(baseline.core, "single_step_exact_match"),
    )
    row(
        "per-variable accuracy",
        _get(latent.core, "per_var_acc"),
        _get(baseline.core, "per_var_acc"),
    )

    # Counterfactual
    row(
        "counterfactual do(reg) exact-match",
        _get(latent.counterfactual, "register_do", "exact_match"),
        _get(baseline.counterfactual, "register_do", "exact_match"),
    )
    row(
        "counterfactual do(action) exact-match",
        _get(latent.counterfactual, "action_swap", "exact_match"),
        _get(baseline.counterfactual, "action_swap", "exact_match"),
    )

    # OOD (in-distribution leg of each shared axis)
    shared_axes = [a for a in latent.ood if a in baseline.ood]
    for axis in shared_axes:
        lrec, brec = latent.ood[axis], baseline.ood[axis]
        if lrec.get("skipped") or brec.get("skipped"):
            continue
        row(
            f"OOD[{axis}] in-dist exact-match",
            _get(lrec, "indist", "exact_match"),
            _get(brec, "indist", "exact_match"),
        )

    L.append("")

    # Verdict ------------------------------------------------------------
    ss_l = _get(latent.core, "single_step_exact_match")
    ss_b = _get(baseline.core, "single_step_exact_match")
    cf_l = _get(latent.counterfactual, "register_do", "exact_match")
    cf_b = _get(baseline.counterfactual, "register_do", "exact_match")

    ss_win = ss_l is not None and ss_b is not None and ss_l > ss_b
    cf_win = cf_l is not None and cf_b is not None and cf_l > cf_b

    if ss_win and cf_win:
        verdict = (
            "✅ **Latent beats token-space** on both single-step exact-match and "
            "counterfactual accuracy — the central thesis holds."
        )
    elif ss_win or cf_win:
        which = "single-step exact-match" if ss_win else "counterfactual accuracy"
        verdict = (
            f"⚠️ **Mixed:** latent wins on {which} but not the other axis — "
            "thesis only partially supported."
        )
    else:
        verdict = (
            "❌ **Latent does not beat token-space** on single-step exact-match or "
            "counterfactual accuracy at this configuration."
        )

    L.append("**Verdict:** " + verdict)
    L.append("")
    return "\n".join(L)
