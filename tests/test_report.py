"""Tests for the ExecWM-Bench report / scorecard layer (pure Python, fast)."""

from __future__ import annotations

import json

from execwm.eval.report import (
    BenchReport,
    compare_reports,
    scorecard,
    scorecard_markdown,
)


def _full_report(name: str = "latent-slotted-d256", *, strong: bool = True) -> BenchReport:
    """A fully-populated report. ``strong`` toggles passing vs failing numbers."""
    if strong:
        ss, pv, rc, fr = 0.70, 0.999, 0.97, 0.95
        cf_em = 0.62
    else:
        ss, pv, rc, fr = 0.21, 0.90, 0.90, 0.40
        cf_em = 0.05
    return BenchReport(
        model_name=name,
        core={
            "single_step_exact_match": ss,
            "per_var_acc": pv,
            "rollout_horizon": [0.62, 0.45, 0.30, 0.18, 0.09],
            "n": 512,
        },
        ood={
            "magnitude": {
                "indist": {"exact_match": 0.70, "per_var": 0.94},
                "ood": {"exact_match": 0.40, "per_var": 0.88},
                "delta_exact_match": -0.30,
                "skipped": False,
                "reason": None,
            },
            "nesting_depth": {
                "indist": {"exact_match": None, "per_var": None},
                "ood": {"exact_match": None, "per_var": None},
                "delta_exact_match": None,
                "skipped": True,
                "reason": "register shape mismatch",
            },
        },
        interpretability={
            "probe_accuracy": {"reg_digits": 0.999, "pc": 1.0, "flags": 1.0},
            "reg_composite": rc,
            "intervention_flip_rate": fr,
        },
        counterfactual={
            "register_do": {"exact_match": cf_em, "per_var": 0.93},
            "action_swap": {"exact_match": 0.30, "per_var": 0.80},
            "identity_baseline": 0.0,
        },
        meta={
            "params": 7_200_000,
            "steps": 1500,
            "device": "mps",
            "codec": "slotted-v1",
            "timestamp": "2026-06-26T12:00:00Z",
        },
    )


def _baseline_report() -> BenchReport:
    """A weaker token-space baseline."""
    return BenchReport(
        model_name="token-space-baseline",
        core={
            "single_step_exact_match": 0.55,
            "per_var_acc": 0.90,
            "rollout_horizon": [0.50, 0.30, 0.15, 0.06, 0.02],
            "n": 512,
        },
        ood={
            "magnitude": {
                "indist": {"exact_match": 0.55, "per_var": 0.90},
                "ood": {"exact_match": 0.20, "per_var": 0.80},
                "delta_exact_match": -0.35,
                "skipped": False,
                "reason": None,
            },
        },
        interpretability={
            "probe_accuracy": {"reg_digits": 0.80},
            "reg_composite": 0.80,
            "intervention_flip_rate": 0.50,
        },
        counterfactual={
            "register_do": {"exact_match": 0.40, "per_var": 0.85},
            "action_swap": {"exact_match": 0.18, "per_var": 0.70},
            "identity_baseline": 0.0,
        },
        meta={"timestamp": "2026-06-26T12:00:00Z"},
    )


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


def test_json_roundtrip_equal_dict():
    rep = _full_report()
    s = rep.to_json()
    assert isinstance(s, str) and s.strip()
    back = BenchReport.from_json(s)
    assert back.to_dict() == rep.to_dict()


def test_json_roundtrip_via_file(tmp_path):
    rep = _full_report()
    path = tmp_path / "report.json"
    s = rep.to_json(path=str(path))
    assert path.exists()
    assert json.loads(path.read_text()) == json.loads(s)
    back = BenchReport.from_json(str(path))
    assert back.to_dict() == rep.to_dict()


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def test_to_markdown_has_sections():
    md = _full_report().to_markdown()
    assert md
    for header in ("## Core", "## OOD axes", "## Interpretability", "## Counterfactual"):
        assert header in md
    # rollout horizon compact line
    assert "k1:" in md and "k2:" in md
    # skipped axis surfaced
    assert "skipped" in md


def test_scorecard_markdown_has_marks():
    md = scorecard_markdown(_full_report())
    assert md and "## Scorecard" in md
    assert ("✅" in md) or ("❌" in md)


def test_compare_reports_renders_and_verdict():
    md = compare_reports(_full_report(strong=True), _baseline_report())
    assert md
    assert "Latent vs Token-space" in md
    assert "Δ (latent − baseline)" in md
    assert "**Verdict:**" in md
    assert ("✅" in md) or ("❌" in md) or ("⚠️" in md)
    # strong latent beats the baseline on both EM and counterfactual
    assert "Latent beats token-space" in md


# ---------------------------------------------------------------------------
# Scorecard logic
# ---------------------------------------------------------------------------


def test_scorecard_rows_and_pass_fail():
    rows = scorecard(_full_report(strong=True))
    assert len(rows) == 5
    by = {r["criterion"]: r for r in rows}

    # per-var 0.999 passes >= 0.99
    pv = next(r for k, r in by.items() if "per-variable" in k)
    assert pv["value"] == 0.999 and pv["pass"] is True

    # reg_composite 0.97 passes >= 0.95
    rc = next(r for k, r in by.items() if "reg_composite" in k)
    assert rc["pass"] is True

    # flip-rate 0.95 passes >= 0.90
    fr = next(r for k, r in by.items() if "flip-rate" in k)
    assert fr["pass"] is True

    # counterfactual 0.62 > identity 0.0 and > 0 -> pass
    cf = next(r for k, r in by.items() if "counterfactual" in k)
    assert cf["pass"] is True

    # single-step 0.70 is informational, fails >= 0.99
    ss = next(r for k, r in by.items() if "single-step" in k)
    assert ss["pass"] is False


def test_scorecard_failing_values():
    rows = scorecard(_full_report(strong=False))
    by = {r["criterion"]: r for r in rows}

    # per-var 0.90 fails >= 0.99
    assert next(r for k, r in by.items() if "per-variable" in k)["pass"] is False
    # reg_composite 0.90 fails >= 0.95
    assert next(r for k, r in by.items() if "reg_composite" in k)["pass"] is False
    # flip-rate 0.40 fails >= 0.90
    assert next(r for k, r in by.items() if "flip-rate" in k)["pass"] is False


# ---------------------------------------------------------------------------
# Partial report (only core set)
# ---------------------------------------------------------------------------


def test_partial_report_serializes_and_renders():
    rep = BenchReport(
        model_name="partial",
        core={
            "single_step_exact_match": 0.5,
            "per_var_acc": 0.8,
            "rollout_horizon": [0.5, 0.25],
            "n": 16,
        },
    )
    # serialize / round-trip
    back = BenchReport.from_json(rep.to_json())
    assert back.to_dict() == rep.to_dict()

    # renders without error
    md = rep.to_markdown()
    assert "## Core" in md
    assert "no OOD axes evaluated" in md

    # scorecard still returns all 5 rows; missing metrics fail (or are None)
    rows = scorecard(rep)
    assert len(rows) == 5
    rc = next(r for r in rows if "reg_composite" in r["criterion"])
    assert rc["value"] is None and rc["pass"] is False

    # scorecard markdown still renders with marks
    assert "❌" in scorecard_markdown(rep)


# ---------------------------------------------------------------------------
# Planning family (R4 calibration; symbolic scorer, model-independent)
# ---------------------------------------------------------------------------


def _planning_dict() -> dict:
    return {
        "baseline_success_rate": 0.90,
        "planned_success_rate": 0.83,
        "baseline_mean_execs": 42.0,
        "planned_mean_execs": 9.5,
        "baseline_mean_execs_solved": 38.0,
        "planned_mean_execs_solved": 8.0,
        "mean_saved_frac": 0.74,
        "n_tasks": 30,
        "n_both_solved": 24,
        "scorer": "cheap_symbolic",
    }


def test_planning_roundtrips_and_renders():
    rep = _full_report()
    rep.planning = _planning_dict()
    # round-trip through JSON preserves the planning family
    back = BenchReport.from_json(rep.to_json())
    assert back.to_dict() == rep.to_dict()
    assert back.planning == _planning_dict()
    # markdown renders the planning section with the headline numbers
    md = rep.to_markdown()
    assert "## Planning (real-VM executions saved)" in md
    assert "brute-force VM search" in md
    assert "cheap-scorer beam planner" in md
    assert "saved-fraction" in md
    # informational scorecard row appears but does not change the binding tally
    rows = scorecard(rep)
    assert len(rows) == 6
    info = next(r for r in rows if r["pass"] == "info")
    assert info["value"] == 0.74
    sc = scorecard_markdown(rep)
    assert "ℹ️" in sc
    assert "5 criteria pass" in sc  # binding bars only; info excluded


def test_planning_empty_renders_gracefully():
    md = _full_report().to_markdown()  # no planning set
    assert "## Planning (real-VM executions saved)" in md
    assert "no planning evaluated" in md


def test_old_report_without_planning_field_loads():
    # An OLD-style report dict with NO "planning" key must still load.
    old = {
        "model_name": "legacy",
        "core": {"single_step_exact_match": 0.5, "per_var_acc": 0.8,
                 "rollout_horizon": [0.5], "n": 8},
        "ood": {},
        "interpretability": {},
        "counterfactual": {},
        "meta": {},
    }
    rep = BenchReport.from_json(json.dumps(old))
    assert rep.planning == {}
    # scorecard keeps exactly the 5 binding criteria when planning is empty
    assert len(scorecard(rep)) == 5
    # markdown still renders, with the empty-planning notice
    assert "no planning evaluated" in rep.to_markdown()
