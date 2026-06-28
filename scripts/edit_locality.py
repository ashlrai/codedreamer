"""Edit-locality analysis for the divergence-head planner thesis (READ-ONLY).

When a single-statement program edit changes execution, HOW FAR does the change
propagate through the trace? This quantifies the payoff of a planner that
re-simulates only from the divergence point (reusing the identical base prefix)
versus re-simulating the whole edited program from scratch.

Pure symbolic analysis over make_edit_example samples. No model, no training.

Run:  PYTHONPATH=. python3 scripts/edit_locality.py
"""

from __future__ import annotations

import random
import statistics
from collections import Counter, defaultdict
from dataclasses import replace

from execwm.data.edit_dataset import EditExample, make_edit_example
from execwm.substrate.edits import EditKind
from execwm.substrate.generators import GenSpec
from execwm.substrate.vm import MachineState, Trace


# ---------------------------------------------------------------------------
# Per-example locality metrics
# ---------------------------------------------------------------------------


def _states_equal(a: MachineState | None, b: MachineState | None) -> bool:
    """Full observational equality of two states; a missing state (one trace
    ended earlier) never matches a present one."""
    if a is None or b is None:
        return False
    return a == b  # dataclass field-wise eq (regs/types/heap/pc/flags/steps)


def analyze(ex: EditExample) -> dict:
    """Compute locality metrics for one edit example.

    State index ``t`` in a trace is the machine state *before* step ``t``;
    states[0] is the shared init_state. state[t] = step(state[t-1], action[t-1]),
    so the first differing STATE index ``d`` implies the first differing ACTION
    (the edited instruction's first execution) is at index ``d-1``.

    A divergence-head planner reuses base states[0..d-1] (identical) and only
    re-executes actions[d-1 ..], i.e. (d-1) leading steps are saved out of the
    edited trace's len(actions) total.
    """
    bs, es = ex.base_trace.states, ex.edited_trace.states
    la, le = len(ex.base_trace.actions), len(ex.edited_trace.actions)
    n_states_a, n_states_b = len(bs), len(es)
    L = max(n_states_a, n_states_b)

    # differ[t] over the full aligned index range [0, L)
    differ = [
        not _states_equal(
            bs[t] if t < n_states_a else None,
            es[t] if t < n_states_b else None,
        )
        for t in range(L)
    ]

    # Divergence onset (first differing STATE index). Guaranteed to exist since
    # make_edit_example only keeps trace-changing edits; states[0] always match.
    d = next((t for t in range(L) if differ[t]), L)
    first_diff_action = max(0, d - 1)

    # Propagation extent: of the positions from divergence to the end, what
    # fraction actually differ (re-convergence lowers this below 1.0).
    remaining = L - d
    changed_after = sum(1 for t in range(d, L) if differ[t])
    prop_fraction = changed_after / remaining if remaining > 0 else 0.0

    # Re-convergence: after first divergence, does an aligned position match
    # again? (Only possible while both traces still have that index.)
    reconverged = any(not differ[t] for t in range(d, L))

    # Trace-length change = control-flow divergence (vs values-only change).
    length_changed = la != le

    # Savings: a from-divergence planner skips the (d-1) identical leading steps
    # of the edited trace and re-simulates the rest. Fraction of the edited
    # trace's execution steps avoided:
    saved_fraction = (first_diff_action / le) if le > 0 else 0.0

    return {
        "kind": ex.edit.kind,
        "base_len": la,
        "edited_len": le,
        "divergence_state_idx": d,
        "first_diff_action": first_diff_action,
        "prop_fraction": prop_fraction,
        "reconverged": reconverged,
        "length_changed": length_changed,
        "saved_fraction": saved_fraction,
        "onset_relative": first_diff_action / le if le > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Aggregation / reporting helpers
# ---------------------------------------------------------------------------


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _dist_line(name: str, xs: list[float]) -> str:
    return (f"  {name:<26} mean={statistics.mean(xs):.3f}  "
            f"p10={_pct(xs,.10):.3f}  p50={_pct(xs,.50):.3f}  "
            f"p90={_pct(xs,.90):.3f}  min={min(xs):.3f}  max={max(xs):.3f}")


def _bucket_hist(xs: list[float], edges: list[float], labels: list[str]) -> str:
    counts = [0] * len(labels)
    for x in xs:
        for i in range(len(edges) - 1):
            if edges[i] <= x < edges[i + 1] or (i == len(edges) - 2 and x == edges[-1]):
                counts[i] += 1
                break
    n = len(xs)
    lines = []
    for lab, c in zip(labels, counts):
        bar = "#" * int(round(40 * c / n)) if n else ""
        lines.append(f"    {lab:<14} {c:>5} ({100*c/n:5.1f}%) {bar}")
    return "\n".join(lines)


def run_block(title: str, spec: GenSpec, n_samples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for _ in range(n_samples):
        ex = make_edit_example(rng, spec)
        rows.append(analyze(ex))

    print("=" * 78)
    print(f"{title}  (n={len(rows)})")
    print("=" * 78)

    onset = [r["first_diff_action"] for r in rows]
    onset_rel = [r["onset_relative"] for r in rows]
    prop = [r["prop_fraction"] for r in rows]
    saved = [r["saved_fraction"] for r in rows]
    edited_len = [r["edited_len"] for r in rows]

    print(f"\n  trace length (edited): {_dist_line('', [float(x) for x in edited_len]).strip()}")

    print("\n(1) DIVERGENCE ONSET  (first differing action index)")
    print(_dist_line("absolute step", [float(x) for x in onset]))
    print(_dist_line("relative (frac of trace)", onset_rel))
    print("  relative-onset histogram:")
    print(_bucket_hist(onset_rel, [0, .1, .25, .5, .75, 1.01],
                       ["0-10% (early)", "10-25%", "25-50%", "50-75%", "75-100% (late)"]))

    print("\n(2) PROPAGATION EXTENT  (frac of post-divergence steps that differ)")
    print(_dist_line("prop_fraction", prop))
    print("  histogram:")
    print(_bucket_hist(prop, [0, .25, .5, .75, .999, 1.01],
                       ["<25%", "25-50%", "50-75%", "75-<100%", "100% (all differ)"]))

    n = len(rows)
    n_recon = sum(r["reconverged"] for r in rows)
    n_len = sum(r["length_changed"] for r in rows)
    print("\n(3) RE-CONVERGENCE  (edited trace returns to matching base after diverging)")
    print(f"  re-converged at least once: {n_recon}/{n} = {100*n_recon/n:.1f}%")
    print(f"  stays fully diverged to end: {n-n_recon}/{n} = {100*(n-n_recon)/n:.1f}%")

    print("\n(5) TRACE-LENGTH CHANGE  (control-flow divergence vs values-only)")
    print(f"  length changed (control-flow): {n_len}/{n} = {100*n_len/n:.1f}%")
    print(f"  values-only (same length):     {n-n_len}/{n} = {100*(n-n_len)/n:.1f}%")

    print("\n(4) BY EDIT KIND")
    print(f"  {'kind':<16}{'n':>5}{'onset_rel':>11}{'prop_frac':>11}"
          f"{'reconv%':>9}{'lenchg%':>9}{'saved%':>9}")
    by_kind: dict[EditKind, list[dict]] = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    for kind in EditKind:
        g = by_kind.get(kind, [])
        if not g:
            print(f"  {kind.name:<16}{0:>5}{'--':>11}")
            continue
        m = len(g)
        print(f"  {kind.name:<16}{m:>5}"
              f"{statistics.mean(x['onset_relative'] for x in g):>11.3f}"
              f"{statistics.mean(x['prop_fraction'] for x in g):>11.3f}"
              f"{100*sum(x['reconverged'] for x in g)/m:>9.1f}"
              f"{100*sum(x['length_changed'] for x in g)/m:>9.1f}"
              f"{100*statistics.mean(x['saved_fraction'] for x in g):>9.1f}")

    print("\n(6) SAVINGS  (divergence-head planner: skip identical leading steps)")
    print(_dist_line("saved_fraction", saved))
    print(f"  >>> MEAN FRACTION OF EXECUTION STEPS SAVED = {statistics.mean(saved):.3f} "
          f"({100*statistics.mean(saved):.1f}%)")
    print(f"  >>> MEDIAN = {_pct(saved,.5):.3f} ({100*_pct(saved,.5):.1f}%)")
    print()
    return rows


def main() -> None:
    base = GenSpec(num_temps=14)
    loop = replace(base, max_loop_count=3, w_for=2.0, max_depth=3)

    rows_a = run_block("DEFAULT GenSpec (num_temps=14)", base, n_samples=400, seed=1)
    rows_b = run_block("LOOP-HEAVY GenSpec (max_loop_count=3, w_for=2.0, max_depth=3)",
                       loop, n_samples=400, seed=2)

    all_rows = rows_a + rows_b
    saved = [r["saved_fraction"] for r in all_rows]
    prop = [r["prop_fraction"] for r in all_rows]
    n = len(all_rows)
    print("=" * 78)
    print(f"COMBINED SUMMARY  (n={n})")
    print("=" * 78)
    print(f"  mean saved_fraction (KEY NUMBER) = {statistics.mean(saved):.3f} "
          f"({100*statistics.mean(saved):.1f}%)")
    print(f"  median saved_fraction            = {_pct(saved,.5):.3f}")
    print(f"  mean prop_fraction after diverge = {statistics.mean(prop):.3f}")
    print(f"  re-convergence rate              = {100*sum(r['reconverged'] for r in all_rows)/n:.1f}%")
    print(f"  trace-length-change rate         = {100*sum(r['length_changed'] for r in all_rows)/n:.1f}%")


if __name__ == "__main__":
    main()
