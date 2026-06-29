"""THE FRONTIER SCORECARD: one canonical measurement of the open challenge.

This is the single command a contributor runs to see whether they have moved the
frontier. It loads ANY checkpoint (the way ``scripts/neurosym_exec_eval.py`` does),
builds an in-distribution eval set (the checkpoint's own small-magnitude training
spec) and a magnitude-OOD eval set (``replace(spec, max_const=400, max_input_val=400)``,
exactly as the diagnostics do), then prints ONE scorecard table with the frontier
metrics in-dist vs OOD and, at the bottom, the THREE "numbers to beat" read LIVE from
this run.

What the columns mean (every number is graded against the VM oracle):

  em_learned        - whole-state exact match, every field decoded by the net (the
                      status-quo readout; collapses to ~0 at OOD magnitude).
  em_digits_oracle  - THE HEADLINE. Same net predictions, but the numeric digit payload
                      is supplied by a perfect ALU (arithmetic offloaded). Everything
                      else (pc / type / sign / flags / which-slot-changed) is still the
                      net's job, so this is NOT "just running the VM" -- it isolates
                      whether the net's *structural* prediction is correct.
  pc                - next-pc accuracy (single step).
  cmp_result        - correctness of comparison-op results (the value-dependent control
                      signal that small-magnitude data cannot teach to generalize).
  written_sign      - sign of the written register (a magnitude-invariant direction).
  full_traj_success - executor: fraction of WHOLE programs that stay exact end-to-end
                      under net-control + ALU-values (compounds control errors).
  control_acc       - executor: mean per-step next-pc accuracy over the rollout.

The challenge: raise the OOD numbers WITHOUT training on large magnitude (that would be
cheating -- the model must learn structure from small-magnitude data only). See
``docs/FRONTIER_CHALLENGE.md``.

Run (CPU ONLY -- a training job may be using the GPU):

    PYTHONPATH=. python scripts/frontier_benchmark.py [--ckpt artifacts/neurosym_model.pt]
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import torch

from execwm.data.dataset import collect_examples
from execwm.data.state_codec import EncodeError
from execwm.eval.checkpoint import load_checkpoint
from execwm.eval.neurosym import field_breakdown
from execwm.eval.neurosym_exec import StepRecord, _batch1, _instr_str
from execwm.substrate import vm as vmmod


# ---------------------------------------------------------------------------
# Robust executor evaluation
# ---------------------------------------------------------------------------
# The shipped ``execwm.eval.neurosym_exec.evaluate_executor`` aborts with an
# ``EncodeError`` at OOD magnitude: once net-control diverges, the symbolic ALU can
# compute a register value outside the codec's digit range, and grading that state
# calls ``scodec.encode(ns)`` which raises. ``scripts/analysis_divergence_cause.py``
# documents this and works around it with a faithful, guarded replica. We do the same
# here so the scorecard is robust on any checkpoint. The guard CHANGES NO GRADING
# OUTCOME: a post-divergence unencodable state can never exact-match the (always
# in-range) ground truth, so it is scored ``state_exact = False`` either way -- it only
# lets an already-diverged program finish instead of crashing the whole run.


@torch.no_grad()
def _neurosym_execute(model, scodec, acodec, ex, device, *, max_steps=None):
    """Net-control + symbolic-ALU rollout with an EncodeError guard (see above)."""
    model.eval()
    program = ex.trace.program
    gt = ex.trace.states
    T = len(ex.trace.actions)
    if max_steps is not None:
        T = min(T, max_steps)
    cur = gt[0].copy()
    records: list[StepRecord] = []
    diverged = False
    for t in range(T):
        if cur.halted or not (0 <= cur.pc < len(program)):
            break
        instr = program[cur.pc]
        s_t = _batch1(scodec.encode(cur).as_dict(), device)
        a_t = _batch1(acodec.encode(instr).as_dict(), device)
        logits = model.heads(model.dynamics(model.encode(s_t), model.action(a_t)))
        pred_pc = int(logits["pc"].argmax(-1).item())
        pred_halted = bool(logits["halted"].argmax(-1).item())
        pred_error = bool(logits["error"].argmax(-1).item())
        try:
            alu = vmmod.step(cur, instr)
        except Exception:  # noqa: BLE001 - VM trap (div0 / OOB)
            alu = cur.copy()
            alu.error = True
            alu.halted = True
        ns = alu.copy()
        ns.pc = pred_pc
        ns.halted = pred_halted
        ns.error = pred_error
        true_next = gt[t + 1] if t + 1 < len(gt) else None
        true_pc = int(true_next.pc) if true_next is not None else -1
        control_ok = (pred_pc == true_pc)
        if true_next is None:
            state_exact = False
        else:
            try:
                state_exact = scodec.exact_match(scodec.encode(ns),
                                                 scodec.encode(true_next))
            except EncodeError:
                state_exact = False  # out-of-range post-divergence value -> no match
        if not state_exact:
            diverged = True
        records.append(StepRecord(
            t=t, pc=int(cur.pc), instr_str=_instr_str(instr), pred_pc=pred_pc,
            true_pc=true_pc, control_ok=control_ok, state_exact=bool(state_exact),
            diverged=diverged))
        try:
            scodec.encode(ns)  # if next encode(cur) would raise, stop (already recorded)
        except EncodeError:
            break
        cur = ns
    full_exact = len(records) > 0 and all(r.state_exact for r in records)
    return records, len(records), full_exact


def evaluate_executor(model, scodec, acodec, examples, device, *, max_steps=None):
    """Aggregate the robust rollout into the same dict shape the shipped one returns."""
    n_full = 0
    step_ok = step_tot = 0
    ctrl_ok = 0
    horizons = []
    for ex in examples:
        recs, nsteps, full = _neurosym_execute(model, scodec, acodec, ex, device,
                                               max_steps=max_steps)
        if nsteps == 0:
            continue
        n_full += int(full)
        step_ok += sum(r.state_exact for r in recs)
        step_tot += nsteps
        ctrl_ok += sum(r.control_ok for r in recs)
        h = 0
        for r in recs:
            if r.state_exact:
                h += 1
            else:
                break
        horizons.append(h)
    n = len(horizons)
    return {
        "full_trajectory_success": n_full / n if n else float("nan"),
        "per_step_state_exact": step_ok / step_tot if step_tot else float("nan"),
        "control_accuracy": ctrl_ok / step_tot if step_tot else float("nan"),
        "mean_exact_horizon": sum(horizons) / n if n else float("nan"),
        "n_programs": n,
        "n_steps": step_tot,
    }


# --- metrics that define the frontier scorecard ----------------------------
# (label, key, source) where source is "field" (field_breakdown) or "exec"
# (evaluate_executor). Order is the print order of the table rows.
SCORECARD = [
    ("em_learned",         "em_learned",              "field"),
    ("em_digits_oracle",   "em_digits_oracle",        "field"),
    ("pc",                 "pc",                      "field"),
    ("cmp_result",         "cmp_result",              "field"),
    ("written_sign",       "written_sign",            "field"),
    ("full_traj_success",  "full_trajectory_success", "exec"),
    ("control_acc",        "control_accuracy",        "exec"),
]


def _fmt(x) -> str:
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return "  nan"


def _scorecard_for(field_m: dict, exec_m: dict) -> dict[str, float]:
    """Pull the scorecard metrics out of the two metric dicts for one split."""
    out: dict[str, float] = {}
    for label, key, src in SCORECARD:
        src_dict = field_m if src == "field" else exec_m
        out[label] = float(src_dict.get(key, float("nan")))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/neurosym_model.pt",
                    help="checkpoint to score (any slotted world-model checkpoint)")
    ap.add_argument("--n", type=int, default=300,
                    help="episodes per split (in-dist and OOD)")
    ap.add_argument("--max-len", type=int, default=18,
                    help="max episode length for the single-step field breakdown")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # CPU ONLY. A GPU/MPS training job may be running -- never touch it here.
    device = torch.device("cpu")
    print(f"=== Frontier scorecard :: {args.ckpt} (device={device}) ===", flush=True)

    ck = load_checkpoint(args.ckpt, device=device)
    model, scodec, acodec, spec = ck["model"], ck["scodec"], ck["acodec"], ck["spec"]
    model.to(device).eval()
    meta = ck.get("meta", {})
    print(f"loaded: model={type(model).__name__}  "
          f"spec(max_const={spec.max_const}, max_input_val={spec.max_input_val})  "
          f"meta={meta}", flush=True)

    # In-distribution = the checkpoint's own (small-magnitude) training spec.
    # Magnitude-OOD = the canonical large-magnitude split used across the findings.
    indist_spec = spec
    ood_spec = replace(spec, max_const=400, max_input_val=400)

    print("\nbuilding eval sets (runs the VM oracle to generate traces)...", flush=True)
    indist_ex, _ = collect_examples(indist_spec, args.n, lambda ex: True,
                                    args.seed + 99, scodec, acodec)
    ood_ex, ood_att = collect_examples(ood_spec, args.n, lambda ex: True,
                                       args.seed + 777, scodec, acodec)
    print(f"  in-dist episodes : {len(indist_ex)}", flush=True)
    print(f"  OOD episodes     : {len(ood_ex)} (from {ood_att} attempts)", flush=True)

    # --- single-step structural breakdown (em_*, pc, cmp_result, written_sign) ---
    print("\nscoring single-step field breakdown...", flush=True)
    indist_field = field_breakdown(model, indist_ex, scodec, acodec, device,
                                   max_len=args.max_len)
    ood_field = field_breakdown(model, ood_ex, scodec, acodec, device,
                                max_len=args.max_len)

    # --- multi-step executor (full_trajectory_success, control_accuracy) ---
    print("scoring multi-step executor (net-control + ALU-values)...", flush=True)
    indist_exec = evaluate_executor(model, scodec, acodec, indist_ex, device)
    ood_exec = evaluate_executor(model, scodec, acodec, ood_ex, device)

    indist = _scorecard_for(indist_field, indist_exec)
    ood = _scorecard_for(ood_field, ood_exec)

    # --- THE canonical scorecard table ---
    print("\n" + "=" * 60)
    print("# FRONTIER SCORECARD")
    print("=" * 60)
    print(f"\n| {'metric':<18} | {'in-dist':>9} | {'OOD':>9} | {'delta':>9} |")
    print(f"|{'-' * 20}|{'-' * 11}|{'-' * 11}|{'-' * 11}|")
    for label, _key, _src in SCORECARD:
        i, o = indist[label], ood[label]
        delta = o - i
        marker = "  <-- HEADLINE" if label == "em_digits_oracle" else ""
        print(f"| {label:<18} | {_fmt(i):>9} | {_fmt(o):>9} | "
              f"{delta:>+9.3f} |{marker}")
    print(f"\n(in-dist n={indist_field.get('n', 0)} steps / "
          f"{indist_exec.get('n_programs', 0)} programs; "
          f"OOD n={ood_field.get('n', 0)} steps / "
          f"{ood_exec.get('n_programs', 0)} programs)")

    # --- THE THREE NUMBERS TO BEAT (read live from this run) ---
    print("\n" + "=" * 60)
    print("# NUMBERS TO BEAT  (the open challenge -- OOD, no large-magnitude training)")
    print("=" * 60)
    print(f"  OOD em_digits_oracle ............. {_fmt(ood['em_digits_oracle'])}   "
          f"(structural readout w/ arithmetic offloaded)")
    print(f"  OOD cmp_result ................... {_fmt(ood['cmp_result'])}   "
          f"(comparison-result correctness)")
    print(f"  OOD executor full_traj_success ... {_fmt(ood['full_traj_success'])}   "
          f"(whole-program exactness)")
    print("\nBeat these by improving the model's STRUCTURE generalization "
          "(encoder / comparison head / prior),")
    print("NOT by training on large magnitude. See docs/FRONTIER_CHALLENGE.md.")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
