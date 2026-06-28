"""Diagnostic: WHY does the multi-step neurosymbolic executor degrade out-of-distribution?

The executor (net drives control flow, symbolic ALU computes register/heap values) gets
~0.39 full-program success at magnitude-OOD vs ~0.70 in-distribution (see
``FINDINGS_NEUROSYM.md``). This script tests a specific mechanistic hypothesis:

    OOD first-divergences concentrate on COMPARISON / BRANCH instructions
    (value-dependent control), NOT on arithmetic or plain sequential steps.

Method (everything graded against the VM oracle, exactly as the executor already is):
  * Load the EXISTING checkpoint ``artifacts/neurosym_model.pt`` on CPU.
  * Build a magnitude-OOD example set (``replace(spec, max_const=400, max_input_val=400)``)
    and an in-distribution set (the checkpoint's own ``spec``) for comparison.
  * Run ``neurosym_execute`` on each program. For every program that is NOT full_exact,
    find the FIRST ``StepRecord`` with ``state_exact == False`` and identify the op of the
    instruction executed at that step (``ex.trace.program[record.pc].op``).
  * Classify that first-divergence op into
    {comparison, jump/branch, arithmetic, movement, heap, other} and record whether the
    failure was a CONTROL error (``control_ok == False``) or a VALUE/FLAG error
    (``control_ok == True`` but state still wrong).

Run (CPU only -- a training job is using the GPU):

    PYTHONPATH=. python scripts/analysis_divergence_cause.py [--n 300]
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace

import torch

from execwm.data.dataset import collect_examples
from execwm.data.state_codec import EncodeError
from execwm.eval.checkpoint import load_checkpoint
from execwm.eval.neurosym_exec import StepRecord, _batch1, _instr_str
from execwm.substrate import vm as vmmod
from execwm.substrate.vm import ARITH_OPS, CMP_OPS, JUMP_OPS, Op


@torch.no_grad()
def neurosym_execute(model, scodec, acodec, ex, device, *, max_steps=None):
    """Faithful replica of ``execwm.eval.neurosym_exec.neurosym_execute``.

    Identical net-control + symbolic-ALU rollout and VM grading, with ONE robustness
    guard: after control diverges the ALU can compute a value outside the codec's
    digit range, which makes ``scodec.encode(ns)`` raise ``EncodeError``. Such a state
    can never exact-match the (always in-range) ground truth, so we treat the
    EncodeError as ``state_exact = False`` instead of letting it abort the rollout.
    This changes no grading outcome -- it only lets a program that has already diverged
    finish so its FIRST divergence step is recorded. (Cannot edit the engine; existing
    files unchanged.)
    """
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
        z = model.encode(s_t)
        logits = model.heads(model.dynamics(z, model.action(a_t)))
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
                # ns carries an out-of-range value (post-divergence) -> cannot match.
                state_exact = False
        if not state_exact:
            diverged = True
        records.append(StepRecord(
            t=t, pc=int(cur.pc), instr_str=_instr_str(instr), pred_pc=pred_pc,
            true_pc=true_pc, control_ok=control_ok, state_exact=bool(state_exact),
            diverged=diverged))
        # if the current state is itself unencodable, the next encode(cur) would raise;
        # we have already recorded the divergence, so stop here.
        try:
            scodec.encode(ns)
        except EncodeError:
            break
        cur = ns
    full_exact = len(records) > 0 and all(r.state_exact for r in records)
    return records, len(records), full_exact

# --- op category map ---------------------------------------------------------
MOVE_OPS = (Op.CONST, Op.MOV)
HEAP_OPS = (Op.LOAD, Op.STORE)

CATEGORIES = ["comparison", "jump/branch", "arithmetic", "movement", "heap", "other"]


def categorize(op: Op) -> str:
    if op in CMP_OPS:
        return "comparison"
    if op in JUMP_OPS:
        return "jump/branch"
    if op in ARITH_OPS:
        return "arithmetic"
    if op in MOVE_OPS:
        return "movement"
    if op in HEAP_OPS:
        return "heap"
    return "other"


def analyze(model, scodec, acodec, examples, device):
    """Return aggregate stats over a list of examples for the first-divergence diagnostic."""
    n_programs = 0
    n_full = 0
    cat_counts: Counter[str] = Counter()          # category of first-divergence op
    cat_op_counts: Counter[str] = Counter()        # raw op name at first divergence
    control_err = 0                                # control_ok == False at divergence
    value_err = 0                                  # control_ok == True but state wrong
    # control-vs-value split *within* each category
    cat_control: Counter[str] = Counter()
    cat_value: Counter[str] = Counter()

    for ex in examples:
        recs, nsteps, full = neurosym_execute(model, scodec, acodec, ex, device)
        if nsteps == 0:
            continue
        n_programs += 1
        if full:
            n_full += 1
            continue
        # first step whose neurosymbolic next-state != ground truth
        first = next((r for r in recs if not r.state_exact), None)
        if first is None:
            # full_exact False but no diverging record should not happen; guard anyway.
            continue
        op = ex.trace.program[first.pc].op
        cat = categorize(op)
        cat_counts[cat] += 1
        cat_op_counts[op.name] += 1
        if not first.control_ok:
            control_err += 1
            cat_control[cat] += 1
        else:
            value_err += 1
            cat_value[cat] += 1

    n_failures = n_programs - n_full
    return {
        "n_programs": n_programs,
        "n_full": n_full,
        "full_success": n_full / n_programs if n_programs else float("nan"),
        "n_failures": n_failures,
        "cat_counts": cat_counts,
        "cat_op_counts": cat_op_counts,
        "control_err": control_err,
        "value_err": value_err,
        "cat_control": cat_control,
        "cat_value": cat_value,
    }


def _pct(num, den):
    return f"{(100.0 * num / den):5.1f}%" if den else "   -  "


def print_report(name: str, stats: dict) -> None:
    nf = stats["n_failures"]
    print(f"\n### {name}")
    print(f"  programs evaluated : {stats['n_programs']}")
    print(f"  full-program exact : {stats['n_full']}  "
          f"({100.0 * stats['full_success']:.1f}% success)")
    print(f"  failures (analyzed): {nf}")
    print(f"\n  First-divergence op category (share of {nf} failures):")
    print(f"  {'category':<14} {'count':>6} {'share':>8}")
    for cat in CATEGORIES:
        c = stats["cat_counts"].get(cat, 0)
        print(f"  {cat:<14} {c:>6} {_pct(c, nf):>8}")
    comp_branch = stats["cat_counts"].get("comparison", 0) + stats["cat_counts"].get("jump/branch", 0)
    print(f"  {'-> cmp+branch':<14} {comp_branch:>6} {_pct(comp_branch, nf):>8}")
    print(f"\n  Control-vs-value split of the failure:")
    print(f"    control error (wrong next pc) : {stats['control_err']:>5}  "
          f"{_pct(stats['control_err'], nf)}")
    print(f"    value/flag error (pc ok)      : {stats['value_err']:>5}  "
          f"{_pct(stats['value_err'], nf)}")
    print(f"\n  Raw op at first divergence:")
    for op_name, c in stats["cat_op_counts"].most_common():
        print(f"    {op_name:<8} {c:>5}  {_pct(c, nf)}")
    print(f"\n  Control-vs-value split *within* each category:")
    print(f"  {'category':<14} {'control':>8} {'value':>8}")
    for cat in CATEGORIES:
        cc = stats["cat_control"].get(cat, 0)
        vv = stats["cat_value"].get(cat, 0)
        if cc or vv:
            print(f"  {cat:<14} {cc:>8} {vv:>8}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/neurosym_model.pt")
    ap.add_argument("--n", type=int, default=300, help="programs per split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cpu")  # CPU ONLY -- GPU is busy with a training job
    print(f"=== Divergence-cause diagnostic on {args.ckpt} (device={device}) ===")

    ck = load_checkpoint(args.ckpt, device=device)
    model, scodec, acodec, spec = ck["model"], ck["scodec"], ck["acodec"], ck["spec"]
    print(f"loaded checkpoint: spec(max_const={spec.max_const}, "
          f"max_input_val={spec.max_input_val}), meta={ck['meta']}")

    indist_spec = spec  # the checkpoint's own (small-magnitude) training distribution
    ood_spec = replace(spec, max_const=400, max_input_val=400)  # magnitude-OOD

    print("\ncollecting examples (this runs the VM oracle to build traces)...")
    indist_ex, _ = collect_examples(indist_spec, args.n, lambda ex: True,
                                    args.seed + 99, scodec, acodec)
    ood_ex, ood_att = collect_examples(ood_spec, args.n, lambda ex: True,
                                       args.seed + 777, scodec, acodec)
    print(f"  in-dist programs : {len(indist_ex)}")
    print(f"  OOD programs     : {len(ood_ex)} (from {ood_att} attempts)")

    indist_stats = analyze(model, scodec, acodec, indist_ex, device)
    ood_stats = analyze(model, scodec, acodec, ood_ex, device)

    print_report("IN-DISTRIBUTION (val<=~30)", indist_stats)
    print_report("MAGNITUDE-OOD (val~300-800)", ood_stats)

    # --- compact side-by-side category table ---
    print("\n\n=== Side-by-side: first-divergence category share (in-dist vs OOD) ===")
    print(f"{'category':<14} {'in-dist':>10} {'OOD':>10}")
    for cat in CATEGORIES:
        i = indist_stats["cat_counts"].get(cat, 0)
        o = ood_stats["cat_counts"].get(cat, 0)
        print(f"{cat:<14} {_pct(i, indist_stats['n_failures']):>10} "
              f"{_pct(o, ood_stats['n_failures']):>10}")
    ic = indist_stats["cat_counts"]; oc = ood_stats["cat_counts"]
    i_cb = ic.get("comparison", 0) + ic.get("jump/branch", 0)
    o_cb = oc.get("comparison", 0) + oc.get("jump/branch", 0)
    print(f"{'cmp+branch':<14} {_pct(i_cb, indist_stats['n_failures']):>10} "
          f"{_pct(o_cb, ood_stats['n_failures']):>10}")
    print(f"\n{'control-err share':<14} "
          f"{_pct(indist_stats['control_err'], indist_stats['n_failures']):>10} "
          f"{_pct(ood_stats['control_err'], ood_stats['n_failures']):>10}")
    print("\nDONE")


if __name__ == "__main__":
    main()
