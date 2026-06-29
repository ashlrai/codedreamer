"""THE OFFLOAD LADDER: attribute the neurosymbolic executor's magnitude-OOD failure
across the three offloadable pieces by walking one extra rung beyond
FINDINGS_FRONTIER.md §5.

§5 showed that handing pc-advance back to the ISA (structural-pc) recovers most of the
OOD degradation, leaving only the value-dependent branch decision on the net. This script
finishes the decomposition. It builds three executor variants and runs all three on the
SAME in-distribution and magnitude-OOD program sets, reporting full-trajectory success and
per-step exact for each:

  RUNG 1 -- baseline: the net predicts the next pc EVERY step (the shipped
            ``neurosym_execute``). Values come from the symbolic ALU.
  RUNG 2 -- structural-pc: the ISA advances the pc for every non-branch op (and to the
            immediate target for JMP); the net is consulted ONLY for the JZ/JNZ
            taken/not-taken decision. (Reuses ``structural_pc_execute`` from
            ``scripts/structural_pc_executor.py``.)
  RUNG 3 -- structural-pc + structural-branch (NEW): additionally resolve JZ/JNZ
            *structurally* from the concrete ALU operand value -- JZ is taken iff the
            operand value == 0, JNZ iff != 0, computed from the current concrete state.
            This comparison is magnitude-invariant (a value's distance from zero is exact
            regardless of how many digits it has), so the net is consulted for NOTHING.
            Values still come from the ALU.

Every produced state is graded against the VM oracle with ``scodec.exact_match`` via the
robust ``_exact`` helper (out-of-range post-divergence states score False instead of
crashing -- which changes no grading outcome, since they can never equal in-range ground
truth).

HONEST FRAMING (see ``docs/finding_offload_ladder.md``): rung 3 is, by construction,
VM-equivalent -- it offloads arithmetic (ALU), the structural pc-advance (ISA), AND the
control comparison (operand-vs-zero) -- so it should be ~1.0 on BOTH splits. That is the
point, not a bug: the OOD gap decomposes ENTIRELY into offloadable pieces. Given value
access, the net's irreducible *learnable* contribution to step-by-step execution is ~zero.
The net's real value is only in regimes where you can't or won't run the VM (planning over
many edits, partial/uninstantiated programs).

CPU ONLY (a GPU/MPS training job is running -- never touch MPS). Runs in-process, no
background work.

    PYTHONPATH=. python scripts/offload_ladder.py [--n 300]
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import torch

from execwm.data.dataset import collect_examples
from execwm.data.state_codec import EncodeError
from execwm.eval.checkpoint import load_checkpoint
from execwm.eval.neurosym_exec import _exact, neurosym_execute
from execwm.substrate import vm as vmmod
from execwm.substrate.vm import Op

# Reuse the rung-1 and rung-2 machinery verbatim from the §5 script so the ladder is an
# apples-to-apples extension (same guarded grading, same horizon accounting).
from scripts.structural_pc_executor import (BRANCH_OPS, SRec, _agg_current,
                                            _agg_structural,
                                            _neurosym_execute_guarded,
                                            structural_pc_execute)


# ---------------------------------------------------------------------------
# RUNG 3: structural-pc + structural-branch (net consulted for NOTHING)
# ---------------------------------------------------------------------------

@torch.no_grad()
def structural_full_execute(model, scodec, acodec, ex, device, *, max_steps=None):
    """Run ``ex`` with the pc advanced structurally for non-branch ops AND the JZ/JNZ
    branch resolved structurally from the concrete operand value (magnitude-invariant).

    The net is never called. Values come from the symbolic ALU (``vm.step``); control
    comes from the ISA's pc-advance plus a single ``operand == 0`` comparison on the
    current concrete state. By construction this reproduces the VM.

    Returns (records, n_steps, full_exact) with the same EncodeError guard as the other
    rungs (an out-of-range ALU value can never equal in-range ground truth -> score
    state_exact=False rather than crash; changes no grading outcome).
    """
    model.eval()
    program = ex.trace.program
    gt = ex.trace.states
    T = len(ex.trace.actions)
    if max_steps is not None:
        T = min(T, max_steps)
    cur = gt[0].copy()
    records: list[SRec] = []
    for t in range(T):
        if cur.halted or not (0 <= cur.pc < len(program)):
            break
        instr = program[cur.pc]

        # symbolic ALU realizes the value effect (and, for non-branch ops, the
        # structural pc -- pc+1 / JMP target / halt-trap, no comparison).
        try:
            alu = vmmod.step(cur, instr)
        except Exception:  # noqa: BLE001 - VM trap (div0 / OOB)
            alu = cur.copy()
            alu.error = True
            alu.halted = True
        ns = alu.copy()

        is_branch = instr.op in BRANCH_OPS
        branch_ok = False
        if is_branch:
            # STRUCTURAL BRANCH: resolve taken/not-taken from the concrete operand value
            # read off the current state. This is magnitude-invariant -- a value's
            # equality with zero is exact no matter how many digits it has -- so no net
            # and no digit head is involved.
            not_taken = int(cur.pc) + 1
            taken = int(instr.target)
            try:
                opval = vmmod._read(cur, instr.a)
            except Exception:  # noqa: BLE001 - undefined read (valid progs never hit this)
                ns.pc = int(alu.pc)
            else:
                if instr.op is Op.JZ:
                    ns.pc = taken if opval == 0 else not_taken
                else:  # Op.JNZ
                    ns.pc = taken if opval != 0 else not_taken
        # else: ns == alu in full -> structural pc, no net.

        true_next = gt[t + 1] if t + 1 < len(gt) else None
        if true_next is None:
            state_exact = False
        else:
            if is_branch:
                branch_ok = (int(ns.pc) == int(true_next.pc))
            state_exact = _exact(scodec, ns, scodec.encode(true_next))

        records.append(SRec(pc=int(cur.pc), is_branch=is_branch,
                            branch_ok=branch_ok, state_exact=bool(state_exact)))
        try:
            scodec.encode(ns)  # guard the next iteration's encode(cur)
        except EncodeError:
            break
        cur = ns
    full_exact = len(records) > 0 and all(r.state_exact for r in records)
    return records, len(records), full_exact


def _agg_structural_full(model, scodec, acodec, examples, device) -> dict:
    """Full-traj + per-step for the RUNG-3 (structural-pc + structural-branch) executor.

    Mirrors ``_agg_structural`` from the §5 script exactly so the three rungs are
    aggregated identically."""
    n_full = step_ok = step_tot = n_prog = 0
    horizons = []
    for ex in examples:
        recs, nsteps, full = structural_full_execute(model, scodec, acodec, ex, device)
        if nsteps == 0:
            continue
        n_prog += 1
        n_full += int(full)
        step_ok += sum(r.state_exact for r in recs)
        step_tot += nsteps
        h = 0
        for r in recs:
            if r.state_exact:
                h += 1
            else:
                break
        horizons.append(h)
    return {
        "full_trajectory_success": n_full / n_prog if n_prog else float("nan"),
        "per_step_state_exact": step_ok / step_tot if step_tot else float("nan"),
        "mean_exact_horizon": sum(horizons) / len(horizons) if horizons else float("nan"),
        "n_programs": n_prog, "n_steps": step_tot,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _row(name: str, e: dict) -> str:
    g = lambda k: f"{e.get(k, float('nan')):.3f}"
    return (f"| {name} | {g('full_trajectory_success')} | {g('per_step_state_exact')} "
            f"| {e.get('mean_exact_horizon', float('nan')):.1f} | {e.get('n_programs', 0)} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/neurosym_model.pt")
    ap.add_argument("--n", type=int, default=300, help="programs per split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cpu")  # CPU ONLY -- GPU/MPS is busy with a training job
    print(f"=== Offload ladder on {args.ckpt} (device={device}) ===", flush=True)

    ck = load_checkpoint(args.ckpt, device=device)
    model, scodec, acodec, spec = ck["model"], ck["scodec"], ck["acodec"], ck["spec"]
    model.to(device).eval()
    print(f"loaded checkpoint: spec(max_const={spec.max_const}, "
          f"max_input_val={spec.max_input_val}), meta={ck['meta']}", flush=True)

    indist_spec = spec
    ood_spec = replace(spec, max_const=400, max_input_val=400)

    print("\ncollecting examples (runs the VM oracle to build traces)...", flush=True)
    indist_ex, _ = collect_examples(indist_spec, args.n, lambda e: True,
                                    args.seed + 99, scodec, acodec)
    ood_ex, ood_att = collect_examples(ood_spec, args.n, lambda e: True,
                                       args.seed + 777, scodec, acodec)
    print(f"  in-dist programs : {len(indist_ex)}", flush=True)
    print(f"  OOD programs     : {len(ood_ex)} (from {ood_att} attempts)", flush=True)

    print("\nrunning rollouts for all three rungs on both splits...", flush=True)
    # rung 1 -- baseline (net predicts every pc); guarded fallback for crash-safety.
    r1_i = _agg_current(model, scodec, acodec, indist_ex, device)
    r1_o = _agg_current(model, scodec, acodec, ood_ex, device)
    # rung 2 -- structural-pc (net only decides JZ/JNZ).
    r2_i = _agg_structural(model, scodec, acodec, indist_ex, device)
    r2_o = _agg_structural(model, scodec, acodec, ood_ex, device)
    # rung 3 -- structural-pc + structural-branch (net consulted for nothing).
    r3_i = _agg_structural_full(model, scodec, acodec, indist_ex, device)
    r3_o = _agg_structural_full(model, scodec, acodec, ood_ex, device)

    if r1_i["n_fallback"] or r1_o["n_fallback"]:
        print(f"\n[note] shipped neurosym_execute hit EncodeError on "
              f"{r1_i['n_fallback']} in-dist / {r1_o['n_fallback']} OOD programs; "
              f"used guarded replica (no grading change) for those.", flush=True)

    print("\n# In-distribution (val<=~30)\n", flush=True)
    print("| rung | full-traj success | per-step exact | mean horizon | n |")
    print("|---|---|---|---|---|")
    print(_row("1. baseline (net predicts every pc)", r1_i))
    print(_row("2. structural-pc (net decides JZ/JNZ only)", r2_i))
    print(_row("3. structural-pc + structural-branch (no net)", r3_i))

    print("\n# Magnitude-OOD (val~300-800)\n", flush=True)
    print("| rung | full-traj success | per-step exact | mean horizon | n |")
    print("|---|---|---|---|---|")
    print(_row("1. baseline (net predicts every pc)", r1_o))
    print(_row("2. structural-pc (net decides JZ/JNZ only)", r2_o))
    print(_row("3. structural-pc + structural-branch (no net)", r3_o))

    print("\n=== Headline (OOD full-trajectory success) ===", flush=True)
    print(f"rung 1 baseline                  : {r1_o['full_trajectory_success']:.3f}")
    print(f"rung 2 + structural pc           : {r2_o['full_trajectory_success']:.3f} "
          f"(delta {r2_o['full_trajectory_success'] - r1_o['full_trajectory_success']:+.3f})")
    print(f"rung 3 + structural branch (NEW) : {r3_o['full_trajectory_success']:.3f} "
          f"(delta {r3_o['full_trajectory_success'] - r2_o['full_trajectory_success']:+.3f})")
    print("\nThe OOD gap decomposes entirely into offloadable pieces -- arithmetic (ALU),")
    print("structural pc (ISA), and the operand-vs-zero control comparison. Rung 3 is")
    print("VM-equivalent by construction, so its ~1.0 is the honest endpoint of the")
    print("decomposition, not a result: given value access, the net's irreducible")
    print("LEARNABLE contribution to execution is ~zero.")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
