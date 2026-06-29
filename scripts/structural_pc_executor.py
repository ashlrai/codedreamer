"""THE STRUCTURAL-PC EXECUTOR: offload the ISA's deterministic pc-advance, keep only
the genuinely-learned control decision (branch taken / not-taken) on the net.

Motivation (FINDINGS_FRONTIER.md §3): out of distribution, ~47% of the neurosymbolic
executor's first-divergences are *wrong-next-pc errors on plain ADD/SUB steps* -- a
failure mode that barely exists in-distribution. Those errors are SPURIOUS: for any
non-jump instruction the next pc is, by the ISA (see ``execwm/substrate/vm.py``),
deterministically ``pc + 1``. There is nothing to predict. The net only mispredicts it
because large *input* operands corrupt the encoder. So we should never have asked the
net for it.

This script builds a variant executor that separates STRUCTURE from the one LEARNED thing:

  * NON-control ops (CONST/MOV/ADD/SUB/.../LOAD/STORE/HALT): the symbolic ALU (``vm.step``)
    realizes BOTH the values AND the next pc. For these ops ``vm.step``'s pc is purely
    structural -- ``pc+1`` (or a no-advance halt/trap) -- it depends on no value comparison.
    The net is NOT consulted.
  * JMP: the target is an immediate in the instruction (structural). ALU handles it; net
    not consulted.
  * JZ / JNZ (the ONLY value-dependent control in the ISA): the NET decides the branch.
    We encode the current state, run the model, read its predicted next pc, and SNAP it to
    the nearer of {pc+1 (not-taken), target (taken)}. So the net only decides taken vs
    not-taken -- the genuinely learned part (it hinges on comparing a value to zero).

We compare this against the current ``neurosym_execute`` (net predicts the next pc every
step) on an in-distribution set and a magnitude-OOD set
(``replace(spec, max_const=400, max_input_val=400)``), reporting for each:
  * full-trajectory success (every rollout step exact-matches the VM oracle),
  * per-step exact (mean over rollout steps),
  * branch-decision accuracy on JZ/JNZ steps only.

Branch-decision accuracy is measured TEACHER-FORCED (from the true state gt[t] at every
true JZ/JNZ step) so it is a clean, identical-step comparison of the two decision RULES
(free-argmax vs snap-to-ISA) on the value-comparison frontier, uncontaminated by rollout
divergence. Full-trajectory / per-step come from the actual free-running rollouts. Every
state is graded against the VM oracle via ``scodec.exact_match``.

CPU ONLY (a GPU/MPS training job is running). Runs in-process, no background work.

    PYTHONPATH=. python scripts/structural_pc_executor.py [--n 300]
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace

import torch

from execwm.data.dataset import collect_examples
from execwm.data.state_codec import EncodeError
from execwm.eval.checkpoint import load_checkpoint
from execwm.eval.neurosym_exec import (StepRecord, _batch1, _instr_str,
                                       neurosym_execute)
from execwm.substrate import vm as vmmod
from execwm.substrate.vm import Op

BRANCH_OPS = (Op.JZ, Op.JNZ)  # the only value-dependent control flow in the ISA


# ---------------------------------------------------------------------------
# Net branch decision (the one learned thing)
# ---------------------------------------------------------------------------

@torch.no_grad()
def _net_pred_pc(model, scodec, acodec, cur, instr, device) -> int:
    """Read the net's predicted next pc from the current state (one forward pass)."""
    s_t = _batch1(scodec.encode(cur).as_dict(), device)
    a_t = _batch1(acodec.encode(instr).as_dict(), device)
    logits = model.heads(model.dynamics(model.encode(s_t), model.action(a_t)))
    return int(logits["pc"].argmax(-1).item())


def _snap_branch(pred_pc: int, not_taken: int, taken: int) -> int:
    """Snap a free pc prediction to the nearer of the two ISA-legal branch targets.

    Ties (equidistant) resolve to not-taken (pc+1) -- the fall-through default.
    """
    d_taken = abs(pred_pc - taken)
    d_not = abs(pred_pc - not_taken)
    return taken if d_taken < d_not else not_taken


# ---------------------------------------------------------------------------
# The structural-pc rollout
# ---------------------------------------------------------------------------

@dataclass
class SRec:
    pc: int
    is_branch: bool
    branch_ok: bool      # only meaningful when is_branch (and pre-divergence)
    state_exact: bool


@torch.no_grad()
def structural_pc_execute(model, scodec, acodec, ex, device, *, max_steps=None):
    """Run ``ex`` with structural pc-advance for non-branch ops and net-decided
    branches for JZ/JNZ. Values always come from the symbolic ALU (``vm.step``).

    Returns (records, n_steps, full_exact). Same VM-oracle grading as the current
    executor, with the EncodeError guard (post-divergence the ALU can produce an
    out-of-range value; that can never match in-range ground truth, so we score it
    state_exact=False rather than crash -- changes no grading outcome).
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

        # symbolic ALU realizes values (and, for non-branch ops, the structural pc).
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
            # the genuinely-learned decision: net picks taken vs not-taken.
            pred_pc = _net_pred_pc(model, scodec, acodec, cur, instr, device)
            not_taken = int(cur.pc) + 1
            taken = int(instr.target)
            ns.pc = _snap_branch(pred_pc, not_taken, taken)
            # halted/error are unchanged by JZ/JNZ -> alu already correct.
        # else: ns = alu in full -> structural pc+1 / target(JMP) / halt-trap, no net.

        true_next = gt[t + 1] if t + 1 < len(gt) else None
        if true_next is None:
            state_exact = False
        else:
            if is_branch:
                branch_ok = (int(ns.pc) == int(true_next.pc))
            try:
                state_exact = scodec.exact_match(scodec.encode(ns),
                                                 scodec.encode(true_next))
            except EncodeError:
                state_exact = False

        records.append(SRec(pc=int(cur.pc), is_branch=is_branch,
                            branch_ok=branch_ok, state_exact=bool(state_exact)))
        try:
            scodec.encode(ns)  # if ns itself is unencodable, next encode(cur) would raise
        except EncodeError:
            break
        cur = ns
    full_exact = len(records) > 0 and all(r.state_exact for r in records)
    return records, len(records), full_exact


# ---------------------------------------------------------------------------
# Current executor: crash-safe wrapper around the SHIPPED neurosym_execute
# ---------------------------------------------------------------------------

@torch.no_grad()
def _neurosym_execute_guarded(model, scodec, acodec, ex, device, *, max_steps=None):
    """Faithful replica of the shipped ``neurosym_execute`` with the EncodeError guard,
    used only as a fallback when the shipped engine would crash on a post-divergence
    out-of-range ALU value (the guard changes no grading outcome)."""
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
        except Exception:  # noqa: BLE001
            alu = cur.copy(); alu.error = True; alu.halted = True
        ns = alu.copy()
        ns.pc = pred_pc; ns.halted = pred_halted; ns.error = pred_error
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
                state_exact = False
        if not state_exact:
            diverged = True
        records.append(StepRecord(
            t=t, pc=int(cur.pc), instr_str=_instr_str(instr), pred_pc=pred_pc,
            true_pc=true_pc, control_ok=control_ok, state_exact=bool(state_exact),
            diverged=diverged))
        try:
            scodec.encode(ns)
        except EncodeError:
            break
        cur = ns
    full_exact = len(records) > 0 and all(r.state_exact for r in records)
    return records, len(records), full_exact


# ---------------------------------------------------------------------------
# Teacher-forced branch-decision accuracy (clean, identical-step comparison)
# ---------------------------------------------------------------------------

@torch.no_grad()
def branch_decision_teacher_forced(model, scodec, acodec, examples, device) -> dict:
    """For every TRUE JZ/JNZ step (encoded from the true state gt[t]), run the net once
    and score two decision rules against the true next pc:
      free  -- argmax pc directly (the current executor's rule)
      snap  -- snap argmax to the nearer of {pc+1, target} (the structural rule)
    Returns counts + accuracies. Teacher-forced so both rules see identical steps,
    isolating the value-comparison frontier from rollout divergence.
    """
    n = free_ok = snap_ok = 0
    for ex in examples:
        program = ex.trace.program
        gt = ex.trace.states
        for t in range(len(ex.trace.actions)):
            cur = gt[t]
            if not (0 <= cur.pc < len(program)):
                continue
            instr = program[cur.pc]
            if instr.op not in BRANCH_OPS:
                continue
            true_pc = int(gt[t + 1].pc)
            pred_pc = _net_pred_pc(model, scodec, acodec, cur, instr, device)
            not_taken = int(cur.pc) + 1
            taken = int(instr.target)
            snapped = _snap_branch(pred_pc, not_taken, taken)
            n += 1
            free_ok += int(pred_pc == true_pc)
            snap_ok += int(snapped == true_pc)
    return {
        "n_branch_steps": n,
        "free_acc": free_ok / n if n else float("nan"),
        "snap_acc": snap_ok / n if n else float("nan"),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _agg_current(model, scodec, acodec, examples, device) -> dict:
    """Full-traj + per-step for the CURRENT executor, calling the SHIPPED
    neurosym_execute, with the guarded replica as a crash fallback."""
    n_full = step_ok = step_tot = n_prog = n_fallback = 0
    horizons = []
    for ex in examples:
        try:
            recs, nsteps, full = neurosym_execute(model, scodec, acodec, ex, device)
        except EncodeError:
            n_fallback += 1
            recs, nsteps, full = _neurosym_execute_guarded(
                model, scodec, acodec, ex, device)
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
        "n_programs": n_prog, "n_steps": step_tot, "n_fallback": n_fallback,
    }


def _agg_structural(model, scodec, acodec, examples, device) -> dict:
    """Full-traj + per-step for the STRUCTURAL-PC executor."""
    n_full = step_ok = step_tot = n_prog = 0
    horizons = []
    for ex in examples:
        recs, nsteps, full = structural_pc_execute(model, scodec, acodec, ex, device)
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

def _row(name: str, e: dict, branch_acc: float) -> str:
    g = lambda k: f"{e.get(k, float('nan')):.3f}"
    return (f"| {name} | {g('full_trajectory_success')} | {g('per_step_state_exact')} "
            f"| {branch_acc:.3f} | {e.get('mean_exact_horizon', float('nan')):.1f} "
            f"| {e.get('n_programs', 0)} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/neurosym_model.pt")
    ap.add_argument("--n", type=int, default=300, help="programs per split")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cpu")  # CPU ONLY -- GPU/MPS is busy with a training job
    print(f"=== Structural-pc executor experiment on {args.ckpt} (device={device}) ===",
          flush=True)

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

    # --- rollouts ---
    print("\nrunning rollouts (current + structural-pc, both splits)...", flush=True)
    cur_i = _agg_current(model, scodec, acodec, indist_ex, device)
    cur_o = _agg_current(model, scodec, acodec, ood_ex, device)
    str_i = _agg_structural(model, scodec, acodec, indist_ex, device)
    str_o = _agg_structural(model, scodec, acodec, ood_ex, device)

    # --- teacher-forced branch-decision accuracy (free vs snap), per split ---
    bd_i = branch_decision_teacher_forced(model, scodec, acodec, indist_ex, device)
    bd_o = branch_decision_teacher_forced(model, scodec, acodec, ood_ex, device)

    if cur_i["n_fallback"] or cur_o["n_fallback"]:
        print(f"\n[note] shipped neurosym_execute hit EncodeError on "
              f"{cur_i['n_fallback']} in-dist / {cur_o['n_fallback']} OOD programs; "
              f"used guarded replica (no grading change) for those.", flush=True)

    print("\n# Current executor (net predicts next pc EVERY step)\n", flush=True)
    print("| split | full-traj success | per-step exact | branch-decision acc | mean horizon | n |")
    print("|---|---|---|---|---|---|")
    print(_row("in-distribution (val<=~30)", cur_i, bd_i["free_acc"]))
    print(_row("magnitude-OOD (val~300-800)", cur_o, bd_o["free_acc"]))

    print("\n# Structural-pc executor (ISA advances non-branch pc; net only decides JZ/JNZ)\n",
          flush=True)
    print("| split | full-traj success | per-step exact | branch-decision acc | mean horizon | n |")
    print("|---|---|---|---|---|---|")
    print(_row("in-distribution (val<=~30)", str_i, bd_i["snap_acc"]))
    print(_row("magnitude-OOD (val~300-800)", str_o, bd_o["snap_acc"]))

    print(f"\n[branch steps graded: in-dist {bd_i['n_branch_steps']}, "
          f"OOD {bd_o['n_branch_steps']} (teacher-forced JZ/JNZ steps)]", flush=True)

    # --- headline deltas ---
    def d(a, b):
        return b - a
    print("\n=== Headline ===", flush=True)
    print(f"OOD full-trajectory success: current {cur_o['full_trajectory_success']:.3f} "
          f"-> structural {str_o['full_trajectory_success']:.3f} "
          f"(delta {d(cur_o['full_trajectory_success'], str_o['full_trajectory_success']):+.3f})")
    print(f"OOD per-step exact:          current {cur_o['per_step_state_exact']:.3f} "
          f"-> structural {str_o['per_step_state_exact']:.3f} "
          f"(delta {d(cur_o['per_step_state_exact'], str_o['per_step_state_exact']):+.3f})")
    print(f"OOD branch-decision acc:     free {bd_o['free_acc']:.3f} "
          f"-> snap {bd_o['snap_acc']:.3f} "
          f"(delta {d(bd_o['free_acc'], bd_o['snap_acc']):+.3f})")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
