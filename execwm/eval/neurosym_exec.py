"""The neurosymbolic executor: run a whole program forward where the **net drives
control flow** (predicts the next pc + flags from its latent each step) and a
**symbolic ALU computes register/heap values** (the offloaded arithmetic).

This is the multi-step payoff of `FINDINGS_NEUROSYM.md`: single-step showed the
magnitude wall is the digit head. Here we let the net actually *run* programs — only
control flow is learned, values are realized by `vm.step` — and measure whether full
trajectories stay exact at out-of-distribution magnitude, where a pure-net rollout
(values from the digit head) collapses from step one.

Honest framing: this is NOT "just the VM". The next pc is the **net's** prediction
(including resolving branch conditions from the latent); a control-flow mistake fetches
the wrong next instruction and compounds. The ALU only fills in the arithmetic the net
was never good at. The engine also returns per-step records for the interactive demo.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

import numpy as np

from ..data.state_codec import EncodedState
from ..data.torch_data import _ACTION_KEYS, _STATE_KEYS
from ..substrate import vm as vmmod


def _batch1(enc_dict: dict, device) -> dict:
    return {k: torch.as_tensor(v)[None].to(device) for k, v in enc_dict.items()}


@dataclass
class StepRecord:
    t: int
    pc: int                      # pc executed this step
    instr_str: str               # human-readable instruction
    pred_pc: int                 # net's predicted next pc
    true_pc: int                 # ground-truth next pc
    control_ok: bool             # did the net pick the right next pc?
    state_exact: bool            # neurosymbolic next state == ground truth?
    diverged: bool               # has control flow diverged from ground truth by now?


@torch.no_grad()
def neurosym_execute(model, scodec, acodec, ex, device, *, max_steps: int | None = None):
    """Run ``ex`` with net-control + ALU-values from the true initial state.

    Returns (records, n_steps, full_exact) where full_exact is True iff every step's
    neurosymbolic state matched ground truth (control never diverged and ALU exact).
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

        # symbolic ALU realizes the value effect (and the *true* control, which we
        # then overwrite with the net's prediction — control is the net's job).
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
        state_exact = (true_next is not None
                       and scodec.exact_match(scodec.encode(ns), scodec.encode(true_next)))
        if not state_exact:
            diverged = True
        records.append(StepRecord(
            t=t, pc=int(cur.pc), instr_str=_instr_str(instr), pred_pc=pred_pc,
            true_pc=true_pc, control_ok=control_ok, state_exact=bool(state_exact),
            diverged=diverged))
        cur = ns
    full_exact = len(records) > 0 and all(r.state_exact for r in records)
    return records, len(records), full_exact


def _instr_str(instr) -> str:
    parts = [instr.op.name]
    if instr.dst is not None:
        parts.append(str(instr.dst))
    for o in (instr.a, instr.b):
        if o is not None:
            parts.append(str(o))
    if instr.list_id is not None:
        parts.append(f"L{instr.list_id}")
    if instr.target is not None:
        parts.append(f"->{instr.target}")
    return " ".join(parts)


@torch.no_grad()
def evaluate_executor(model, scodec, acodec, examples, device, *,
                      max_steps: int | None = None) -> dict:
    """Aggregate neurosymbolic-executor exactness over a list of examples.

    Returns full-trajectory success rate, mean per-step state-exactness, mean
    control (next-pc) accuracy, and mean exact horizon (steps before first error).
    """
    n_full = 0
    step_ok = step_tot = 0
    ctrl_ok = 0
    horizons = []
    for ex in examples:
        recs, nsteps, full = neurosym_execute(model, scodec, acodec, ex, device,
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


def _decode_net_state(logits: dict, scodec):
    """Decode the net's full predicted next state (pure-net readout) to a MachineState."""
    am = lambda k: logits[k].argmax(-1)[0].cpu().numpy()
    enc = EncodedState(
        reg_type=am("reg_type"), reg_sign=am("reg_sign"), reg_digits=am("reg_digits"),
        heap_sign=am("heap_sign"), heap_digits=am("heap_digits"),
        pc=np.array(int(logits["pc"].argmax(-1)[0])),
        halted=np.array(int(logits["halted"].argmax(-1)[0])),
        error=np.array(int(logits["error"].argmax(-1)[0])),
    )
    return scodec.decode(enc)


def _regs_view(state, reg_names) -> dict:
    return {n: state.regs.get(n) for n in reg_names}


@torch.no_grad()
def demo_trace(model, scodec, acodec, ex, device) -> dict:
    """Teacher-forced side-by-side trace for the interactive demo.

    At every TRUE step ``s_t -> s_{t+1}`` we read the net two ways from the true
    current state: ``pure-net`` (every field decoded by the net, incl. the digit
    payload) and ``neurosym`` (the net's structure with arithmetic offloaded to the
    ALU). Each is compared to ground truth. This matches the rigorous single-step
    metric and makes the magnitude wall visible step by step: as inputs grow OOD the
    pure-net column turns red while the neurosym column stays green.

    Returns ``{reg_names, init, steps, summary}`` where each step carries the
    instruction, the ground-truth/pure-net/neurosym register views, and exact flags.
    """
    model.eval()
    program = ex.trace.program
    gt = ex.trace.states
    reg_names = list(scodec.reg_names)
    steps = []
    pure_ok = neuro_ok = 0
    for t in range(len(ex.trace.actions)):
        cur = gt[t]
        if cur.halted or not (0 <= cur.pc < len(program)):
            break
        instr = program[cur.pc]
        true_next = gt[t + 1]
        s_t = _batch1(scodec.encode(cur).as_dict(), device)
        a_t = _batch1(acodec.encode(instr).as_dict(), device)
        logits = model.heads(model.dynamics(model.encode(s_t), model.action(a_t)))

        pure = _decode_net_state(logits, scodec)
        try:
            alu = vmmod.step(cur, instr)
        except Exception:  # noqa: BLE001
            alu = cur.copy(); alu.error = True; alu.halted = True
        neuro = alu.copy()
        neuro.pc = int(logits["pc"].argmax(-1).item())
        neuro.halted = bool(logits["halted"].argmax(-1).item())
        neuro.error = bool(logits["error"].argmax(-1).item())

        enc_true = scodec.encode(true_next)
        pure_exact = scodec.exact_match(scodec.encode(pure), enc_true)
        neuro_exact = scodec.exact_match(scodec.encode(neuro), enc_true)
        pure_ok += int(pure_exact)
        neuro_ok += int(neuro_exact)
        steps.append({
            "t": t, "pc": int(cur.pc), "instr": _instr_str(instr),
            "dst": instr.dst,
            "ground_truth": _regs_view(true_next, reg_names),
            "pure_net": _regs_view(pure, reg_names),
            "neurosym": _regs_view(neuro, reg_names),
            "pure_exact": bool(pure_exact), "neurosym_exact": bool(neuro_exact),
        })
    n = len(steps)
    return {
        "reg_names": reg_names,
        "init": _regs_view(gt[0], reg_names),
        "steps": steps,
        "summary": {
            "n_steps": n,
            "pure_net_exact_frac": pure_ok / n if n else float("nan"),
            "neurosym_exact_frac": neuro_ok / n if n else float("nan"),
            "max_abs_value": max((abs(v) for s in gt for v in s.regs.values()
                                  if isinstance(v, int)), default=0),
        },
    }
