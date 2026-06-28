"""Read-only diagnosis of WHERE and WHY latent multi-step rollout breaks down.

Loads the already-trained `artifacts/latent_easy.pt` and, on CPU, measures how
the grounded latent world model compounds error when rolled forward in latent
space. This is the evidence base for designing a "divergence head".

Two rollout regimes are compared (both start from encode(s_0), the TRUE state):

  * teacher-forced (TF):   feed the TRUE instruction sequence at each step; the
                           latent is never re-grounded. Measures pure DYNAMICS
                           compounding along the correct program path.
  * autoregressive (AR):   decode the predicted pc, FETCH program[pc], encode that
                           instruction, predict_next. Control flow is now the
                           model's own job, so a single wrong pc derails the rest.
                           This is the regime the WM-as-scorer actually runs in.

Analyses
  1. per-horizon exact-match + per-field accuracy curves (TF and AR)
  2. which FIELD (reg / pc / heap / flags) drops below 0.5 first
  3. single-step error rate bucketed by the executed Op (what arithmetic/control
     op the one-step predictor gets wrong)
  4. straight-line episodes vs episodes containing a backward jump (loop)

Everything is short, foreground, CPU-only. No training, no background jobs.

    PYTHONPATH=. python3 scripts/rollout_analysis.py [--n 400] [--horizon 64]
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from execwm.data.dataset import collect_examples
from execwm.data.torch_data import _ACTION_KEYS, _STATE_KEYS
from execwm.eval.checkpoint import load_checkpoint
from execwm.model.world_model import exact_match, field_correct
from execwm.substrate.vm import JUMP_OPS, Op

FIELDS = ("reg", "pc", "heap", "flags")


# ---------------------------------------------------------------------------
# tensor helpers
# ---------------------------------------------------------------------------

def stack_to_tensor(dicts: list[dict], keys) -> dict[str, torch.Tensor]:
    """Stack a list of `as_dict()` numpy dicts into a batched long-tensor dict."""
    out = {}
    for k in keys:
        out[k] = torch.from_numpy(np.stack([d[k] for d in dicts])).long()
    return out


def encode_states_batch(model, state_asdicts: list[dict]) -> torch.Tensor:
    s = stack_to_tensor(state_asdicts, _STATE_KEYS)
    return model.encode(s)                       # (B, S, d)


def step_latent(model, z: torch.Tensor, action_asdicts: list[dict]) -> torch.Tensor:
    a = stack_to_tensor(action_asdicts, _ACTION_KEYS)
    return model.dynamics(z, model.action(a))    # (B, S, d)


def correctness(model, z: torch.Tensor, tgt_asdicts: list[dict]):
    """Return (exact_match bool (B,), field_correct dict of bool (B,))."""
    logits = model.heads(z)
    tgt = stack_to_tensor(tgt_asdicts, _STATE_KEYS)
    em = exact_match(logits, tgt)
    fc = field_correct(logits, tgt)
    return em, fc


# ---------------------------------------------------------------------------
# episode pre-encoding
# ---------------------------------------------------------------------------

class Episode:
    __slots__ = ("states_enc", "actions_enc", "ops", "length", "program",
                 "has_loop")

    def __init__(self, ex, scodec, acodec, horizon):
        states = ex.trace.states
        actions = ex.trace.actions
        L = len(actions)
        keep = min(L, horizon)
        # encode states[0 .. keep] and actions[0 .. keep-1]
        self.states_enc = [scodec.encode(states[t]).as_dict()
                           for t in range(keep + 1)]
        self.actions_enc = [acodec.encode(actions[t]).as_dict()
                            for t in range(keep)]
        self.ops = [actions[t].op for t in range(keep)]
        self.length = keep
        # bytecode program (Instr list); fetch by predicted pc in AR rollout
        self.program = ex.bytecode
        # loop = a backward jump (target <= pc of the jump) anywhere in the trace
        self.has_loop = any(
            a.op in JUMP_OPS and a.target is not None
            and a.target <= states[t].pc
            for t, a in enumerate(actions))


# ---------------------------------------------------------------------------
# rollouts
# ---------------------------------------------------------------------------

@torch.no_grad()
def rollout_curves(model, scodec, acodec, eps: list[Episode], horizon: int,
                   regime: str):
    """Return per-horizon dicts: em[h], field[field][h] = accuracy over episodes
    whose TRUE trace reaches horizon h. regime in {'tf', 'ar'}."""
    B = len(eps)
    z = encode_states_batch(model, [e.states_enc[0] for e in eps])  # (B,S,d)
    none_target = acodec.none_target

    # per-horizon accumulators
    em_hit = np.zeros(horizon); em_tot = np.zeros(horizon)
    f_hit = {f: np.zeros(horizon) for f in FIELDS}

    done = np.zeros(B, dtype=bool)          # AR: model walked off the program
    halt_dummy = acodec.encode(_halt_instr()).as_dict()

    for h in range(1, horizon + 1):
        # ---- choose the action for this step (uses z = z_{h-1}) ----
        if regime == "tf":
            act = []
            active = np.zeros(B, dtype=bool)
            for b, e in enumerate(eps):
                if e.length >= h:
                    act.append(e.actions_enc[h - 1])
                    active[b] = True
                else:
                    act.append(halt_dummy)
            z_step = step_latent(model, z, act)
            keep = torch.from_numpy(active)            # advance only true-active
        else:  # autoregressive: decode pc -> fetch program[pc]
            pc_pred = model.heads(z)["pc"].argmax(-1).cpu().numpy()  # (B,)
            act = []
            for b, e in enumerate(eps):
                if done[b]:
                    act.append(halt_dummy)
                    continue
                pc = int(pc_pred[b])
                if 0 <= pc < len(e.program):
                    act.append(acodec.encode(e.program[pc]).as_dict())
                else:
                    done[b] = True
                    act.append(halt_dummy)
            z_step = step_latent(model, z, act)
            keep = torch.from_numpy(~done)             # frozen once off-program

        # advance the latent only for episodes still rolling; freeze the rest
        z = torch.where(keep[:, None, None], z_step, z)

        # ---- score vs the TRUE state at horizon h ----
        active_truth = np.array([e.length >= h for e in eps])
        if not active_truth.any():
            break
        tgt = [e.states_enc[h] if e.length >= h else e.states_enc[-1] for e in eps]
        em, fc = correctness(model, z, tgt)
        em = em.cpu().numpy(); fcn = {f: fc[f].cpu().numpy() for f in FIELDS}
        sel = active_truth
        em_hit[h - 1] = em[sel].sum(); em_tot[h - 1] = sel.sum()
        for f in FIELDS:
            f_hit[f][h - 1] = fcn[f][sel].sum()

    em_curve = np.divide(em_hit, em_tot, out=np.full(horizon, np.nan),
                         where=em_tot > 0)
    field_curves = {f: np.divide(f_hit[f], em_tot, out=np.full(horizon, np.nan),
                                 where=em_tot > 0) for f in FIELDS}
    return em_curve, field_curves, em_tot


def _halt_instr():
    from execwm.substrate.vm import Instr
    return Instr(op=Op.HALT)


# ---------------------------------------------------------------------------
# single-step per-op error analysis
# ---------------------------------------------------------------------------

@torch.no_grad()
def per_op_single_step(model, scodec, acodec, eps: list[Episode],
                       chunk: int = 2048):
    """Teacher-forced ONE-step prediction error bucketed by executed Op.
    Returns dict op_name -> {n, em, pc, reg, heap, flags} fractional accuracy."""
    cur, act, nxt, ops = [], [], [], []
    for e in eps:
        for t in range(e.length):
            cur.append(e.states_enc[t])
            act.append(e.actions_enc[t])
            nxt.append(e.states_enc[t + 1])
            ops.append(e.ops[t].name)
    ops = np.array(ops)

    agg = defaultdict(lambda: defaultdict(float))
    for i in range(0, len(cur), chunk):
        sl = slice(i, i + chunk)
        s = stack_to_tensor(cur[sl], _STATE_KEYS)
        a = stack_to_tensor(act[sl], _ACTION_KEYS)
        z = model.dynamics(model.encode(s), model.action(a))
        em, fc = correctness(model, z, nxt[sl])
        em = em.cpu().numpy()
        fcn = {f: fc[f].cpu().numpy() for f in FIELDS}
        for j, op in enumerate(ops[sl]):
            d = agg[op]
            d["n"] += 1
            d["em"] += float(em[j])
            for f in FIELDS:
                d[f] += float(fcn[f][j])
    out = {}
    for op, d in agg.items():
        n = d["n"]
        out[op] = {"n": int(n), "em": d["em"] / n,
                   **{f: d[f] / n for f in FIELDS}}
    return out


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def first_below(curve: np.ndarray, thr: float) -> int | None:
    for k, v in enumerate(curve):
        if not np.isnan(v) and v < thr:
            return k + 1            # 1-indexed horizon
    return None


def print_horizon_table(em_tf, em_ar, fields_ar, em_tot, horizon):
    print("\n## 1+2. Per-horizon accuracy (autoregressive = the scorer regime)")
    print("h    n_eps  EM_tf  EM_ar | reg_ar  pc_ar  heap_ar flags_ar")
    rows = list(range(horizon))
    show = [r for r in rows if em_tot[r] > 0]
    # print every horizon up to 16, then sparsely
    printed = 0
    for r in show:
        h = r + 1
        if h <= 16 or h % 4 == 0 or r == show[-1]:
            print(f"{h:<4d} {int(em_tot[r]):<6d} {em_tf[r]:.3f}  {em_ar[r]:.3f} | "
                  f"{fields_ar['reg'][r]:.3f}   {fields_ar['pc'][r]:.3f}  "
                  f"{fields_ar['heap'][r]:.3f}   {fields_ar['flags'][r]:.3f}")
            printed += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    torch.manual_seed(0)
    device = torch.device("cpu")
    lk = load_checkpoint("artifacts/latent_easy.pt", device=device)
    model, scodec, acodec, spec = lk["model"], lk["scodec"], lk["acodec"], lk["spec"]
    model.eval()

    print(f"[load] latent_easy.pt  regs={scodec.num_regs} cells={scodec.num_cells} "
          f"d_model={model.cfg.d_model}  meta={lk['meta']}")
    print(f"[data] collecting {args.n} terminating episodes (seed={args.seed})...")
    exs, _ = collect_examples(spec, args.n, lambda e: True, args.seed, scodec, acodec)
    eps = [Episode(e, scodec, acodec, args.horizon) for e in exs]
    lens = np.array([e.length for e in eps])
    n_loop = sum(e.has_loop for e in eps)
    print(f"[data] trace length (capped at horizon): mean {lens.mean():.1f} "
          f"median {int(np.median(lens))} max {lens.max()}  | "
          f"{n_loop} have a backward jump (loop), {len(eps) - n_loop} straight-line")

    # ---- analyses 1 + 2 ----
    em_tf, _, em_tot = rollout_curves(model, scodec, acodec, eps, args.horizon, "tf")
    em_ar, fields_ar, _ = rollout_curves(model, scodec, acodec, eps, args.horizon, "ar")
    print_horizon_table(em_tf, em_ar, fields_ar, em_tot, args.horizon)

    print("\n  exact-match crossing thresholds (horizon = #latent steps):")
    for name, c in (("teacher-forced", em_tf), ("autoregressive", em_ar)):
        print(f"    {name:<16s} <0.5 at h={first_below(c, 0.5)}   "
              f"<0.1 at h={first_below(c, 0.1)}")
    print("  per-field <0.5 crossing (autoregressive):")
    for f in FIELDS:
        print(f"    {f:<6s} <0.5 at h={first_below(fields_ar[f], 0.5)}   "
              f"<0.9 at h={first_below(fields_ar[f], 0.9)}")

    # ---- analysis 3 ----
    print("\n## 3. Single-step (teacher-forced) accuracy bucketed by executed Op")
    op_stats = per_op_single_step(model, scodec, acodec, eps)
    print("op       n      EM     reg    pc     heap   flags   1-EM(err)")
    order = sorted(op_stats.items(), key=lambda kv: kv[1]["em"])
    for op, d in order:
        print(f"{op:<8s} {d['n']:<6d} {d['em']:.3f}  {d['reg']:.3f}  {d['pc']:.3f}  "
              f"{d['heap']:.3f}  {d['flags']:.3f}   {1 - d['em']:.3f}")

    # ---- analysis 4 ----
    print("\n## 4. Straight-line vs looping episodes (autoregressive rollout)")
    sl_eps = [e for e in eps if not e.has_loop]
    lp_eps = [e for e in eps if e.has_loop]
    print("group         n_eps  EM@h1  EM@h2  EM@h4  EM@h8  EM@h16  <0.5_at  <0.1_at")
    for name, group in (("straight-line", sl_eps), ("looping", lp_eps)):
        if not group:
            print(f"{name:<13s} (none)")
            continue
        emc, _, tot = rollout_curves(model, scodec, acodec, group, args.horizon, "ar")
        def at(h):
            return f"{emc[h-1]:.3f}" if h - 1 < len(emc) and tot[h-1] > 0 else " -- "
        print(f"{name:<13s} {len(group):<6d} {at(1)}  {at(2)}  {at(4)}  {at(8)}  "
              f"{at(16):<6s}  h={first_below(emc, 0.5)}    h={first_below(emc, 0.1)}")

    print("\n[done] read-only analysis complete; no processes left running.")


if __name__ == "__main__":
    main()
