"""Neurosymbolic readout analysis — the M3.5 "self-imposed wall" experiment.

Thesis: program execution decomposes into **learnable structure/control** (which
slot is written, the operation's type, the sign, the next pc, branch outcomes) and
**offloadable arithmetic** (the exact digit value of a computed register). A grounded
latent world model that is forced to *decode digits* hits a magnitude wall; but the
wall may live entirely in the digit head, not the latent.

We test this with an **eval-time intervention on ONE trained model** (no retraining,
no architecture change), so the only thing that varies is the readout:

* ``em_learned``        — standard whole-state exact-match, every field from the net.
* ``em_digits_oracle``  — same predictions, but the numeric *digit* payload of
  registers/heap is replaced by ground truth (i.e. a perfect symbolic ALU fills the
  values). Everything else (pc, type, sign, flags, which-slot-changed) is still the
  net's job, so this is NOT "just running the VM" — it isolates whether the net's
  *structural* prediction is correct.

Run this on an in-distribution split and a magnitude-OOD split. If structure
generalizes while only digits collapse, ``em_digits_oracle`` stays high where
``em_learned`` falls to ~0 — and the magnitude wall is the digit head, which should
be offloaded to the interpreter's ALU. That is the neurosymbolic case.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.action_codec import ALL_OPS
from ..data.torch_data import (EpisodeDataset, _ACTION_KEYS, _STATE_KEYS,
                               collate_episodes, flatten_time)
from ..model.world_model import exact_match, valued_mask
from ..substrate.vm import Op

_ARITH = {Op.ADD, Op.SUB, Op.MUL, Op.DIV, Op.MOD}
_CMP = {Op.LT, Op.LE, Op.EQ, Op.NE, Op.GT, Op.GE}
_JUMP = {Op.JMP, Op.JZ, Op.JNZ}
_ARITH_IDX = torch.tensor([i for i, op in enumerate(ALL_OPS) if op in _ARITH])
_CMP_IDX = torch.tensor([i for i, op in enumerate(ALL_OPS) if op in _CMP])
_JUMP_IDX = torch.tensor([i for i, op in enumerate(ALL_OPS) if op in _JUMP])


def _oracle_digit_logits(logits: dict, tgt: dict, base: int) -> dict:
    """Return a copy of ``logits`` whose reg/heap digit logits argmax to the true
    digits (one-hot of the target) — i.e. a perfect ALU supplies the numbers."""
    out = dict(logits)
    out["reg_digits"] = F.one_hot(tgt["reg_digits"], base).float()
    out["heap_digits"] = F.one_hot(tgt["heap_digits"], base).float()
    return out


class _Acc:
    """Running num/den accumulator keyed by name."""

    def __init__(self) -> None:
        self.n: dict[str, float] = {}
        self.d: dict[str, float] = {}

    def add(self, key: str, num: float, den: float) -> None:
        self.n[key] = self.n.get(key, 0.0) + float(num)
        self.d[key] = self.d.get(key, 0.0) + float(den)

    def ratio(self, key: str) -> float:
        d = self.d.get(key, 0.0)
        return self.n.get(key, 0.0) / d if d else float("nan")


@torch.no_grad()
def field_breakdown(model, examples, scodec, acodec, device, *,
                    max_len: int = 24, batch_size: int = 64) -> dict:
    """Single-step per-field breakdown + the learned-vs-digit-oracle exact-match.

    Returns a dict of scalar metrics; all accuracies are over valid transitions.
    """
    model.eval()
    ds = EpisodeDataset(examples, scodec, acodec, max_len=max_len)
    base = model.cfg.base
    num_regs = model.cfg.num_regs
    acc = _Acc()
    if len(ds) == 0:
        return {"n": 0}
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_episodes)
    arith_idx = _ARITH_IDX.to(device)
    cmp_idx = _CMP_IDX.to(device)
    jump_idx = _JUMP_IDX.to(device)

    def _in(values, idx):  # MPS-safe membership (torch.isin is unsupported on MPS)
        return (values.unsqueeze(-1) == idx).any(-1)

    for batch in loader:
        valid = batch["valid"].to(device)
        B, L = valid.shape
        sel = valid.reshape(B * L)
        if not sel.any():
            continue
        cur = {k: batch[f"s_{k}"][:, :L].to(device) for k in _STATE_KEYS}
        cur_flat, _, _ = flatten_time(cur)
        z = model.encode(cur_flat)
        act = {k: batch[f"a_{k}"][:, :L].to(device) for k in _ACTION_KEYS}
        act_flat, _, _ = flatten_time(act)
        zn = model.dynamics(z, model.action(act_flat))
        logits = model.heads(zn)
        tgt = {k: batch[f"s_{k}"][:, 1:L + 1].to(device).reshape(
                   -1, *batch[f"s_{k}"].shape[2:]) for k in _STATE_KEYS}

        logits = {k: v[sel] for k, v in logits.items()}
        tgt = {k: v[sel] for k, v in tgt.items()}
        op = act_flat["op"][sel]                        # (N,)
        dst = act_flat["dst"][sel]                       # (N,) reg idx or none(=num_regs)
        N = int(sel.sum().item())

        # --- whole-state exact match: learned vs digit-oracle ---
        acc.add("em_learned", exact_match(logits, tgt).sum().item(), N)
        oracle = _oracle_digit_logits(logits, tgt, base)
        acc.add("em_digits_oracle", exact_match(oracle, tgt).sum().item(), N)

        pred = {k: v.argmax(-1) for k, v in logits.items()}
        # --- whole-state field accuracies ---
        acc.add("pc", (pred["pc"] == tgt["pc"]).sum().item(), N)
        acc.add("flags", ((pred["halted"] == tgt["halted"])
                          & (pred["error"] == tgt["error"])).sum().item(), N)

        # --- written-register breakdown (where the op deposits its result) ---
        has_dst = dst < num_regs                          # (N,)
        if has_dst.any():
            d_idx = dst[has_dst].clamp(max=num_regs - 1)  # (M,)
            rows = torch.arange(N, device=device)[has_dst]
            def at(field):  # gather the dst register's field for each row
                return field[rows, d_idx]
            w_type = at(pred["reg_type"]) == at(tgt["reg_type"])
            w_sign = at(pred["reg_sign"]) == at(tgt["reg_sign"])
            w_dig = (pred["reg_digits"][rows, d_idx] == tgt["reg_digits"][rows, d_idx]).all(-1)
            w_op = op[has_dst]
            acc.add("written_type", w_type.sum().item(), w_type.numel())
            acc.add("written_sign", w_sign.sum().item(), w_sign.numel())
            acc.add("written_digits", w_dig.sum().item(), w_dig.numel())
            acc.add("written_full", (w_type & w_sign & w_dig).sum().item(), w_type.numel())
            # split digit accuracy by op family
            am = _in(w_op, arith_idx)
            cm = _in(w_op, cmp_idx)
            if am.any():
                acc.add("arith_digits", w_dig[am].sum().item(), int(am.sum()))
                acc.add("arith_sign", w_sign[am].sum().item(), int(am.sum()))
            if cm.any():
                # comparison result is a 0/1 BOOL: "digits" == the boolean outcome
                acc.add("cmp_result", (w_dig[cm] & w_type[cm]).sum().item(), int(cm.sum()))

        # --- branch correctness: pc accuracy on jump steps specifically ---
        jm = _in(op, jump_idx)
        if jm.any():
            acc.add("branch_pc", (pred["pc"][jm] == tgt["pc"][jm]).sum().item(), int(jm.sum()))

    keys = ["em_learned", "em_digits_oracle", "pc", "flags", "written_type",
            "written_sign", "written_digits", "written_full", "arith_digits",
            "arith_sign", "cmp_result", "branch_pc"]
    out = {k: acc.ratio(k) for k in keys}
    out["n"] = int(acc.d.get("em_learned", 0))
    return out
