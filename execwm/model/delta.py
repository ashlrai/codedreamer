"""M1.6 — copy-vs-compute (delta) world model.

The M1 grounding heads re-predict the *entire* next state from the predicted
latent every step. But each VM step changes exactly one slot, so re-predicting all
~30 slots wastes capacity and lets any per-slot error compound over a rollout.

``DeltaWM`` instead predicts, per slot, a **change gate** ("did this slot change?")
plus a **new value** (computed only where it changed); the predicted next state
*copies* unchanged slots from the current state and writes the computed value only
where the gate fires. Consequences:
* unchanged slots are exact by construction → exact-match jumps,
* in a rollout, a wrong value stays local (copied forward) instead of corrupting
  the whole state → horizon extends.

The value path gets a small per-slot trunk (it must do arithmetic); the gate is a
shallow per-slot linear. Interpretability is still served by the separate frozen
linear probes in ``execwm/eval/probes.py``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..substrate.vm import VType
from .world_model import GroundedLatentWM, ModelConfig, valued_mask


class DeltaHeads(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d, D, base = cfg.d_model, cfg.max_digits, cfg.base
        self.cfg = cfg
        self.gate = nn.Linear(d, 2)                       # per-slot changed? (shared)
        self.reg_trunk = nn.Sequential(nn.Linear(d, d), nn.GELU())
        self.heap_trunk = nn.Sequential(nn.Linear(d, d), nn.GELU())
        self.reg_type = nn.Linear(d, len(VType))
        self.reg_sign = nn.Linear(d, 2)
        self.reg_digits = nn.Linear(d, D * base)
        self.heap_sign = nn.Linear(d, 2)
        self.heap_digits = nn.Linear(d, D * base)
        self.pc = nn.Linear(d, cfg.max_pc + 1)
        self.halted = nn.Linear(d, 2)
        self.error = nn.Linear(d, 2)

    def forward(self, z: torch.Tensor):
        cfg = self.cfg
        N = z.shape[0]
        R, C, D, base = cfg.num_regs, cfg.num_cells, cfg.max_digits, cfg.base
        reg, heap = z[:, :R], z[:, R:R + C]
        pc_slot, flags_slot = z[:, R + C], z[:, R + C + 1]
        gate = self.gate(z)                               # (N, S, 2)
        rt, ht = self.reg_trunk(reg), self.heap_trunk(heap)
        value = {
            "reg_type": self.reg_type(rt),
            "reg_sign": self.reg_sign(rt),
            "reg_digits": self.reg_digits(rt).view(N, R, D, base),
            "heap_sign": self.heap_sign(ht),
            "heap_digits": self.heap_digits(ht).view(N, C, D, base),
            "pc": self.pc(pc_slot),
            "halted": self.halted(flags_slot),
            "error": self.error(flags_slot),
        }
        return gate, value


class DeltaWM(GroundedLatentWM):
    """Reuses the slotted encoder/action/dynamics/target-encoder; adds delta heads."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.delta = DeltaHeads(cfg)


# ---------------------------------------------------------------------------
# Changed-slot masks, loss, composition, exact-match (all over label dicts)
# ---------------------------------------------------------------------------


def changed_masks(cur: dict, nxt: dict, cfg: ModelConfig):
    """Per-slot 'did it change?' booleans from current vs next label dicts.
    Returns (slot_changed (N,S), reg (N,R), heap (N,C), pc (N,), flags (N,))."""
    valued_cur = valued_mask(cur["reg_type"])
    valued_nxt = valued_mask(nxt["reg_type"])
    reg = cur["reg_type"] != nxt["reg_type"]
    val_diff = ((cur["reg_sign"] != nxt["reg_sign"])
                | (cur["reg_digits"] != nxt["reg_digits"]).any(-1))
    reg = reg | (val_diff & valued_cur & valued_nxt)
    heap = ((cur["heap_sign"] != nxt["heap_sign"])
            | (cur["heap_digits"] != nxt["heap_digits"]).any(-1))
    pc = cur["pc"] != nxt["pc"]
    flags = (cur["halted"] != nxt["halted"]) | (cur["error"] != nxt["error"])
    slot = torch.cat([reg, heap, pc[:, None], flags[:, None]], dim=1)
    return slot, reg, heap, pc, flags


def _masked_ce(logits: torch.Tensor, tgt: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                           reduction="none").view(tgt.shape)
    while m.dim() < loss.dim():
        m = m.unsqueeze(-1)
    m = m.expand_as(loss)
    return (loss * m).sum() / m.sum().clamp_min(1.0)


def delta_loss(gate: torch.Tensor, value: dict, cur: dict, nxt: dict,
               cfg: ModelConfig):
    """Gate cross-entropy (every slot) + value cross-entropy (only on slots that
    actually changed — the 'compute' path; unchanged slots are copied, not learned).
    pc and flags are cheap single slots, supervised always."""
    slot_changed, reg_ch, heap_ch, pc_ch, flags_ch = changed_masks(cur, nxt, cfg)
    gate_loss = F.cross_entropy(gate.reshape(-1, 2), slot_changed.long().reshape(-1))
    valn = valued_mask(nxt["reg_type"])
    v = _masked_ce(value["reg_type"], nxt["reg_type"], reg_ch)
    v = v + _masked_ce(value["reg_sign"], nxt["reg_sign"], reg_ch & valn)
    v = v + _masked_ce(value["reg_digits"], nxt["reg_digits"], reg_ch & valn)
    v = v + _masked_ce(value["heap_sign"], nxt["heap_sign"], heap_ch)
    v = v + _masked_ce(value["heap_digits"], nxt["heap_digits"], heap_ch)
    v = v + F.cross_entropy(value["pc"], nxt["pc"])
    v = v + F.cross_entropy(value["halted"], nxt["halted"])
    v = v + F.cross_entropy(value["error"], nxt["error"])
    return gate_loss + v, gate_loss.detach(), v.detach()


@torch.no_grad()
def compose_next(gate: torch.Tensor, value: dict, cur: dict,
                 cfg: ModelConfig) -> dict:
    """Predicted next labels: copy ``cur`` where the gate says unchanged, else the
    argmax of the value head."""
    R, C = cfg.num_regs, cfg.num_cells
    g = gate.argmax(-1)
    greg, gheap = g[:, :R].bool(), g[:, R:R + C].bool()
    gpc, gfl = g[:, R + C].bool(), g[:, R + C + 1].bool()
    return {
        "reg_type": torch.where(greg, value["reg_type"].argmax(-1), cur["reg_type"]),
        "reg_sign": torch.where(greg, value["reg_sign"].argmax(-1), cur["reg_sign"]),
        "reg_digits": torch.where(greg.unsqueeze(-1),
                                  value["reg_digits"].argmax(-1), cur["reg_digits"]),
        "heap_sign": torch.where(gheap, value["heap_sign"].argmax(-1), cur["heap_sign"]),
        "heap_digits": torch.where(gheap.unsqueeze(-1),
                                   value["heap_digits"].argmax(-1), cur["heap_digits"]),
        "pc": torch.where(gpc, value["pc"].argmax(-1), cur["pc"]),
        "halted": torch.where(gfl, value["halted"].argmax(-1), cur["halted"]),
        "error": torch.where(gfl, value["error"].argmax(-1), cur["error"]),
    }


@torch.no_grad()
def exact_match_labels(pred: dict, tgt: dict) -> torch.Tensor:
    """Per-sample exact state match over predicted vs target label dicts."""
    mask = valued_mask(tgt["reg_type"])
    ok = torch.ones(tgt["pc"].shape[0], dtype=torch.bool, device=tgt["pc"].device)
    ok &= pred["pc"] == tgt["pc"]
    ok &= pred["halted"] == tgt["halted"]
    ok &= pred["error"] == tgt["error"]
    ok &= (pred["reg_type"] == tgt["reg_type"]).all(1)
    ok &= (pred["heap_sign"] == tgt["heap_sign"]).all(1)
    ok &= (pred["heap_digits"] == tgt["heap_digits"]).all((1, 2))
    ok &= ((pred["reg_sign"] == tgt["reg_sign"]) | ~mask).all(1)
    ok &= ((pred["reg_digits"] == tgt["reg_digits"]).all(-1) | ~mask).all(1)
    return ok


@torch.no_grad()
def gate_accuracy(gate: torch.Tensor, cur: dict, nxt: dict, cfg: ModelConfig):
    """Diagnostics: fraction of slots whose change-gate is predicted correctly,
    plus recall on the (rare) actually-changed slots."""
    slot_changed, *_ = changed_masks(cur, nxt, cfg)
    pred = gate.argmax(-1).bool()
    acc = (pred == slot_changed).float().mean()
    changed = slot_changed
    recall = (pred & changed).sum().float() / changed.sum().clamp_min(1).float()
    return acc.item(), recall.item()
