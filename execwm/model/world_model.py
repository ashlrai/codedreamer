"""Grounded latent world model of computation (M1) — slotted-latent design.

The latent is **not** a single pooled vector but one vector per state *slot*:
one per register, one per heap cell, plus a program-counter slot and a flags
slot. This matters: packing ~30 registers' exact integer values into one vector
and decoding them with a single linear is information-bottlenecked (we measured
it plateauing far below exact match). With a per-slot latent, an unchanged
register is a near-identity copy and each slot is decoded from its own vector by
a *shallow shared* head — which keeps the JEPA efficiency and the
Othello-GPT-style interpretability claim while making exact decode tractable.

Flow:
* ``StateEncoder`` — embed each slot (register/heap/pc/flags) as a token and run
  a small Transformer; return the per-slot hidden states ``z`` of shape (B, S, d).
* ``ActionEncoder`` — embed the executed instruction into one vector.
* ``LatentDynamics`` — a Transformer over the S slots with the action injected
  into every slot; deterministic (transitions are deterministic), residual to z.
* ``GroundingHeads`` — shallow per-slot linear decoders ``z -> symbolic state``.

Losses: grounded decode at t and t+1, JEPA feature-prediction vs an EMA target
encoder (VICReg-regularized), and a curriculum rollout that unrolls the dynamics
in latent space and keeps the decoded state exact (the plannability thesis).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.action_codec import ALL_OPS, OPK_IMM, OPK_REG
from ..data.state_codec import CodecConfig
from ..substrate.vm import VType


@dataclass
class ModelConfig:
    num_regs: int
    num_cells: int
    max_digits: int
    base: int
    max_pc: int
    num_lists: int
    d_model: int = 256
    n_heads: int = 4
    enc_layers: int = 3
    dyn_layers: int = 3
    ffn_mult: int = 4
    dropout: float = 0.0

    @property
    def num_slots(self) -> int:
        return self.num_regs + self.num_cells + 2  # +pc +flags

    @classmethod
    def from_codec(cls, num_regs: int, num_cells: int, num_lists: int,
                   codec: CodecConfig, **kw) -> "ModelConfig":
        return cls(num_regs=num_regs, num_cells=num_cells, num_lists=num_lists,
                   max_digits=codec.max_digits, base=codec.base,
                   max_pc=codec.max_pc, **kw)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class ValueEmbedding(nn.Module):
    """Embed a signed integer given as (sign, MSB-first digits) compositionally:
    a shared per-digit-value table plus a per-position table summed over digits,
    plus a sign embedding. Compositional digits help the magnitude OOD axis."""

    def __init__(self, base: int, max_digits: int, d: int) -> None:
        super().__init__()
        self.digit = nn.Embedding(base, d)
        self.pos = nn.Embedding(max_digits, d)
        self.sign = nn.Embedding(2, d)
        self.register_buffer("pos_idx", torch.arange(max_digits), persistent=False)

    def forward(self, sign: torch.Tensor, digits: torch.Tensor) -> torch.Tensor:
        d_emb = self.digit(digits) + self.pos(self.pos_idx)  # (..., D, d)
        return d_emb.sum(dim=-2) + self.sign(sign)           # (..., d)


# slot token-type ids
_TT_REG, _TT_HEAP, _TT_PC, _TT_FLAGS = 0, 1, 2, 3


def reg_dev(s: dict[str, torch.Tensor]) -> torch.device:
    return s["reg_type"].device


class StateEncoder(nn.Module):
    """Encode a state into one latent vector per slot: (B, S, d)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.value = ValueEmbedding(cfg.base, cfg.max_digits, d)
        self.reg_pos = nn.Embedding(cfg.num_regs, d)
        self.reg_type = nn.Embedding(len(VType), d)
        self.heap_pos = nn.Embedding(cfg.num_cells, d)
        self.pc_emb = nn.Embedding(cfg.max_pc + 1, d)
        self.halted_emb = nn.Embedding(2, d)
        self.error_emb = nn.Embedding(2, d)
        self.toktype = nn.Embedding(4, d)
        layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, dim_feedforward=d * cfg.ffn_mult,
            dropout=cfg.dropout, batch_first=True, activation="gelu",
            norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, cfg.enc_layers)

    def forward(self, s: dict[str, torch.Tensor]) -> torch.Tensor:
        dev = reg_dev(s)
        tt = lambda i: self.toktype(torch.full((1,), i, device=dev))
        reg = (self.reg_pos.weight[None]                       # (1,R,d)
               + self.reg_type(s["reg_type"])                  # (B,R,d)
               + self.value(s["reg_sign"], s["reg_digits"])    # (B,R,d)
               + tt(_TT_REG))
        heap = (self.heap_pos.weight[None]
                + self.value(s["heap_sign"], s["heap_digits"])
                + tt(_TT_HEAP))
        pc = self.pc_emb(s["pc"])[:, None] + tt(_TT_PC)        # (B,1,d)
        flags = (self.halted_emb(s["halted"])[:, None]
                 + self.error_emb(s["error"])[:, None] + tt(_TT_FLAGS))
        tokens = torch.cat([reg, heap, pc, flags], dim=1)      # (B,S,d)
        return self.transformer(tokens)                        # per-slot latent


class ActionEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.op = nn.Embedding(len(ALL_OPS), d)
        self.dst = nn.Embedding(cfg.num_regs + 1, d)          # +1 none sentinel
        self.kind = nn.Embedding(3, d)
        self.reg = nn.Embedding(cfg.num_regs + 1, d)
        self.value = ValueEmbedding(cfg.base, cfg.max_digits, d)
        self.list_id = nn.Embedding(cfg.num_lists + 1, d)
        self.target = nn.Embedding(cfg.max_pc + 1, d)
        self.mlp = nn.Sequential(nn.Linear(d, d * cfg.ffn_mult), nn.GELU(),
                                 nn.Linear(d * cfg.ffn_mult, d))

    def _operand(self, kind, reg, sign, digits) -> torch.Tensor:
        is_reg = (kind == OPK_REG).unsqueeze(-1)
        is_imm = (kind == OPK_IMM).unsqueeze(-1)
        return (self.kind(kind) + self.reg(reg) * is_reg
                + self.value(sign, digits) * is_imm)

    def forward(self, a: dict[str, torch.Tensor]) -> torch.Tensor:
        h = (self.op(a["op"]) + self.dst(a["dst"]) + self.list_id(a["list_id"])
             + self.target(a["target"])
             + self._operand(a["a_kind"], a["a_reg"], a["a_sign"], a["a_digits"])
             + self._operand(a["b_kind"], a["b_reg"], a["b_sign"], a["b_digits"]))
        return h + self.mlp(h)


class LatentDynamics(nn.Module):
    """Deterministic slot-transformer: inject the action into every slot, attend
    across slots, residual to the input latent. ẑ_{t+1} = z + g(z, a)."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.action_proj = nn.Linear(d, d)
        layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, dim_feedforward=d * cfg.ffn_mult,
            dropout=cfg.dropout, batch_first=True, activation="gelu",
            norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, cfg.dyn_layers)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # z: (N, S, d)   a: (N, d)
        h = z + self.action_proj(a).unsqueeze(1)
        return self.out_norm(z + self.transformer(h))


class GroundingHeads(nn.Module):
    """Shallow per-slot linear decoders (the anchor + the interpretability claim).
    Register/heap heads are *shared* across their slots, applied per-slot."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d, D, base = cfg.d_model, cfg.max_digits, cfg.base
        self.cfg = cfg
        self.reg_type = nn.Linear(d, len(VType))
        self.reg_sign = nn.Linear(d, 2)
        self.reg_digits = nn.Linear(d, D * base)
        self.heap_sign = nn.Linear(d, 2)
        self.heap_digits = nn.Linear(d, D * base)
        self.pc = nn.Linear(d, cfg.max_pc + 1)
        self.halted = nn.Linear(d, 2)
        self.error = nn.Linear(d, 2)

    def forward(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        N = z.shape[0]
        R, C, D, base = cfg.num_regs, cfg.num_cells, cfg.max_digits, cfg.base
        reg = z[:, :R]                       # (N,R,d)
        heap = z[:, R:R + C]                  # (N,C,d)
        pc_slot = z[:, R + C]                 # (N,d)
        flags_slot = z[:, R + C + 1]          # (N,d)
        return {
            "reg_type": self.reg_type(reg),
            "reg_sign": self.reg_sign(reg),
            "reg_digits": self.reg_digits(reg).view(N, R, D, base),
            "heap_sign": self.heap_sign(heap),
            "heap_digits": self.heap_digits(heap).view(N, C, D, base),
            "pc": self.pc(pc_slot),
            "halted": self.halted(flags_slot),
            "error": self.error(flags_slot),
        }


# ---------------------------------------------------------------------------
# Losses / metrics over the grounding heads
# ---------------------------------------------------------------------------


def valued_mask(reg_type: torch.Tensor) -> torch.Tensor:
    """Mask of registers whose type is INT/BOOL (payload meaningful). Explicit
    comparisons (not torch.isin) to stay MPS-compatible."""
    return (reg_type == VType.INT.value) | (reg_type == VType.BOOL.value)


def grounding_loss(logits: dict, tgt: dict) -> torch.Tensor:
    """Sum of cross-entropies over all state fields. Register sign/digit losses
    are masked to registers whose target type is valued (UNDEF payload is junk)."""
    mask = valued_mask(tgt["reg_type"]).float()  # (B,R)

    def ce(l, t):
        return F.cross_entropy(l.reshape(-1, l.shape[-1]), t.reshape(-1))

    def ce_masked(l, t, m):
        loss = F.cross_entropy(l.reshape(-1, l.shape[-1]), t.reshape(-1),
                               reduction="none").view(t.shape)
        while m.dim() < loss.dim():
            m = m.unsqueeze(-1)
        m = m.expand_as(loss)
        return (loss * m).sum() / m.sum().clamp_min(1.0)

    total = ce(logits["reg_type"], tgt["reg_type"])
    total = total + ce_masked(logits["reg_sign"], tgt["reg_sign"], mask)
    total = total + ce_masked(logits["reg_digits"], tgt["reg_digits"], mask)
    total = total + ce(logits["heap_sign"], tgt["heap_sign"])
    total = total + ce(logits["heap_digits"], tgt["heap_digits"])
    total = total + ce(logits["pc"], tgt["pc"])
    total = total + ce(logits["halted"], tgt["halted"])
    total = total + ce(logits["error"], tgt["error"])
    return total


@torch.no_grad()
def field_correct(logits: dict, tgt: dict) -> dict[str, torch.Tensor]:
    """Per-sample boolean correctness of each field group (for diagnostics)."""
    pred = {k: v.argmax(-1) for k, v in logits.items()}
    mask = valued_mask(tgt["reg_type"])
    reg_ok = (pred["reg_type"] == tgt["reg_type"])
    sign_ok = (pred["reg_sign"] == tgt["reg_sign"]) | ~mask
    dig_ok = (pred["reg_digits"] == tgt["reg_digits"]).all(-1) | ~mask
    return {
        "reg": (reg_ok & sign_ok & dig_ok).all(1),
        "heap": ((pred["heap_sign"] == tgt["heap_sign"]).all(1)
                 & (pred["heap_digits"] == tgt["heap_digits"]).all((1, 2))),
        "pc": pred["pc"] == tgt["pc"],
        "flags": (pred["halted"] == tgt["halted"]) & (pred["error"] == tgt["error"]),
    }


@torch.no_grad()
def exact_match(logits: dict, tgt: dict) -> torch.Tensor:
    """Per-sample exact state match (bool, (B,)) mirroring the codec rule."""
    fc = field_correct(logits, tgt)
    return fc["reg"] & fc["heap"] & fc["pc"] & fc["flags"]


@torch.no_grad()
def per_var_accuracy(logits: dict, tgt: dict) -> torch.Tensor:
    """Mean per-register correctness (type+sign+digits) over valued registers."""
    pred = {k: v.argmax(-1) for k, v in logits.items()}
    mask = valued_mask(tgt["reg_type"])
    correct = ((pred["reg_type"] == tgt["reg_type"])
               & (pred["reg_sign"] == tgt["reg_sign"])
               & (pred["reg_digits"] == tgt["reg_digits"]).all(-1)) & mask
    return correct.sum().float() / mask.sum().clamp_min(1).float()


def vicreg(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Variance + covariance regularizers (invariance handled by the JEPA loss).
    Expects a 2-D (N, d) batch."""
    z = z - z.mean(0, keepdim=True)
    std = torch.sqrt(z.var(0) + eps)
    var_loss = F.relu(1.0 - std).mean()
    n, d = z.shape
    cov = (z.T @ z) / max(n - 1, 1)
    off = cov - torch.diag(torch.diag(cov))
    return var_loss + (off ** 2).sum() / d


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class GroundedLatentWM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = StateEncoder(cfg)
        self.action = ActionEncoder(cfg)
        self.dynamics = LatentDynamics(cfg)
        self.heads = GroundingHeads(cfg)
        self.target_encoder = StateEncoder(cfg)   # EMA target for JEPA (no grad)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_target(self, momentum: float = 0.996) -> None:
        for tp, p in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            tp.mul_(momentum).add_(p, alpha=1 - momentum)

    def encode(self, s: dict) -> torch.Tensor:
        return self.encoder(s)

    def predict_next(self, z: torch.Tensor, a: dict) -> torch.Tensor:
        return self.dynamics(z, self.action(a))
