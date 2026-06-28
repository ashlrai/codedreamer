"""M1.6b — carry-aware arithmetic value head.

The M1 digit head predicts each digit independently from a fixed-width linear map
(MSB-first). The arithmetic literature (Abacus embeddings, Learning-to-Execute,
Nogueira'21, Lee'23) is unanimous that this is the worst case: carries flow from
the least-significant digit upward, so digits must be (a) emitted LSB-first and
(b) decoded *conditioned on the lower digits already produced*, with the position
encoded as an input (weight-shared across positions) rather than baked into
separate output weights.

``ArithDigitHead`` implements exactly that: a GRU over digit positions, LSB-first,
input-injecting the slot vector and a significance embedding at every step, with
the previous digit fed back (teacher-forced in training, greedy at inference).
Output logits are flipped back to MSB-first so all existing loss / exact-match /
codec code is unchanged. This attacks single-step arithmetic error, which the M1.5
experiment proved is the binding constraint on rollout horizon.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..substrate.vm import VType
from .world_model import GroundedLatentWM, ModelConfig


class ArithDigitHead(nn.Module):
    """Carry-aware per-slot digit decoder. Operates on a flat batch of slot
    vectors (N, d) and emits digit logits (N, D, base) in MSB-first order."""

    def __init__(self, d: int, base: int, max_digits: int, hidden: int | None = None) -> None:
        super().__init__()
        h = hidden or d
        self.base, self.D, self.H = base, max_digits, h
        self.proj = nn.Linear(d, h)
        self.pos = nn.Embedding(max_digits, h)          # significance (0 = LSB)
        self.digit_emb = nn.Embedding(base + 1, h)      # +1 = start token (index base)
        self.gru = nn.GRU(h, h, batch_first=True)       # optimized vs a manual cell loop
        self.out = nn.Linear(h, base)
        self.start = base
        self.register_buffer("_pos_idx", torch.arange(max_digits), persistent=False)

    def forward(self, slot: torch.Tensor, teacher_digits_msb: torch.Tensor | None = None):
        N = slot.shape[0]
        s = self.proj(slot)                              # (N, H) — injected each step
        h0 = s.unsqueeze(0).contiguous()                 # (1, N, H) init hidden from slot
        pos = self.pos(self._pos_idx)                    # (D, H)

        if teacher_digits_msb is not None:
            # parallel teacher-forced pass: previous-digit sequence is known
            tgt_lsb = teacher_digits_msb.flip(-1)        # (N, D) LSB-first
            start = torch.full((N, 1), self.start, dtype=torch.long, device=slot.device)
            prev = torch.cat([start, tgt_lsb[:, :-1]], dim=1)        # (N, D)
            inp = s.unsqueeze(1) + pos.unsqueeze(0) + self.digit_emb(prev)  # (N, D, H)
            out, _ = self.gru(inp, h0)                    # (N, D, H)
            return self.out(out).flip(1)                  # (N, D, base) MSB-first

        # autoregressive greedy decode (eval only): step the GRU one digit at a time
        h = h0
        prev = torch.full((N,), self.start, dtype=torch.long, device=slot.device)
        logits_lsb = []
        for i in range(self.D):
            inp = (s + pos[i] + self.digit_emb(prev)).unsqueeze(1)   # (N, 1, H)
            out, h = self.gru(inp, h)
            logit = self.out(out[:, 0])                  # (N, base)
            logits_lsb.append(logit)
            prev = logit.argmax(-1)
        return torch.stack(logits_lsb, dim=1).flip(1)    # (N, D, base) MSB-first


class ArithGroundingHeads(nn.Module):
    """Drop-in replacement for GroundingHeads whose digit fields use the
    carry-aware head. ``forward(z, teacher=...)`` teacher-forces digits in
    training (pass the target digit labels) and greedily decodes at eval."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.reg_type = nn.Linear(d, len(VType))
        self.reg_sign = nn.Linear(d, 2)
        self.heap_sign = nn.Linear(d, 2)
        h = min(128, d)  # GRU hidden — kept small; it dominates step cost on MPS
        self.reg_digits = ArithDigitHead(d, cfg.base, cfg.max_digits, hidden=h)
        self.heap_digits = ArithDigitHead(d, cfg.base, cfg.max_digits, hidden=h)
        self.pc = nn.Linear(d, cfg.max_pc + 1)
        self.halted = nn.Linear(d, 2)
        self.error = nn.Linear(d, 2)

    def forward(self, z: torch.Tensor, teacher: dict | None = None) -> dict:
        cfg = self.cfg
        N = z.shape[0]
        R, C, D, base = cfg.num_regs, cfg.num_cells, cfg.max_digits, cfg.base
        reg, heap = z[:, :R], z[:, R:R + C]
        pc_slot, flags_slot = z[:, R + C], z[:, R + C + 1]

        reg_td = teacher["reg_digits"].reshape(N * R, D) if teacher else None
        heap_td = teacher["heap_digits"].reshape(N * C, D) if teacher else None
        reg_dig = self.reg_digits(reg.reshape(N * R, -1), reg_td).view(N, R, D, base)
        heap_dig = self.heap_digits(heap.reshape(N * C, -1), heap_td).view(N, C, D, base)
        return {
            "reg_type": self.reg_type(reg),
            "reg_sign": self.reg_sign(reg),
            "reg_digits": reg_dig,
            "heap_sign": self.heap_sign(heap),
            "heap_digits": heap_dig,
            "pc": self.pc(pc_slot),
            "halted": self.halted(flags_slot),
            "error": self.error(flags_slot),
        }


class ArithWM(GroundedLatentWM):
    """Slotted world model with the carry-aware arithmetic grounding heads."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.heads = ArithGroundingHeads(cfg)
