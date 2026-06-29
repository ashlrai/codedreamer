"""A magnitude-invariant integer comparator with an MSB-first lexicographic prior.

The frontier question (FINDINGS_FRONTIER): comparison `a < b` is a fixed function of two
values, yet the world model can't do it out-of-distribution because it never sees nonzero
high-order digits in small-magnitude training. Can an *architectural prior* fix that?

`DigitComparator` is the test. It compares two integers given as (sign, MSB-first digits):
a small **learned, position-shared cell** scores each digit position (lt/eq/gt), and a
**fixed lexicographic combiner** (the most-significant non-equal position decides) turns
those into an order prediction. Two properties make it magnitude-invariant *by
construction*:
  * the per-position cell sees only bounded digit values [0, base) and is SHARED across all
    positions, so a cell trained on the (always-populated) low-order positions transfers
    verbatim to the high-order positions that only light up out of distribution;
  * the lexicographic combiner is a fixed prefix-product reduction, identical at every
    magnitude.

`PlainComparator` is the control: an MLP over the concatenated one-hot digits, no
position-sharing and no lexicographic structure — the failure mode we expect to reproduce.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DigitComparator(nn.Module):
    """Predict order(a, b) in {0:a<b, 1:a==b, 2:a>b} from (sign, MSB-first digits)."""

    def __init__(self, base: int, max_digits: int, h: int = 64) -> None:
        super().__init__()
        self.base = base
        self.max_digits = max_digits
        self.digit = nn.Embedding(base, h)
        # position-shared cell: (emb_a, emb_b) -> 3 logits (a_p<b_p, ==, >)
        self.cell = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 3))

    def _mag_order(self, da: torch.Tensor, db: torch.Tensor) -> torch.Tensor:
        """MSB-first lexicographic magnitude comparison. da,db: (B, D) digit indices.
        Returns (B, 3) probabilities over (mag a<b, a==b, a>b)."""
        ea, eb = self.digit(da), self.digit(db)              # (B, D, h)
        logits = self.cell(torch.cat([ea, eb], dim=-1))      # (B, D, 3)
        p = F.softmax(logits, dim=-1)                        # per-position lt/eq/gt
        lt, eq, gt = p[..., 0], p[..., 1], p[..., 2]         # (B, D), MSB-first
        # prefix product of "equal so far" BEFORE each position
        eq_shift = torch.cat([torch.ones_like(eq[:, :1]), eq[:, :-1]], dim=1)
        prefix_eq = torch.cumprod(eq_shift, dim=1)           # (B, D)
        mag_lt = (prefix_eq * lt).sum(1)
        mag_gt = (prefix_eq * gt).sum(1)
        mag_eq = torch.cumprod(eq, dim=1)[:, -1]             # all positions equal
        return torch.stack([mag_lt, mag_eq, mag_gt], dim=-1)  # (B, 3)

    def forward(self, sign_a, digits_a, sign_b, digits_b) -> torch.Tensor:
        """Returns (B, 3) logits over {a<b, a==b, a>b}. Signs in {0:+, 1:-}."""
        mag = self._mag_order(digits_a, digits_b)            # order of |a| vs |b|
        sa = sign_a.float().unsqueeze(-1)                    # (B,1) 1 if negative
        sb = sign_b.float().unsqueeze(-1)
        # magnitude order, but reversed when both negative
        both_neg = sa * sb
        mag_rev = mag.flip(-1)                               # swap lt<->gt
        same_sign_order = both_neg * mag_rev + (1 - both_neg) * mag
        # different signs: the negative one is smaller (a<b iff a negative & b not)
        diff = (sa - sb).abs()                               # 1 if signs differ
        a_lt_b = (sa > sb).float()                           # a negative, b non-negative
        diff_order = torch.cat([a_lt_b, torch.zeros_like(a_lt_b), 1 - a_lt_b], dim=-1)
        # handle |a|==|b|==0 with different sign bits (both zero) -> treat as equal:
        out = diff * diff_order + (1 - diff) * same_sign_order
        return torch.log(out.clamp_min(1e-9))                # log-probs as logits


class PlainComparator(nn.Module):
    """Control: an MLP over concatenated one-hot digits + signs. No positional prior."""

    def __init__(self, base: int, max_digits: int, h: int = 128) -> None:
        super().__init__()
        self.base = base
        self.max_digits = max_digits
        in_dim = 2 * (max_digits * base + 1)
        self.net = nn.Sequential(nn.Linear(in_dim, h), nn.GELU(),
                                 nn.Linear(h, h), nn.GELU(), nn.Linear(h, 3))

    def _feat(self, sign, digits):
        oh = F.one_hot(digits, self.base).float().flatten(1)  # (B, D*base)
        return torch.cat([oh, sign.float().unsqueeze(-1)], dim=-1)

    def forward(self, sign_a, digits_a, sign_b, digits_b) -> torch.Tensor:
        return self.net(torch.cat([self._feat(sign_a, digits_a),
                                   self._feat(sign_b, digits_b)], dim=-1))
