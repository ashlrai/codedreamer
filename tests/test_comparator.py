"""Tests for the magnitude-invariant comparator (the frontier prior).

Contract + a fast functional check: the MSB-first prior generalizes integer comparison
to out-of-distribution magnitudes (trained on |v|<=30, tested on |v| in 300-800) far
better than a plain MLP. Kept small so it runs in a couple of seconds on CPU.
"""
import random

import torch

from execwm.data.state_codec import CodecConfig, encode_int
from execwm.model.comparator import DigitComparator, PlainComparator

_CODEC = CodecConfig(max_digits=4, base=10, max_pc=64)


def _order(a, b):
    return 0 if a < b else (1 if a == b else 2)


def _batch(rng, n, lo, hi):
    sa, da, sb, db, y = [], [], [], [], []
    for _ in range(n):
        def pick():
            v = rng.randint(lo, hi)
            return -v if rng.random() < 0.5 else v
        a, b = pick(), pick()
        s1, d1 = encode_int(a, _CODEC)
        s2, d2 = encode_int(b, _CODEC)
        sa.append(s1); da.append(d1); sb.append(s2); db.append(d2); y.append(_order(a, b))
    return (torch.tensor(sa), torch.tensor(da), torch.tensor(sb), torch.tensor(db),
            torch.tensor(y))


def _train_acc(model, steps=250, seed=0):
    rng = random.Random(seed)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(steps):
        sa, da, sb, db, y = _batch(rng, 256, 0, 30)
        loss = torch.nn.functional.cross_entropy(model(sa, da, sb, db), y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        ood = _batch(random.Random(999), 2000, 300, 800)
        pred = model(*ood[:4]).argmax(-1)
        return (pred == ood[4]).float().mean().item()


def test_comparator_shapes():
    torch.manual_seed(0)
    m = DigitComparator(_CODEC.base, _CODEC.max_digits)
    sa, da, sb, db, _ = _batch(random.Random(0), 8, 0, 30)
    out = m(sa, da, sb, db)
    assert out.shape == (8, 3)


def test_msb_prior_generalizes_better_than_plain():
    torch.manual_seed(0)
    prior_ood = _train_acc(DigitComparator(_CODEC.base, _CODEC.max_digits))
    plain_ood = _train_acc(PlainComparator(_CODEC.base, _CODEC.max_digits))
    # the prior is magnitude-invariant by construction -> strong OOD; plain collapses
    assert prior_ood > 0.9, f"prior OOD acc {prior_ood:.3f} unexpectedly low"
    assert prior_ood > plain_ood + 0.1, f"prior {prior_ood:.3f} not clearly > plain {plain_ood:.3f}"
