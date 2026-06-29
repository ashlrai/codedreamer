"""Does an MSB-first architectural prior make integer comparison magnitude-invariant?

Train two comparators to predict order(a,b) in {<,==,>} from (sign, MSB-first digits),
on SMALL-magnitude pairs only (|v|<=30), then test on far-OOD pairs (|v| in 300-800) with
the SAME codec width. If the prior works, `DigitComparator` (position-shared cell + fixed
lexicographic combiner) generalizes where `PlainComparator` (an MLP over concatenated
digits) collapses — isolating the frontier mechanism from the world model.

    PYTHONPATH=. python scripts/comparator_probe.py
"""
from __future__ import annotations

import argparse
import random

import torch

from execwm.data.state_codec import CodecConfig, encode_int
from execwm.model.comparator import DigitComparator, PlainComparator


def _order(a: int, b: int) -> int:
    return 0 if a < b else (1 if a == b else 2)


def _batch(rng: random.Random, n: int, lo: int, hi: int, codec: CodecConfig):
    """n random (a,b) pairs with |value| in [lo, hi]; returns tensors + labels."""
    sa, da, sb, db, y = [], [], [], [], []
    for _ in range(n):
        def pick():
            v = rng.randint(lo, hi)
            return -v if rng.random() < 0.5 else v
        a, b = pick(), pick()
        s1, d1 = encode_int(a, codec)
        s2, d2 = encode_int(b, codec)
        sa.append(s1); da.append(d1); sb.append(s2); db.append(d2); y.append(_order(a, b))
    return (torch.tensor(sa), torch.tensor(da), torch.tensor(sb), torch.tensor(db),
            torch.tensor(y))


@torch.no_grad()
def _acc(model, batch) -> float:
    sa, da, sb, db, y = batch
    pred = model(sa, da, sb, db).argmax(-1)
    return (pred == y).float().mean().item()


def _train(model, codec, steps: int, seed: int) -> None:
    rng = random.Random(seed)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    model.train()
    for _ in range(steps):
        sa, da, sb, db, y = _batch(rng, 256, 0, 30, codec)   # in-distribution magnitudes
        logits = model(sa, da, sb, db)
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    codec = CodecConfig(max_digits=4, base=10, max_pc=128)   # represents up to 9999
    rng = random.Random(args.seed + 1)
    indist = _batch(rng, 4000, 0, 30, codec)      # held-out in-distribution
    ood = _batch(rng, 4000, 300, 800, codec)      # far out-of-distribution

    rows = []
    for name, ctor in [("DigitComparator (MSB-first prior)", DigitComparator),
                       ("PlainComparator (MLP, no prior)", PlainComparator)]:
        model = ctor(codec.base, codec.max_digits)
        _train(model, codec, args.steps, args.seed)
        rows.append((name, _acc(model, indist), _acc(model, ood)))

    print("\n# Comparator probe — can an MSB-first prior generalize comparison OOD?\n")
    print("| comparator | in-dist acc | OOD acc |")
    print("|---|---|---|")
    for name, i, o in rows:
        print(f"| {name} | {i:.3f} | {o:.3f} |")
    print("\n(chance = 0.333; trained only on |value|<=30, tested on |value| in 300-800)")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
