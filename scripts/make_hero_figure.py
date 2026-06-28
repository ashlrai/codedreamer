"""Generate the README hero figure: the magnitude wall, and how offloading arithmetic
dissolves it. Computes fresh from the trained checkpoint so the figure is reproducible.

    PYTHONPATH=. python scripts/make_hero_figure.py
writes assets/magnitude_wall.png
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from execwm.eval.demo_backend import LEVELS, DemoEngine


def main() -> None:
    os.makedirs("assets", exist_ok=True)
    eng = DemoEngine()
    pure, neuro = [], []
    for mag in LEVELS:
        a = eng.aggregate(mag, seed=0, n=60)
        pure.append(100 * a["pure_net"])
        neuro.append(100 * a["neurosym"])
        print(f"  mag {mag:>4}: pure-net {pure[-1]:5.1f}%   neurosym {neuro[-1]:5.1f}%",
              flush=True)

    x = list(range(len(LEVELS)))
    plt.figure(figsize=(8, 4.5))
    plt.rcParams.update({"font.size": 11})
    plt.axvspan(-0.3, 0.3, color="#2e7d32", alpha=0.08)
    plt.text(0, 101, "trained here", ha="center", fontsize=9, color="#2e7d32")
    plt.plot(x, neuro, "-o", color="#2e9e4f", lw=2.5, label="🟢 neurosymbolic (offload arithmetic)")
    plt.plot(x, pure, "-o", color="#d6453d", lw=2.5, label="🔴 pure-net (decode digits)")
    plt.xticks(x, [f"≤{m}" for m in LEVELS])
    plt.ylim(-5, 108)
    plt.xlabel("input magnitude  (left = training regime → right = far out-of-distribution)")
    plt.ylabel("next-state exact-match (%)")
    plt.title("CodeDreamer: the magnitude wall is the digit head, not the latent")
    plt.legend(loc="center left", frameon=False)
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    out = "assets/magnitude_wall.png"
    plt.savefig(out, dpi=140)
    print(f"[hero] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
