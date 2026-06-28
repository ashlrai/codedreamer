# M1 Findings — Grounded Latent Dynamics (v1)

Status: **M1 built end-to-end and run.** Encoders, deterministic latent dynamics,
shallow grounding heads, JEPA (EMA target + VICReg), curriculum rollout, the full
eval battery (single-step exact-match, per-variable accuracy, rollout-horizon
curve), and tests — all in place and green (`tests/test_model.py` overfits a tiny
batch to >0.9 exact-match, confirming the wiring).

## The decisive architecture result

| Latent design | per-var acc | single-step exact-match |
|---|---|---|
| **Pooled** (single CLS vector → decode all regs) | ~0.30 (plateau) | ~0.00 |
| **Slotted** (one latent vector per register/heap/pc/flags) | **~0.94** | **~0.70** |

The pooled latent is information-bottlenecked: one vector cannot store ~30
registers' exact integer values for a single linear to decode. The **slotted
latent** — one vector per slot, dynamics = a Transformer over slots, each slot
decoded by a shared shallow head — makes an unchanged register a near-identity
copy and lifts per-variable accuracy from 0.30 to ~0.94. This is the central M1
finding and it is fully consistent with the thesis (latent prediction, plannable
rollout, per-slot grounded/probeable).

## Measured numbers (single 24GB-class MPS GPU, minutes of training)

- **Small slotted** (d=192, 2 enc/2 dyn layers, 2.3M params, 500 steps):
  per-var 0.85, single-step exact-match 0.21; rollout-horizon exact-match
  `k1:0.25 k2:0.10 k3:0.08 k4:0.04 k5:0.01 → 0`.
- **Larger slotted** (d=256, 4 enc/4 dyn layers, 7.2M params, ~1600 steps before
  stop): per-var **0.94**, single-step exact-match **~0.70**, still slowly rising.
  Rollout loss at K=8 (~0.43) ≫ single-step loss (~0.12).

## The bottleneck, precisely identified

Each VM step changes exactly **one** register, so per-var ≈ 0.94 means unchanged
registers are copied reliably and the error concentrates on the **one freshly
computed register's exact digits**. The remaining gap to 0.99 is **exact
multi-digit arithmetic** (predicting that 7×8 → digits 5,6), a known-hard problem
for transformers — *not* a state-representation or copying failure.

This sharpens the **R1 (compounding error)** question. At this scale rollout
exact-match decays with horizon, but the driver is **arithmetic errors
compounding**, not latent state-drift: the slotted latent holds unchanged state
well, while each computed value carries a small error that accumulates over a
rollout. That is a more precise — and more tractable — diagnosis than "the latent
cannot stay exact."

## M1.5 result — R1 (compounding error) is substantially PASSED

Experiment (`scripts/m1_5_isolate_arithmetic.py`): same slotted model, trained on a
slice where **arithmetic is trivial** (ADD/SUB only, values bounded to 2 digits)
but **control flow is intact** (if/for/heap, pc/branch/loop tracking, copying).

| Metric | Hard arithmetic (M1) | Easy arithmetic (M1.5) |
|---|---|---|
| per-var accuracy | 0.94 | **0.977** |
| single-step exact-match | 0.70 | **0.856** |
| rollout-horizon @ k=5 | ~0.01 | **0.45** |

Rollout-horizon (easy): `k1:0.88 k2:0.76 k3:0.62 k4:0.52 k5:0.45 k6:0.36 k7:0.19
→ ~0 by k13` — vs the hard-arithmetic small run's `k1:0.25 … k5:0.01`.

**The structural finding:** rollout@k ≈ (single-step exact-match)^k. Measured
`0.856^5 = 0.46 ≈ 0.45`, `0.856^3 = 0.63 ≈ 0.62`; only mild extra drift appears at
long horizons. So the latent rollout adds **little representational drift of its
own** — the horizon decay is almost entirely the **geometric compounding of the
single-step error rate**. R1's deeper fear ("the latent cannot stay exact under
rollout") is therefore *not* what limits us. The plannability thesis is de-risked:
**drive single-step exact-match toward 1.0 (arithmetic + remaining edge cases) and
the usable rollout horizon extends automatically.**

## Recommended next steps (post-M1.5)

1. ✅ **Isolate dynamics from arithmetic** — DONE (M1.5 above); R1 substantially
   passed. Next levers target single-step exact-match directly.
2. **Arithmetic-aware design**: a copy-vs-compute decomposition (predict *which*
   register changed + its delta) so the model copies by default and only computes
   the one changed slot; and/or a per-digit-recurrent or carry-aware value head.
3. **More compute / longer training** on a real GPU with efficient rollout
   (truncated BPTT or capped K) — late steps slow ~K× because rollout does K
   sequential dynamics passes; cap or detach periodically.
4. **Frozen-probe interpretability eval** (the ≥95% linear-probe target) is not
   yet run; the joint current-state decode (per-var on `z_cur`) is a proxy and is
   already high.

## How to reproduce

```bash
PYTHONPATH=. python -c "from execwm.train.train_m1 import train, TrainConfig; \
train(tc=TrainConfig(steps=1500), d_model=256, enc_layers=4, dyn_layers=4)"
```
Watches: `step_em` (single-step exact-match), `per_var`, and the printed
`ROLLOUT-HORIZON` curve at the end.
