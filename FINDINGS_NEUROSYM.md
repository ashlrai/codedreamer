# The self-imposed wall: execution = learnable control + offloadable arithmetic

**One-line result:** the grounded latent world model's magnitude-generalization wall
lives **entirely in the learned digit-decoder**, not in the latent. Offload arithmetic
to the interpreter's ALU and whole-state exact-match at 10–25× out-of-distribution
magnitude jumps from **0.000 → 0.790** — on the *same frozen weights*. The net's
prediction of *control flow* is essentially magnitude-invariant (next-pc 0.999 → 0.986).

## The experiment (one trained model, eval-time intervention)

Train ONE `GroundedLatentWM` (d=256, ~10M params) on **small-magnitude** programs
(values ≤ ~30) with a codec **wide** enough to represent big numbers
(`max_digits=4` → up to 9999). Then read the *same* model out two ways on an
in-distribution split (values ≤ 30) and a **magnitude-OOD** split (values ~300–800):

- **`EM learned`** — every state field decoded by the net (the status-quo readout).
- **`EM digits-oracle`** — the net's predictions, but the numeric *digit payload* of
  registers/heap is supplied by a perfect ALU. Everything else — pc, type, sign,
  flags, which-slot-changed, comparison outcomes — is still the net's job. This is
  **not** "just running the VM": it isolates whether the net's *structural* prediction
  is correct, with arithmetic (and only arithmetic) offloaded.

`scripts/neurosym_spike.py` + `execwm/eval/neurosym.py`; 1500 steps, MPS;
n ≈ 10.2k transitions per split.

## Result

| split | EM learned | EM digits-oracle | pc acc | written sign | written digits | arith digits | cmp result | branch pc | n |
|---|---|---|---|---|---|---|---|---|---|
| in-distribution (val ≤ 30) | 0.721 | 0.904 | 0.999 | 0.881 | 0.773 | 0.672 | 0.788 | 0.996 | 10,223 |
| **magnitude-OOD (val 300–800)** | **0.000** | **0.790** | **0.986** | 0.798 | 0.254 | 0.103 | 0.626 | 0.989 | 10,285 |

## What it means (and the honest limits)

1. **The wall is the digit head.** Monolithic whole-state EM collapses to **0.000** at
   OOD magnitude, but with arithmetic offloaded the *same model* reaches **0.790**.
   79% of OOD states are fully correct once the net is freed from computing digits.

2. **Control flow is magnitude-invariant.** Next-pc accuracy is **0.999 → 0.986** and
   branch (JZ/JNZ/JMP) pc accuracy **0.996 → 0.989** across a 10–25× magnitude shift.
   The genuinely *predictive* content of execution — which instruction runs next — is
   learnable and barely depends on numeric scale. This is the core of the thesis:
   **execution = (learnable, magnitude-invariant control) + (offloadable arithmetic).**

3. **Arithmetic is the only component that collapses.** Written-register digit accuracy
   **0.773 → 0.254** (arithmetic-op digits 0.672 → 0.103). This is exactly — and only —
   the part a symbolic ALU computes for free given the operands.

4. **The honest residual (the frontier).** `EM digits-oracle` is 0.790, not ~1.0.
   The gap from 1.0 is *not* digits — it is **value-derived predicates**: comparison
   outcomes (0.788 → 0.626) and sign (0.881 → 0.798) degrade because they depend on
   *relative* magnitude, which the net tracks only approximately OOD. (Note both the
   input *encoding* and output *decoding* of large magnitudes are OOD here — the
   structural prediction survives both; only the arithmetic decode does not.) So the
   precise open problem is: **represent comparison-relevant value properties (sign,
   ordering) in a magnitude-invariant way** — i.e. a learned *abstract interpreter*.
   That is the contributor challenge and the v2 architecture.

## Why this matters for the project

- It **reframes the bottleneck**: the entire arithmetic saga (`FINDINGS_M3.md` §5 — the
  carry-aware head and magnitude curriculum both failing at EM ≈ 0.003) was the model
  paying a tax for a design choice. The latent/dynamics were never the limiter for
  *structure*; the digit-decode objective was.
- It **sharpens the latent-vs-token moat**: a token-space trace predictor must emit the
  digits and eats the arithmetic error; a latent model can carry an abstract operation
  and defer numeric realization to the ALU. This factoring is natural in latent space
  and awkward in token space.
- It points to a concrete **neurosymbolic architecture**: net predicts control + dataflow
  structure (magnitude-invariant); the interpreter's ALU realizes values; the residual
  research problem is magnitude-invariant value *comparison*.

## Reproduce

```
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.3 \
  PYTHONPATH=. caffeinate -i python scripts/neurosym_spike.py --steps 1500
```
Artifact: `artifacts/neurosym_model.pt` (the trained model, reused by the demo).
