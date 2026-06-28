# M1.6 Findings — Attacking the Single-Step Bottleneck + Eval Suite

M1.5 established that rollout-horizon decay is ≈ geometric in the **single-step
error rate** (rollout@k ≈ EM_single^k), not extra latent drift. So the one lever
that matters is single-step exact-match. M1.6 tried two routes at it, and the
fleet built out the evaluation suite.

## 1. Copy-vs-compute (delta heads) — NEGATIVE result, and why it's informative

`execwm/model/delta.py` predicts, per slot, a **change gate** + a value, and the
predicted next state *copies* unchanged slots and writes the computed value only
where the gate fires. Intuition: stop re-predicting all ~30 slots each step so
errors can't spread.

Result on the hard-arithmetic spec (same as M1's 0.94 per-var / 0.70 exact-match):
single-step exact-match **plateaued at ~0.27** — *worse* than M1. The change-gate
must be correct on *every* slot for a whole-state match, and it introduces its own
error; meanwhile the value head (arithmetic) is still the real bottleneck (its loss
never converges to 0).

This is a clean confirmation of the M1.5 thesis: copy-vs-compute only fixes
**state-drift**, which M1.5 already showed is *not* the binding constraint. The
binding constraint is **single-step arithmetic error**. The module and its tests
are kept as a documented negative result.

## 2. Carry-aware arithmetic head (`execwm/model/arith.py`) — the right lever

Following a dedicated literature review (Abacus embeddings 2405.17399,
Learning-to-Execute 1410.4615, Nogueira'21, Lee'23, Zhou'24), the M1 digit head is
the worst case the field warns about: **MSB-first, independent per-digit, fixed-width
linear**. The replacement `ArithDigitHead`:
* emits digits **LSB-first** (carry flows low→high),
* is **autoregressive over digits** (digit *i* conditions on the lower digits) via a
  GRU — but only *within a single readout*, so latent rollout stays
  non-autoregressive and re-encoding-free,
* **input-injects** the slot vector and a weight-shared **significance embedding** at
  every digit step,
* teacher-forced (parallel) in training, greedy at inference; output flipped back to
  MSB-first so all existing loss/exact-match/codec code is untouched.

Unit-tested (`tests/test_arith_delta.py`): the head **learns a slot→digits map and
greedy-decodes it back at >0.9** — the AR machinery works.

**Result (honest, and inconclusive at this scale):** the GRU-over-all-register-slots
is heavy on MPS (~0.5 steps/s), so the affordable run was *smaller* than the M1
baseline — 3.4M params, 900 steps, d=192/3-layer — and reached **single-step
exact-match 0.396, per-var 0.893** (greedy), rollout `k1:0.38 k3:0.20 k5:0.09`. The
M1 baseline numbers (0.70/0.94) came from a *larger* run (7.2M params, 1500 steps,
d=256/4-layer), so this is **not** a clean head-to-head — the carry-aware head
neither clearly beat nor lost to M1 at matched honesty, because the configs differ.
The teacher-forced training metric (~0.45) sits above the greedy eval (0.396),
the expected exposure-bias gap of AR decoding.

**Conclusion:** the head is built, literature-backed, and unit-proven to learn and
greedy-decode arithmetic; a *matched-config* comparison (same params/steps as M1,
ideally with input-injection + a curriculum over magnitude per the memo) is the
clean experiment, and it is compute-bound on a laptop — the natural first task on a
real GPU. We did not overclaim a win.

## 3. Evaluation suite (built in parallel by three agents)

* **`execwm/eval/ood_eval.py`** — single-step / rollout exact-match in-dist vs each
  of the 5 OOD axes (reusing the M1 `evaluate`/`rollout_horizon`). Note: the
  nesting-depth and program-size axes change the register shape, so they need a
  model trained on that axis' spec; magnitude/trace-length/compositional evaluate on
  one model. (4 tests.)
* **`execwm/eval/probes.py`** — frozen-encoder linear probes + Othello-GPT-style
  causal intervention. Demo on a *briefly* trained encoder: **linear probe accuracy
  ≥95% on every field** (reg_digits 0.999, others 1.0), **causal-intervention
  flip-rate 1.0**. The latent linearly encodes — and is causally manipulable as — the
  machine state. Meets the M1 interpretability target. (3 tests.)
* **`execwm/eval/counterfactual.py`** — the M2 headline metric: intervene on a
  register value or swap the instruction, predict the next state, grade against the
  VM oracle. Both interventions **crush the identity ("no change") baseline (0.0)**
  and discriminate cleanly (register-do tracks in-dist; action-swap is harder — the
  intended causal signal). (5 tests.)

## Status

36 tests green. The project now has: the M0 substrate, the M1 slotted world model
(per-var ~0.94), the M1.5 R1-pass finding, the M1.6 arithmetic head (the validated
lever), and a three-pronged eval suite (OOD generalization, interpretability,
causal counterfactuals) ready to grade trained models — i.e. the seeds of
`ExecWM-Bench` (M2).

## Next (M2)
Wire the eval suite into a single `ExecWM-Bench` report over a properly-trained
model (real GPU): single-step/rollout exact-match, per-axis OOD, frozen-probe
interpretability, and counterfactual causal accuracy vs the token-space baseline —
the project's headline contribution.
