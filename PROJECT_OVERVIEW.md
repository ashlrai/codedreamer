# ExecWM — Project Overview

*A grounded latent world model of computation. Synthesis of M0–M3.*

This document pulls the M0–M3 findings into one honest narrative. Every number
traces to a findings doc (`FINDINGS_M1.md`, `FINDINGS_M1_6.md`, `FINDINGS_M2.md`,
`FINDINGS_M3.md`); where two docs disagree it is flagged in §8. All results are
from a single laptop (Apple MPS / CPU), minutes-to-hours of training — not a
GPU-scale run.

---

## 1. Thesis & moat

Learn a world model where the *world is a program executing*: predict program-state
evolution in a **learned latent** that is simultaneously **grounded** (exactly
decodable to symbolic machine state via shallow heads), **interpretable** (linear
probes recover state; latent is causally manipulable), **causal/plannable**
(action-conditioned, counterfactual-correct, searchable in latent imagination), and
**efficient** (a transformer over ~30 state slots, not a 100s-of-tokens trace
string). Code is the only domain with a free, exact, symbolic oracle at every step,
which lets one model be latent-efficient *and* verifiable *and* interpretable at
once. The moat: **every prior execution predictor — Learning-to-Execute, CodeExecutor,
TRACED, NExT, SemCoder, CRUXEval, and Meta's 32B Code World Model (2510.02387) —
predicts execution in token/text space. Nobody has built a grounded *latent* one.**
A simplification that is itself a contribution: because transitions are deterministic
and ground truth is free, this needs no Dreamer-style stochastic RSSM — a
deterministic latent is correct and keeps the JEPA objective clean.

---

## 2. What's built (M0 → M3)

| Milestone | Deliverable | Status |
|---|---|---|
| **M0** | DSL → register bytecode VM + per-instruction tracer; lossless state↔tensor codec; 5 OOD-axis disjoint split generators | Complete |
| **M1** | Slotted-latent world model: structured state/action encoders, deterministic latent dynamics, shallow grounding heads, JEPA (EMA target + VICReg), curriculum rollout, eval battery | Built + run |
| **M1.5** | Arithmetic-isolation experiment (R1 / compounding-error kill-test) | Run; R1 substantially passed |
| **M1.6** | Carry-aware LSB-first arithmetic head; copy-vs-compute delta heads; three eval modules (OOD, interpretability/probes, counterfactual) | Built (arith inconclusive, delta negative) |
| **M2** | ExecWM-Bench: 4 metric families → one `BenchReport` from one command; matched-budget token-space baseline; latent-vs-token + causal head-to-heads | Built + run |
| **M3** | Edit-as-action substrate; planning harness + R4 calibration; partial-program regime; WM-as-scorer; arithmetic magnitude curriculum; edit-conditioned dynamics + divergence head | Built; key results in §3–§5 |

**Package map** (`execwm/`, 43 files):

```
substrate/   vm.py · dsl.py · generators.py · edits.py            (ground-truth oracle + edit actions)
data/        state_codec.py · action_codec.py · edit_codec.py · dataset.py · edit_dataset.py · torch_data.py
model/       world_model.py (slotted latent WM) · arith.py (carry-aware head) · delta.py (copy-vs-compute)
             token_baseline.py (matched control) · edit_dynamics.py (edit-conditioned + divergence head)
train/       train_m1.py · train_arith.py · train_delta.py · train_token.py
             curriculum.py · train_arith_curriculum.py · train_edit.py
eval/        execwm_bench.py · report.py · ood_eval.py · probes.py · counterfactual.py
             token_eval.py · checkpoint.py
plan/        goal_tasks.py · planner.py · search_baseline.py · metrics.py
             partial_tasks.py · partial_search.py · wm_scorer.py
tests/       98 tests green (per FINDINGS_M3)
scripts/     m0_report.py · m1_5_isolate_arithmetic.py · run_execwm_bench.py · compare_token.py
             compare_causal.py · causal_easy.py · curriculum_experiment.py · rollout_analysis.py
             edit_locality.py · wm_plan_eval.py · train_edit_easy.py
```

**The architecture result that unlocked everything (M1):** a **pooled** latent (one
CLS vector → decode all registers) is information-bottlenecked — per-var ~0.30,
single-step exact-match ~0.00. A **slotted** latent (one vector per register/heap/
pc/flags, dynamics = transformer over slots, shared shallow decode head) lifts
per-var to ~0.94 and single-step exact-match to ~0.70. Unchanged slots become
near-identity copies; error concentrates on the one freshly-computed register.

---

## 3. Key results — positives and negatives side by side

### 3a. Latent vs token-space (the headline comparison), by regime

| Metric | **Hard-arith regime** (M2, 6-digit codec) | **Learnable regime** (M3, ADD/SUB 2-digit) |
|---|---|---|
| single-step whole-state EM — latent | 0.396 | **0.843** |
| single-step whole-state EM — token | 0.326 | 0.437 |
| Δ (latent − token) | **+0.070** | **+0.406** |
| per-variable acc — latent / token | 0.882 / 0.940 (token +0.058) | — |
| `do(register)` EM — latent / token | 0.265 / 0.285 → **TIE** | **0.557 / 0.353 (+0.203)** |
| `do(action)` EM — latent / token | 0.045 / 0.035 | **0.150 / 0.053 (+0.097)** |
| identity ("no change") baseline | 0.000 | 0.000 |

*Mechanism (constant across regimes):* the token model's per-token accuracy is
~0.99, but whole-state exact-match needs *all* serialized tokens right (≈71 tokens
→ 0.99⁷¹ ≈ 0.49 ≈ its 0.44) and greedy autoregressive decode scatters errors. The
latent predicts all slots **jointly, in one shot** — no autoregressive whole-state
decode — so it lands the entire next state far more often. The token model (600
steps) was fully converged (teacher-forced acc 0.99 by step 300): this is a
**decode-architecture gap, not a budget gap.**

### 3b. Interpretability & causal-manipulability (latent only; token has no analogue)

| Metric | Result | Source |
|---|---|---|
| Frozen linear-probe accuracy, every field | ≥0.95 (reg_digits 0.999, others 1.0) | M1.6 / M2 |
| Causal-intervention flip-rate | 1.0 | M1.6 / M2 |

You cannot linearly probe or causally edit the "internal state" of a token string —
this property is structurally unavailable to the baseline.

### 3c. Planning / R4 execution-saving

| Experiment | Method | Solved | Mean real VM execs (solved) |
|---|---|---|---|
| 40 2-edit goal tasks (M3) | `vm_search` brute force (no WM) | 40/40 | ~74 |
| | `beam_plan` + cheap structural scorer | 30/40 | **~1.6 (≈82% fewer)** |
| | `beam_plan` + oracle (VM) scorer | 40/40 | ~302 |
| WM-as-scorer, easy model, n=20 (M3) | `vm_search` brute force | 20/20 | 241.9 |
| | beam + cheap structural scorer | 14/20 | 2.1 (**96% saved**) |
| | **beam + learned WM scorer** | **1/20** | 1.0 (96% saved *when it solves*) |

The learned WM scorer runs **zero** VM calls and saves 96% on the one short program
it solves — mechanism proven — but times out on 19/20 longer/looping programs
(latent rollout over a full looping program compounds error: 0.84¹²⁸ ≈ 0). Honest
negative; the wall is precisely diagnosed in §4.

### 3d. Efficiency (held decisively)

The latent path (transformer over ~30 slots) is dramatically cheaper than the token
path (≈375-token serialized sequences + autoregressive decode that OOM'd a 128 GB
machine). The JEPA-efficiency half of the thesis held without qualification.

### 3e. The arithmetic-magnitude wall (the central negative)

| Regime | spec | single-step EM |
|---|---|---|
| Easy | ADD/SUB only, 2-digit, vals ≤ ~30 | **0.84** (M3) / 0.856 (M1.5) |
| Hard (M2) | 6-digit codec, small vals | **0.40** (0.396) |
| Stressed | 6-digit, vals to 300, real carry chains | **0.003** |

Single-step EM collapses two-plus orders of magnitude as numeric magnitude grows.
The carry-aware head and the magnitude curriculum (§5) **both fail** the stressed
regime at laptop budget.

---

## 4. The central finding

**Single-step arithmetic accuracy gates every downstream metric** — whole-state
exact-match, rollout horizon, causal accuracy, and planning. Three independent
results converge on this:

1. **Rollout decay is geometric in single-step error, not latent drift (M1.5).**
   Measured `rollout@k ≈ EM_single^k` (e.g. `0.856⁵ = 0.46 ≈ 0.45` observed). The
   slotted latent holds unchanged state well; horizon decay is almost entirely the
   compounding of the single-step error rate. So the lever that matters is
   single-step EM — drive it toward 1.0 and the usable rollout horizon extends
   automatically.
2. **The M2 causal "tie" was an artifact of arithmetic noise (M3).** At ~0.40
   single-step EM both models' causal metrics were digit-noise-dominated. On a
   *learnable* slice the latent's causal accuracy more than doubles
   (`do(register)` 0.265 → 0.557) and decisively beats token-space.
3. **Where arithmetic is learnable, the grounded-latent bet pays off (M3).** The
   latent wins on every axis — +0.41 whole-state EM, +0.20 / +0.10 causal — exactly
   the thesis, now with direct evidence.

So: **the grounded-latent bet wins where arithmetic is learnable; hard,
high-magnitude multi-digit arithmetic is the open frontier.**

---

## 5. Honest limitations & open problems

- **Arithmetic-at-scale is unsolved.** Single-step EM 0.84 (2-digit) → 0.40 (6-digit,
  small vals) → 0.003 (vals to 300). The **magnitude curriculum did not fix it**:
  at matched full budget (1500 steps, d=256, 4 layers) curriculum scored EM 0.0023
  vs baseline 0.0034 — a tie/slight loss, and both essentially fail. Stage-0
  (magnitude 1) trivially hit EM 0.34 then **collapsed the instant magnitude ramped
  to 150–300.** A clean kill of "curriculum fixes arithmetic" at laptop budget.
- **The long-horizon rollout wall.** The naive WM-as-scorer planner fails on long
  programs (1/20). `rollout_analysis.py` (on `latent_easy.pt`): EM crosses 0.5 at
  horizon 5, 0.1 at horizon 8. The decisive surprise: **teacher-forced ≈
  autoregressive rollout** — feeding the true instruction stream barely helps, so
  the wall is **register-value misprediction, not control-flow/pc divergence**. pc
  stays >0.9 to h≈6 then collapses as a *downstream* consequence. Worst ops:
  **SUB / LE / GE / GT** (SUB ≫ ADD — borrows are harder than carries). Loops don't
  hurt more; dense SUB/comparison is the driver. → Re-ground every ~4 steps / after
  SUB/comparison; build a per-*register* confidence signal.
- **Copy-vs-compute (delta heads) was a negative.** Predicting a per-slot change
  gate + value plateaued at single-step EM ~0.27 — *worse* than M1's 0.70 — because
  it only fixes state-drift, which M1.5 already showed is not the binding
  constraint, while adding its own gate error.
- **Carry-aware arithmetic head is inconclusive, not a win.** Built,
  literature-backed, unit-proven to learn and greedy-decode (>0.9), but the
  GRU-over-slots is heavy on MPS (~0.5 steps/s) so the affordable run was *smaller*
  than the M1 baseline (3.4M params / 900 steps vs 7.2M / 1500) — **not a clean
  head-to-head.** No win claimed.
- **Edit-conditioned dynamics + divergence head: built, not yet run at budget.**
  `EditConditionedWM` predicts edited states directly from the base latent + an edit
  embedding via FiLM with **no rollout** (sidesteps the wall); a `DivergenceHead`
  predicts per-step P(changed). Single-batch overfit is clean (loss 16.0 → 0.04).
  Honest compromise: conditioning is index-aligned, so after a control-flow edit the
  changed-state decode past divergence isn't semantically aligned — **divergence
  point is the primary signal, changed-state decode secondary.** Real at-budget test
  is pending.
- **Laptop-budget caveats everywhere.** All runs are tiny (≤1600 steps, 2–10M
  params, MPS/CPU). Neither model is near the ≥0.99 exact-match bar. The token
  baseline ran on CPU for memory reasons; matched on data/steps, not hardware. The
  clean win/loss verdicts (especially the causal-superiority claim at the hard spec)
  **need a GPU-scale matched run** — exactly the experiment the plan reserved for
  real compute.

---

## 6. Two strategic paths forward

**Path 1 — Arithmetic as a scale/architecture problem (the frontier).** Treat
high-magnitude multi-digit arithmetic as the unsolved core and attack it with much
more compute, a higher-capacity or explicitly-carry-structured arithmetic head, and
a *matched-config* re-test of the carry-aware head. Upside: if single-step EM reaches
~0.99, rollout horizon, planning, and causal accuracy all unlock automatically (per
§4). Risk: it may be a genuine scale problem requiring a GPU and is not guaranteed to
yield at solo budget.

**Path 2 — Scope the publishable contribution to the in-regime wins (recommended
near-term core).** Lead with what is already validated and robust:
- the grounded latent **decisively beats** matched token-space where arithmetic is
  learnable (whole-state EM 0.84 vs 0.44; `do(register)` 0.56 vs 0.35);
- **interpretability** (frozen-probe ≥0.95, flip-rate 1.0) and **causal
  manipulability** the token baseline structurally cannot offer;
- **efficiency** (latent rollout vs OOM-prone token decode);
- the **planning substrate + R4 result** (cheap scorer saves 82–96% of VM
  executions; partial-program regime where the VM literally cannot evaluate a
  candidate without enumerating the input domain);
- hard arithmetic presented as **characterized-but-open**, with the curriculum
  negative and the rollout-wall diagnosis as honest, informative results.

**Recommendation:** make Path 2 the near-term publishable core (ExecWM-Bench + the
in-regime latent-vs-token win + interpretability/causal/efficiency/planning story),
with Path 1 as the explicitly-scoped scaling frontier. **Defer the final decision** —
both are legitimate and the choice depends on available compute.

---

## 7. Reproduce

```bash
# Test suite (98 tests per M3)
python -m pytest -q

# M0 acceptance report
PYTHONPATH=. python scripts/m0_report.py

# M2: full ExecWM-Bench + matched token-space baseline (latent vs token, all families)
PYTHONPATH=. python scripts/run_execwm_bench.py --steps 800 --n-eval 400
#   --quick (core+counterfactual only) · --arith (carry-aware ArithWM latent) · --reuse-token

# Latent-vs-token causal head-to-head (hard spec — the M2 "tie")
PYTHONPATH=. python scripts/compare_causal.py

# Easy-arithmetic causal re-test (the M3 win: 0.84 vs 0.44, do(register) 0.56 vs 0.35)
PYTHONPATH=. python scripts/causal_easy.py

# Arithmetic magnitude curriculum (the full-budget NEGATIVE result)
PYTHONPATH=. python scripts/curriculum_experiment.py

# Supporting diagnostics
PYTHONPATH=. python scripts/m1_5_isolate_arithmetic.py   # R1 / rollout@k ≈ EM^k
PYTHONPATH=. python scripts/rollout_analysis.py          # rollout wall: value misprediction, SUB/comparison
PYTHONPATH=. python scripts/edit_locality.py             # divergence-from-point saves ~46%
PYTHONPATH=. python scripts/wm_plan_eval.py              # WM-as-scorer (honest 1/20 negative)
```

---

## 8. Inconsistencies noticed across the findings docs

These are minor and mostly stale-cache artifacts, surfaced for cleanup:

1. **Test count in `README.md` is stale and internally inconsistent.** The layout
   block says "tests/ 63 tests" while the Quick-start says "(51 tests)" twice — both
   in the same file. The true progression is M1.6 = 36 → M2 = 51 → **M3 = 98**
   (`FINDINGS_M3.md`). README predates M3.
2. **`do(register)` for the latent on the hard spec differs by run.** `FINDINGS_M2.md`
   reports **0.255** in the counterfactual-family caveat but **0.265** in the
   `compare_causal.py` head-to-head table; `README.md` quotes 0.255. These are two
   different eval runs (the per-family eval vs the n=200 paired head-to-head), not a
   contradiction, but the doc text does not flag that.
3. **Easy-spec single-step EM differs slightly between docs:** `FINDINGS_M1.md`/M1.5
   report **0.856**, `FINDINGS_M3.md` reports **0.843** (rounded "0.84"). Different
   training runs of the same easy spec; both are cited above as a range.
4. **Latent parameter count varies by run** (not flagged as such): `FINDINGS_M1.md`
   larger slotted run = **7.2M params**; `FINDINGS_M2.md` latent = **10.4M params**.
   Different model configs across milestones; worth stating explicitly when the
   numbers are reused.
5. **README causal framing predates M3.** The README still frames the causal axis as
   an unproven null result ("tie", "unproven at laptop budget"). `FINDINGS_M3.md`
   resolves this: the tie was an arithmetic-noise artifact, and on the learnable
   regime the latent wins. The README's M2 framing was correct *at the time* but is
   superseded.
