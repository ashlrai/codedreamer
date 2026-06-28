# M3 Findings — Edit-as-action, planning, and the masked-causality result

M2 left two open questions: (1) was the latent-vs-token causal *tie* real, or an
artifact of both models being arithmetic-limited? and (2) is there any agentic payoff
from planning over edits (risk R4)? M3 built the substrate + planning harness to
answer (2), and a clean "easy-arithmetic" re-test to answer (1).

## 1. The masked-causality result (answers M2's open question)

M2's causal counterfactual test was a tie (latent 0.265 / token 0.285 on `do(register)`),
but BOTH models sat at ~0.40 single-step exact-match — so the causal metric was
arithmetic-noise-dominated. Re-running on the **easy-arithmetic slice** (ADD/SUB, 2-digit
values) isolates causal structure from digit noise. Both models trained on the same
data/spec; graded on the same 300 intervention pairs:

| Model | single-step EM | `do(register)` EM | `do(action)` EM | identity |
|---|---|---|---|---|
| **grounded latent** | **0.843** | **0.557** | **0.150** | 0.0 |
| token-space | 0.437 | 0.353 | 0.053 | 0.0 |
| **latent − token** | **+0.406** | **+0.203** | **+0.097** | — |

(latent hard-spec, for contrast: single-step 0.40, `do(register)` 0.265, `do(action)` 0.045.)

**Two findings:**

1. **The M2 causal "tie" was an artifact of arithmetic error.** Both hard-spec models
   sat at ~0.40 single-step EM, so the causal metric was digit-noise-dominated. On a
   learnable task the latent's causal accuracy more than doubles (`do(register)`
   0.265→0.557, `do(action)` 0.045→0.150).

2. **On a learnable task, the grounded latent decisively beats token-space** on every
   axis (+0.41 whole-state EM, +0.20 / +0.10 causal). The mechanism is the thesis: the
   token model's *per-token* accuracy is 0.99, but whole-state exact-match needs all ~71
   serialized tokens correct (0.99⁷¹≈0.49 → its 0.44), and greedy autoregressive decode
   scatters errors across the serialization. The **latent predicts all state slots
   jointly from grounded representations** — no autoregressive whole-state decode — so it
   lands the entire next state far more often (0.84). Token (600 steps) was fully
   converged (teacher-forced acc 0.99 by step 300): this is a decode-architecture gap,
   not a training-budget one.

This reframes the project: **single-step arithmetic gates every downstream metric**
(exact-match, rollout horizon, causal accuracy, planning) — the M1.6 carry-aware head +
magnitude curriculum target — and **where the task is learnable, the grounded-latent bet
pays off over token-space**, exactly the thesis, now with direct evidence.

Bonus: the easy-arith latent's **rollout horizon** is k1:0.88 → k5:0.45 → k6:0.35 (vs the
hard spec collapsing by k5), reconfirming M1.5's `rollout@k ≈ EM^k`.

(Process note: an earlier apparent token-training "hang" was a false alarm — ~4s/step on
CPU + `log_every=200` ⇒ ~13 min between log lines, compounded by a runaway subagent
`wm_plan_eval` monitor since killed. No bug.)

## 2. Edit-as-action substrate (M3-A) — built

`substrate/edits.py` (Edit / EditKind = CHANGE_OP/DST/OPERAND/IMM, `apply_edit`,
`sample_edit`, `enumerate_valid_edits`), `data/edit_codec.py` (bit-exact round-trip),
`data/edit_dataset.py` (`make_edit_example` → base+edited trace pairs from one
init_state). ~344 examples/s; **~35% of edits change the trace length** (flip a
branch/loop outcome) — the rich control-flow-divergence case. 8 tests.

## 3. Planning + the R4 calibration (M3-B step 2) — the key payoff result

`execwm/plan/` (`goal_tasks.py`, `search_baseline.py`, `planner.py`, `metrics.py`,
13 tests). On 40 constructed 2-edit goal tasks:

| Method | Solved | Mean real VM executions (solved) |
|---|---|---|
| `vm_search` brute-force (no WM) | 40/40 | ~74 |
| `beam_plan` + cheap (non-VM) scorer | 30/40 | **~1.6 (≈82% fewer)** |
| `beam_plan` + oracle (VM) scorer | 40/40 | ~302 (counts every VM scoring call) |

**R4 is de-risked: execution-saving is achievable**, and must come from a *cheap*
(non-VM) scorer. This sets the falsifiable bar for the learned model (M3 step 3): be
the cheap scorer that ALSO handles control flow (the linear heuristic misses ~25% of
tasks because it ignores loops/branches).

### Partial-program regime (the strongest R4 case)

`execwm/plan/partial_tasks.py` + `partial_search.py` (6 tests): tasks with UNBOUND
inputs and a quantified goal (`forall` / `frac≥p`). The VM must enumerate the input
domain per candidate (cost = #candidates × |input domain|; e.g. 1,950 VM runs for one
1-edit task over a 25-input domain) — a world model scoring over inputs in latent space
amortizes this. This is the regime where the WM's payoff should be largest.

## 4. World-model-as-scorer (M3-B step 3 start) — built + evaluated (honest negative)

`execwm/plan/wm_scorer.py` (`WorldModelScorer`: simulates a program in latent space by
decoding `pc` → fetching the instruction → `predict_next`, ZERO VM calls; 3 tests) +
`scripts/wm_plan_eval.py`. Evaluated on the **easy-arith** model (`latent_easy.pt`, the
one that rolls out best), n=20 2-edit goal tasks:

| Method | Success | Mean real-VM execs (solved) |
|---|---|---|
| vm_search brute-force (no WM) | 20/20 | 241.9 |
| beam + cheap structural scorer | 14/20 | 2.1 (**96% saved**) |
| **beam + WM scorer (learned)** | **1/20** | 1.0 (96% saved *when it solves*) |

**Honest result: the WM-as-scorer works in principle but is not yet practical.** When
it solves (1/20 — the one short, 6-instruction program), it runs **0 VM executions**
and saves 96% — the mechanism is proven. But it **times out on 19/20** longer/looping
programs (len 15–63, executed traces up to 128 steps): rolling the latent out over a
full looping program compounds single-step error catastrophically (0.84¹²⁸≈0) and
exceeds the compute guard. The cheap structural heuristic dominates at this scale
(14/20). This is R1/R4 biting exactly where predicted: **long-horizon latent rollout is
the wall.** Two implications: (1) it re-confirms single-step arithmetic accuracy must
reach ~0.99+ for rollout to survive long programs; (2) the right M3 step-3 design is the
**divergence head** (predict *where* an edit changes execution and re-simulate only from
there, instead of rolling the whole program) — `PLAN_M3` §2. The learned planner is not
abandoned; its blocker is now precisely measured.

## 5. Arithmetic magnitude curriculum — full-budget result: NEGATIVE (and revealing)

`train/curriculum.py` + `train/train_arith_curriculum.py` (5 tests) ramp data magnitude
small→target (codec width fixed; eval always at full target magnitude).
`scripts/curriculum_experiment.py` ran baseline vs curriculum at **matched 1500 steps,
d=256, 4 layers** on a magnitude-stressed spec (`max_const=max_input_val=300`,
`max_digits=6` → real multi-digit carry chains):

| arm | single-step EM | per-var | rollout k1/k3/k5 |
|---|---|---|---|
| baseline (hard magnitude from step 0) | 0.0034 | 0.615 | 0.003 / 0.003 / 0.002 |
| curriculum (ramp 1→150→300) | 0.0023 | 0.605 | 0.000 / 0.005 / 0.002 |

**Two findings, both negative and both important:**

1. **The curriculum does not help** (−0.001 EM, −0.01 per-var — a tie/slight loss at
   matched budget). The late-training-wins hypothesis is not supported here.
2. **Both arms essentially fail at high-magnitude arithmetic** (single-step EM ≈ 0.003 —
   the one multi-digit computed register is almost never exactly right; per-var ~0.61 is
   carried by the *other* fields). The curriculum's stage-0 (magnitude 1) trivially hit
   EM 0.34 / per-var 0.90, then **collapsed to ~0.01 / 0.64 the instant magnitude ramped
   to 150–300.** The model learns small-magnitude arithmetic easily and cannot learn
   6-digit arithmetic with values to ~300 at this budget, curriculum or not.

**Interpretation.** Multi-digit arithmetic at scale is the project's **unsolved core**,
and it is *harder than a training-schedule fix*. The difficulty scales steeply with
magnitude: easy slice (2-digit, vals≤~30) → single-step EM 0.84; M2 hard (6-digit codec,
small vals) → 0.40; stressed (6-digit, vals to 300) → 0.003. The carry-aware LSB-first
GRU digit head + magnitude curriculum are not enough. Candidate next levers (untested):
much larger scale/compute, a higher-capacity or explicitly-carry-structured arithmetic
head, or scoping the contribution to the regimes where arithmetic IS learnable (where the
grounded latent already beats token-space decisively) plus the interpretability/causal/
efficiency story. This is a clean kill of the "curriculum fixes arithmetic" hypothesis at
laptop budget — recorded as such.

## 6. Edit-conditioned dynamics + the divergence evidence (M3 step-3, built)

The WM-as-scorer failed because full-program latent rollout compounds error. Two
analyses + a new model attack this directly.

**Rollout-breakdown analysis** (`scripts/rollout_analysis.py`, on `latent_easy.pt`):
exact-match crosses 0.5 at horizon 5 and 0.1 at horizon 8. The decisive surprise:
**teacher-forced ≈ autoregressive rollout** — feeding the true instruction stream barely
helps, so the wall is **register-value misprediction, NOT control-flow/pc divergence**.
Registers break first (<0.5 by h5); pc stays >0.9 to h≈6 then collapses as a *downstream*
consequence of accumulated value errors. Worst ops: **LE/GE/GT/SUB** (SUB ≫ ADD — borrows
are harder than carries for the digit head). Loops don't hurt more (loop bodies repeat
simple ops; dense SUB/comparison is the driver). → Re-ground every ~4 steps / after
SUB/comparison; build a per-*register* confidence signal, not a pc one.

**Edit-locality analysis** (`scripts/edit_locality.py`, 800 samples): a divergence-from-
point planner (reuse the identical base prefix, re-simulate only from the edit's onset)
saves **~46% of execution steps** on average. Once an edit diverges it **stays diverged**
(~82% propagation, only ~14% re-converge), so the head's job is to locate the *onset*,
not handle re-convergence. **~45% of edits change trace length** → it must predict
control-flow divergence, not just value patches. CHANGE_OPERAND edits are most local
(~56–66% saved); CHANGE_DST/OP propagate globally.

**The model** (`model/edit_dynamics.py`, `train/train_edit.py`, 5 tests): `EditConditionedWM`
wraps a `GroundedLatentWM` and predicts each edited state **directly from the base latent
+ an edit embedding via FiLM — with NO latent rollout**, which is precisely how it sidesteps
the compounding-error wall. An `EditEncoder` (mirrors `ActionEncoder`) + zero-init FiLM
(starts at identity) + a `DivergenceHead` predicting per-step P(changed); `edit_loss` =
divergence BCE + grounding CE on edited states (`true_divergence_mask` from base-vs-edited
traces). Single-batch overfit is clean (total 16.0→0.04, div-BCE→0.001). Honest compromise
(documented): conditioning is index-aligned, so after a control-flow edit the index-aligned
grounding target past divergence isn't the semantically-corresponding state — so the
**divergence point is the primary signal, changed-state decode secondary**, and grounding
is trained only on valid steps.

**Synthesis across §4–§6.** Three independent results converge: (a) the WM-scorer's wall is
value misprediction over long horizons (B); (b) a divergence head can avoid ~half the
re-simulation (D) and the new model avoids rollout entirely by predicting from the base
latent (A); but (c) accurate *changed-state* decode after control-flow edits, and accurate
values generally, still hinge on single-step arithmetic — **so the divergence head and the
arithmetic curriculum (§5) are complementary necessities, not alternatives.** The edit-
**Trained result (easy-arith, 1500 steps, d=256, `artifacts/edit_easy.pt`):** the model
WORKS in-regime — **divergence first-step acc 0.734, per-step acc 0.911, edited-state
exact-match 0.703, edited per-var 0.886** (n=500 episodes). It locates where an edit
changes execution ~73% of the time and predicts the edited state exactly ~70% of the
time, all WITHOUT latent rollout. This directly answers the rollout wall: the naive
WM-scorer (full-program rollout) solved 1/20; the edit-conditioned model predicts edited
traces at 0.70 EM by conditioning on the base latent + edit instead of re-simulating.
The architecture is validated where arithmetic is learnable; the open question is whether
it transfers to hard-arithmetic regimes (gated by §5's unsolved arithmetic) and whether
a planner built on it (next) realizes the ~46% execution-saving D predicted.

**The M3 payoff measurement — divergence-aware planner (`scripts/divergence_plan_eval.py`,
`plan/divergence_planner.py`, 6 tests).** The trained `edit_easy.pt` was used as a
divergence-aware planning scorer (`EditConditionedWMScorer`: predicts each candidate edit's
edited state from the cached base latent — ZERO search-time VM calls, NO rollout). On 30
constructed 1-edit goal tasks (easy-arith spec, beam=4):

| Method | Success | Mean VM execs |
|---|---|---|
| `vm_search` brute-force (no WM) | 30/30 | 152.8 |
| beam + cheap structural scorer | 15/30 | 2.8 |
| **beam + edit-WM scorer (trained)** | **7/30** | 4.6 (**61.4% saved** when it solves) |

**Honest verdict: the no-rollout architecture is a validated mechanism but not yet
competitive on solve rate.** It is a real improvement over the naive full-program-rollout
WM-scorer (§4: 1/20 → 7/30), with zero search-time VM calls and 61.4% executions saved over
brute-force *when it solves* — confirming the divergence/edit-conditioning design fixes the
rollout wall. **But it underperforms even the cheap structural heuristic (7/30 vs 15/30):**
at 0.70 edited-state EM the scorer mis-ranks edits ~30% of the time, so the correct edit
often falls outside the verified beam and the task fails. This is the *same root cause* one
level up — prediction accuracy (gated by arithmetic, §5) caps the planning payoff. The
mechanism is proven; closing the gap to the cheap scorer requires the edited-state EM to
rise (better arithmetic / more capacity), exactly the §5 frontier.

## Status

**104 tests green** across the M3 additions (edit substrate, planning, partial tasks,
wm_scorer, curriculum, token_eval, bench planning family, edit-conditioned dynamics,
divergence planner).
ExecWM-Bench reports a `planning` family (`run_bench` family `"planning"`;
`BenchReport.planning`).

## Honest scorecard
- ✅ Masked-causality resolved + thesis validated: on a learnable task the grounded latent BEATS token-space (single-step 0.84 vs 0.44; `do(register)` 0.56 vs 0.35; `do(action)` 0.15 vs 0.05).
- ✅ R4 de-risked: planning saves ~82–96% of executions with a cheap scorer.
- ✅ M3 substrate + planning harness + partial-program regime + bench integration built and tested.
- ✅ WM-as-scorer evaluated: mechanism proven (0 VM calls, 96% saved when it solves) but times out on long/looping programs (1/20) — long-horizon latent rollout is the wall (honest negative).
- ✅ Rollout wall diagnosed (B): value misprediction (SUB/comparison), not control flow; re-ground every ~4 steps.
- ✅ Edit locality quantified (D): divergence-from-point saves ~46%; edits stay diverged; must predict control flow.
- ✅ Edit-conditioned dynamics + divergence head built (A): predicts edited states from base latent + edit via FiLM, NO rollout (sidesteps the wall); 5 tests, clean single-batch overfit.
- ❌ Arith magnitude curriculum: full-budget (1500-step, d=256) run — NEGATIVE. Curriculum doesn't beat baseline, and BOTH fail high-magnitude arithmetic (single-step EM ≈0.003). "Curriculum fixes arithmetic" is killed at this budget; multi-digit arithmetic at scale is harder than a schedule fix.
- ✅ Edit-conditioned model trained (easy-arith): div first-step 0.73, edited-state EM 0.70, NO rollout — fixes the rollout wall in-regime.
- ⚠️ Divergence-aware planner measured (the M3 payoff): edit-WM scorer solves 7/30 with 0 search-time VM calls and 61.4% saved when it solves — beats the naive rollout scorer (1/20) but UNDERPERFORMS the cheap heuristic (15/30). Mechanism proven; solve rate capped by 0.70 edited-EM (mis-ranks edits ~30%) → gated by arithmetic (§5).

The through-line: **arithmetic accuracy gates everything** — and at laptop budget,
high-magnitude multi-digit arithmetic is genuinely *unsolved*: the carry-aware head AND
the magnitude curriculum both fail (single-step EM ≈0.003 at vals→300). Two honest
strategic options follow: **(1)** treat arithmetic as a scale/architecture problem (much
more compute, or a higher-capacity / explicitly-carry-structured head) and revisit; or
**(2)** scope the contribution to the regimes where arithmetic IS learnable — where the
grounded latent already *decisively beats* token-space on exactness AND causality — and
lead with the interpretability + causal + efficiency + planning-substrate story, treating
hard arithmetic as characterized-but-open. The evidence now cleanly supports framing (2)
as the publishable core, with (1) as the scaling frontier.
