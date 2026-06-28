# M3 Design — Action = Edit, and Planning in Latent Imagination

M0–M2 established: a slotted grounded-latent world model that predicts single-step
state transitions, is linearly probeable + causally manipulable, and (at laptop
budget) is roughly on par with a token-space baseline on raw exactness while
uniquely carrying interpretability + causal structure. M3 is the **agentic payoff**:
make *edits to a program* the actions, plan over them inside the latent, and show the
plan saves real interpreter executions.

This doc fixes the design **before** building the planner, because risk **R4 (the
latent isn't cheaper than running the VM)** is the most underrated threat and is an
attribute of the *task*, not the model. If we pick a task where running the VM is
free, no world model can win. So we choose the task first.

## 1. Edit as action — the substrate (M3-A, being built)

An `Edit` mutates one bytecode `Instr` (change op / dst / operand / immediate),
applied with `apply_edit(program, edit) -> program'`, encoded by an `EditCodec`
analogous to the statement `ActionCodec`. An `EditExample` carries
`(base_bytecode, edit, init_state, base_trace, edited_trace)` — both traces from the
same `init_state`. The learning signal is the **trace divergence**: given the base
execution and an edit, predict how the trajectory of machine states changes.

Two framings, to ablate:
- **Re-encode** (simplest, honest baseline): apply the edit symbolically, re-encode
  the new program, roll the existing single-step dynamics forward. The WM never
  "sees" the edit as an action — it just simulates the new program. This is the
  control that R2/R4 must beat.
- **Edit-conditioned dynamics** (the claim): the WM takes `(base latent trajectory,
  edit action)` and predicts the *new* trajectory's latents **without** re-running
  per-step ground truth — i.e. it learns the *effect of an edit on execution*. A
  FiLM/cross-attention conditioning of the edit embedding onto the slot dynamics,
  trained on `(base_trace, edit) -> edited_trace` pairs with the same grounding +
  JEPA + rollout losses as M1. Success = predicting the edited trace's divergence
  point and downstream states at exact-match, cheaper than re-encoding+rolling.

## 2. Architecture sketch (extends `model/world_model.py`, do not rewrite)

- **Edit encoder** `Q_θ(edit)` → an edit embedding (reuse the structured-field
  embedding style of `ActionEncoder`: kind-emb ⊕ target-index-emb ⊕ changed-value-emb).
- **Conditioning**: inject the edit embedding into `LatentDynamics` via FiLM on the
  slot stream (γ,β per slot from the edit embedding) OR a cross-attention block. Keep
  the per-slot grounding heads UNCHANGED so all M1/M2 eval + probes still apply.
- **Divergence head** (optional, cheap): per-step predict P(this step's state differs
  from base) — a learned "where does the edit first bite" signal that prunes planning.
- Train with the M1 objective set (`L_ground + L_jepa + L_rollout`) on edited traces,
  plus a **counterfactual-edit pair loss**: same base, two different edits → two
  different edited traces (free labels), forcing real edit-use (mirrors M1's
  `L_action`).

## 3. The planner (M3-B)

Search over edits in latent imagination to hit a goal predicate on the final/any
state (e.g. "make the program output 42", "make the assertion pass", "reach pc=K with
reg2>0"). Planner = **beam search** (and a CEM variant to ablate) over edit sequences,
scoring candidates by rolling the latent forward and decoding the goal predicate —
**never touching the VM during search**. Only the chosen plan is verified once on the
real VM.

## 4. R4-aware task design (decided up front)

The payoff is **pruning agent search across many hypothetical edits in regimes where
running the VM is NOT free**. We will NOT claim to beat the interpreter on one short
program. The benchmark tasks are chosen so a real execution is expensive or
impossible per-candidate:
1. **Expensive tests** — the goal predicate requires running the edited program on a
   large/long input battery; the WM scores candidates in latent space and we run the
   VM only on the top-k. Metric: VM executions to first success.
2. **Partial programs / missing inputs** — holes or unbound inputs mean the program
   *can't* be run as-is; the WM reasons over the distribution of completions. The VM
   baseline literally cannot evaluate a candidate without enumerating completions.
3. **Large edit branching** — k candidate edits per step, depth d → kᵈ executions for
   brute force; the WM prunes to a beam. Metric: executions at matched success vs
   no-WM breadth/best-first search.

## 5. Falsifiable success criteria (M3)

- **Edit-conditioned dynamics beats re-encode-and-roll** on edited-trace exact-match
  at matched compute — or it does not, and we honestly report that re-encoding the
  edited program is sufficient (which would be a finding about R2, not a failure).
- **Planner solves goal tasks with ≥X% fewer real VM executions** than no-WM search
  (breadth-first / best-first) at equal-or-better success rate. Pick X after a
  no-WM-baseline calibration run; do not invent it.
- **The payoff survives R4**: the win must come from the expensive/partial-program
  regimes above, shown explicitly, not from a toy where the VM is free.

## 6. Build order

1. M3-A substrate (`edits.py`, `edit_codec.py`, edit-example generator) + tests. ← in flight
2. Re-encode baseline planner + goal-task generator + no-WM search baseline (calibrate X).
3. Edit-conditioned dynamics (extend the WM) + train on edit pairs.
4. Planner over the learned dynamics; measure executions-saved vs the baseline.
5. Fold edit-counterfactual + executions-saved into ExecWM-Bench.

Sequenced to fail cheap: step 2 alone tests R4 (is there ANY execution-saving to be
had?) before we invest in edit-conditioned dynamics in step 3.

## R4 calibration result (step 2 — BUILT, `execwm/plan/`)

The harness is built and tested (`goal_tasks.py`, `search_baseline.py`, `planner.py`,
`metrics.py`; 13 tests). Calibration on 40 constructed 2-edit goal tasks:

| Method | Solved | Mean real VM executions (solved) |
|---|---|---|
| `vm_search` brute-force (no WM) | 40/40 | ~74 |
| `beam_plan` + cheap (non-VM) scorer | 30/40 | **~1.6 (≈82% fewer)** |
| `beam_plan` + oracle (VM) scorer | 40/40 | ~302 (counts every VM scoring call) |

**R4 verdict: execution-saving IS achievable.** A cheap scorer that defers VM calls
reaches goals with ~1.6 vs ~74 real executions. The oracle contrast proves the saving
must come from a *cheap* scorer (not beam search itself). This sets the exact bar for
**step 3 (learned edit-conditioned dynamics)**: *be the cheap scorer that ALSO handles
control flow*, closing the ~25% success gap the linear heuristic misses (it ignores
loops/branches) while keeping executions far below the no-WM baseline. The learned
latent world model is the natural cheap scorer — this is now a concrete, falsifiable
target, not a hope.
