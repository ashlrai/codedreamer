# CodeDreamer Architecture

CodeDreamer is a **grounded latent world model of code execution**. A custom DSL program
is compiled to bytecode and run on an instrumented VM; a per-instruction tracer emits the
exact machine state at every step (free, perfect ground truth). Those states/actions are
encoded losslessly to tensors, the model predicts the next state *in a slotted latent
space*, and shallow per-slot decoders read the prediction back into a symbolic state that
is graded against the VM.

The result the repo is built around: the model's *control/structure* prediction is
magnitude-invariant and learnable, while *arithmetic* is the only component that breaks
out of distribution — so at readout time we split the work between the **net** (control +
structure) and a **symbolic ALU** (values).

---

## Data flow

```
  DSL program (execwm/substrate/dsl.py, generators.py)
        │  compile
        ▼
  Bytecode VM + per-instruction tracer            ── GROUND TRUTH ──┐
  (execwm/substrate/vm.py)                                          │
        │  emits (state_t, action_t, state_t+1) per step           │
        ▼                                                          (grade
  Lossless codecs                                                  exact-
  state_codec.py  ·  action_codec.py  (execwm/data/)               match
        │  sign + MSB-first digits, op/operands → tensors           against
        ▼                                                          truth)
  StateEncoder        slot tokens: 1/register, 1/heap-cell,         │
  (world_model.py)    + pc slot + flags slot → Transformer          │
        │             per-slot latent  z : (B, S, d=256)            │
        ▼                                                           │
  ActionEncoder ──► LatentDynamics    ẑ_{t+1} = z + g(z, a)         │
  (world_model.py)  action injected into every slot, attend,        │
        │           deterministic, residual                         │
        ▼                                                           │
  GroundingHeads      shallow per-slot linear decoders              │
  (world_model.py)    ẑ → {reg_type, reg_sign, reg_digits,          │
        │              heap_*, pc, halted, error}                   │
        ▼                                                           │
  Neurosymbolic readout split  (execwm/eval/neurosym.py,            │
  neurosym_exec.py):                                                │
        ├─ NET  → control + structure (pc, type, sign, which slot ──┘
        │         changed, comparison/branch outcomes)
        └─ ALU  → arithmetic (the exact digit payload of registers/heap)
```

The split is an **eval-time intervention on one frozen model** — no retraining, no
architecture change. `em_learned` decodes every field from the net; `em_digits_oracle`
keeps the net's structural prediction but lets a perfect ALU supply the digit payload
(`_oracle_digit_logits` in `neurosym.py`). The gap between them localizes the magnitude
wall to the digit head.

---

## Key design choices

- **Slotted latent, not a pooled vector.** The latent is one vector per state *slot*
  (one per register, one per heap cell, plus a pc slot and a flags slot:
  `num_slots = num_regs + num_cells + 2`). Packing ~30 registers' exact integers into a
  single vector is information-bottlenecked; per-slot latents make exact decode tractable
  and keep the Othello-GPT-style interpretability claim (an unchanged register is a
  near-identity copy).
- **Compositional value embedding.** `ValueEmbedding` embeds a signed integer as a shared
  per-digit-value table + a per-position table summed over digits + a sign embedding,
  which helps the magnitude-OOD axis on the *input* side.
- **Deterministic dynamics.** Transitions are deterministic, so `LatentDynamics` is a
  residual slot-Transformer with the action projected into every slot — no stochastic
  latent.
- **Shallow grounding heads.** The decoders are single linears per field (register/heap
  heads shared across their slots). Keeping them shallow is what makes "the wall lives in
  the digit head" a meaningful, testable claim, and keeps frozen linear probes honest.
- **Training losses.** Grounded decode at t and t+1, a JEPA feature-prediction loss
  against an EMA `target_encoder` (VICReg-regularized), and a curriculum rollout that
  unrolls the dynamics in latent space while keeping the decoded state exact.

---

## Module map

```
execwm/
  substrate/   vm.py            bytecode VM + per-instruction tracer (ground truth)
               dsl.py           the DSL surface
               generators.py    program sampling across the 5 OOD axes
               edits.py         program edits (edit-as-action substrate)
  data/        state_codec.py   lossless state ↔ tensor (sign + MSB-first digits)
               action_codec.py  lossless instruction ↔ tensor (op + operands)
               dataset.py       trace → transition examples
               torch_data.py    EpisodeDataset, collate, flatten_time
               edit_codec.py    edit ↔ tensor
               edit_dataset.py  edit-conditioned dataset
  model/       world_model.py   ModelConfig · ValueEmbedding · StateEncoder ·
                                ActionEncoder · LatentDynamics · GroundingHeads ·
                                GroundedLatentWM · grounding_loss / exact_match
               edit_dynamics.py edit-conditioned dynamics
               delta.py         delta-prediction variant
               arith.py         arithmetic head experiments
               token_baseline.py matched token-space trace predictor
  eval/        neurosym.py      field_breakdown — the learned-vs-offloaded readout split
               neurosym_exec.py the neurosymbolic whole-program executor
               demo_backend.py  DemoEngine + render helpers for the Gradio demo
               probes.py        frozen linear probes (interpretability)
               counterfactual.py do(register) / do(replace-instruction) causal eval
               ood_eval.py      the 5 OOD axes
               execwm_bench.py  ExecWM-Bench (causal + OOD benchmark)
               checkpoint.py · report.py · token_eval.py
  train/       train_m1.py      train the grounded latent WM
               train_edit.py · curriculum.py · train_arith.py ·
               train_arith_curriculum.py · train_delta.py · train_token.py
  plan/        planner.py · divergence_planner.py · goal_tasks.py · wm_scorer.py ·
               partial_search.py · partial_tasks.py · search_baseline.py · metrics.py
demo/          app.py           the magnitude-slider Gradio demo
               requirements.txt
scripts/       neurosym_spike.py · neurosym_exec_eval.py · run_execwm_bench.py · …
artifacts/     neurosym_model.pt  the trained checkpoint the demo loads
```

(Every path above was verified against the tree; `eval/demo_backend.py` is where the demo
gets its `DemoEngine`, `render_trace_html`, and `summary_md`.)

---

## The five OOD axes

Splits are disjoint by construction (a numeric-gap assertion or a structural inverse),
sampled by `execwm/substrate/generators.py` and evaluated by `execwm/eval/ood_eval.py`:
numeric magnitude, trace length, nesting depth, program size, and compositional
(held-out operator/context pairings). The headline result lives on the **numeric
magnitude** axis.
