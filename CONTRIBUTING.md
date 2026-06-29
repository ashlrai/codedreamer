# Contributing to CodeDreamer

Thanks for being here. CodeDreamer is a small, honest research repo: one ~10M-param
grounded latent world model of code execution, every prediction graded against a real
interpreter. It is deliberately laptop-sized and CPU-runnable, so you can reproduce the
headline result and start hacking on the open frontier in an afternoon.

The Python import package is `execwm`; the public brand is **CodeDreamer**.

---

## 1. Set up and run the test suite

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r demo/requirements.txt        # torch, numpy, gradio — enough to run + test
python -m pytest -q                          # 111 tests
```

The suite (111 tests) covers the VM + tracer, the lossless state/action codecs, the
disjoint OOD splits, the model and grounding heads, the neurosymbolic readout, the
executor, and the planner. **A green suite is the contract.** Run it before and after
any change; if you add behavior, add a test that grades it against the VM oracle (see
§5).

The interactive demo is the fastest way to *see* the result:

```bash
PYTHONPATH=. python demo/app.py              # open the printed local URL; CPU is fine
```

---

## 2. Reproduce the headline result

One frozen model, trained only on small values (≤ ~30) with a codec wide enough to
represent big numbers, read out two ways on an in-distribution split and a 10–25×
magnitude-OOD split.

**Single-step readout breakdown** (trains a fresh model, then runs the field breakdown):

```bash
PYTHONPATH=. python scripts/neurosym_spike.py --steps 1500
```

On MPS/CPU this writes `artifacts/neurosym_model.pt` and prints, at n ≈ 10k transitions
per split:

| split | EM (learned) | EM (arithmetic offloaded) | next-pc acc | branch acc |
|---|---|---|---|---|
| in-distribution (val ≤ 30) | 0.721 | 0.904 | 0.999 | 0.996 |
| magnitude-OOD (val 300–800) | **0.000** | **0.790** | 0.986 | 0.989 |

The single load-bearing claim: whole-state exact-match collapses to **0.000** out of
distribution with the learned digit readout, but the *same weights* reach **0.790** when
arithmetic is offloaded to a symbolic ALU — and control flow barely moves.

**Whole-program executor** (reuses the checkpoint above):

```bash
PYTHONPATH=. python scripts/neurosym_exec_eval.py
```

This runs the neurosymbolic executor (net drives control flow, ALU computes values)
across in-distribution and magnitude-OOD programs. A pure-net executor scores 0.00 at
OOD magnitude; this measures how far the neurosymbolic split gets instead.

Numbers can drift slightly with seed/steps — that is expected. If your reproduction is
*qualitatively* different (e.g. offloaded OOD EM near zero), please open an issue with
your environment and the printed table.

---

## 3. The Open Challenge — a magnitude-invariant abstract interpreter

This is the place to push the frontier.

`EM (arithmetic offloaded)` is **0.790, not 1.0**. The remaining gap is **not** digits —
a symbolic ALU already computes those for free. The gap is **value-derived predicates**:
properties the net must still decide from the (out-of-distribution) operands.

| field | in-distribution | magnitude-OOD |
|---|---|---|
| comparison outcome (`cmp_result`) | 0.788 | **0.626** |
| written sign (`written_sign`) | 0.881 | **0.798** |
| arithmetic digits (`arith_digits`) | 0.672 | 0.103 (offloaded away) |

Comparison outcomes and signs degrade because they depend on *relative* magnitude, which
the net tracks only approximately once values leave the training regime. Over a long
rollout those branch errors compound, which is what caps the whole-program executor.

**The precise open problem:** represent comparison-relevant value properties (sign,
ordering) in a **magnitude-invariant** way — i.e. learn an *abstract interpreter* that
decides `a < b`, `sign(a−b)`, etc. without needing to realize the magnitudes. If you
crack this, the offloaded EM and the executor's OOD success should both rise toward 1.0
without retraining the arithmetic.

Where to work:

- **Measurement:** `execwm/eval/neurosym.py` → `field_breakdown(...)`. It already isolates
  `cmp_result`, `written_sign`, `arith_digits`, `branch_pc`, and the learned-vs-offloaded
  exact-match (`em_learned` vs `em_digits_oracle`). Any new approach should move these
  numbers, measured here, on the magnitude-OOD split.
- **Context:** [`FINDINGS_NEUROSYM.md`](FINDINGS_NEUROSYM.md), §"The honest residual (the
  frontier)" — the full method, the exact table, and the honest caveats.

---

## 4. Adding a new model variant or grounding head

The model lives in `execwm/model/world_model.py`. The pieces you will touch:

- `ModelConfig` — dims and slot bookkeeping (`num_slots = num_regs + num_cells + 2`).
- `StateEncoder` → `LatentDynamics` — the slotted latent and the action-conditioned
  dynamics. New *architecture* variants go here (or as a sibling module).
- `GroundingHeads` — the shallow per-slot decoders (`reg_type`, `reg_sign`,
  `reg_digits`, `heap_*`, `pc`, `halted`, `error`). To add a **new state field or
  predicate** (e.g. an abstract sign/order head for §3), add a `nn.Linear` here and emit
  it from `forward`, then teach `grounding_loss` / `field_correct` / `exact_match` about
  it so it is trained and graded.

To make a new variant or head show up in the headline metrics, wire it into the readout:
`execwm/eval/neurosym.py` → `field_breakdown` (single-step) and
`execwm/eval/neurosym_exec.py` (whole-program executor). Add accumulator keys the same
way `cmp_result` and `branch_pc` are added. Keep the learned-vs-offloaded split intact so
your variant is comparable to the baseline.

For a brand-new model class, mirror `GroundedLatentWM`: encoder + action encoder +
dynamics + heads, with the EMA `target_encoder` for the JEPA loss. Keep linear probes
decoding each field at ≥0.99 — interpretability is part of the claim, not an afterthought.

---

## 5. Code style and the grading rule

- **Type hints** on public functions and `__init__` signatures; `from __future__ import
  annotations` at the top of new modules (match the existing files).
- **Concise docstrings** — one short paragraph stating what the module/function does and
  any non-obvious design choice (see `world_model.py` and `neurosym.py` for the tone).
- Stay **MPS-safe**: prefer explicit comparisons over ops like `torch.isin` (see the
  `_in` helper in `neurosym.py` and `valued_mask` in `world_model.py`).
- Keep it dependency-light. No new heavyweight deps without a reason.

**The grading rule (non-negotiable):** ground truth is always one VM run away, so every
claim must be graded against the VM oracle — never asserted from intuition. Report
**negative results honestly**: a clean negative (e.g. the magnitude curriculum failing at
EM ≈ 0.003, `FINDINGS_M3.md` §5) is a contribution. Put new findings in a `FINDINGS_*.md`
with the method, the exact numbers, the `n`, and the caveats, the same way the existing
ones do.

Open an issue or a PR — small, well-tested, honestly-measured changes are exactly what
this repo wants.
