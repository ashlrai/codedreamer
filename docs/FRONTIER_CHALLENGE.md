# The Frontier Challenge

This is the open challenge for `execwm`: **push the out-of-distribution (OOD) frontier of
a grounded latent world model of computation — using only small-magnitude training data.**

The shipped checkpoint (`artifacts/neurosym_model.pt`) was trained on small-magnitude
programs (operand/constant values ≤ ~5, reachable states ≤ ~30) with a codec wide enough
to *represent* large numbers (4 digits → up to 9999). Read out two ways — pure-net vs
arithmetic-offloaded — and scored in-distribution vs magnitude-OOD, it exposes a precise
boundary between what execution structure a net learns to generalize and what it does not.
`FINDINGS_FRONTIER.md` is the honest writeup; this document is the *contributor-facing*
version: the goal, how to measure, the numbers to beat, and where to attack.

## The goal

Raise the **OOD** scorecard numbers — chiefly `em_digits_oracle`, `cmp_result`, and the
executor's `full_trajectory_success` — **without training on large-magnitude data.**

> The model must learn the *structure* of execution (which slot is written, the op type,
> the sign, the next pc, branch outcomes, comparison results) from small-magnitude
> programs only, and have that structure generalize to operands/values in the hundreds.

**What counts as cheating:** training (or fine-tuning, or curriculum-extending) on
large-magnitude programs. That trivially closes the gap by erasing the distribution
shift — it answers a different, uninteresting question. The interesting claim is about
*generalization of structure*, so the training distribution must stay small-magnitude.
Architectural priors, better readout heads, smarter encodings, and symbolic offload are
all fair game; large-magnitude *data* is not.

Everything is graded against the VM oracle. Arithmetic offload (the `em_digits_oracle`
readout and the executor's ALU) is **not** "just running the VM": the net still owns all
control and structure (pc, flags, types, signs, which slot changed, branch resolution);
only the exact digit payload of a computed register is supplied by a symbolic ALU. The
remaining OOD gap is therefore a *structure* gap, which is exactly what this challenge is
about.

## How to measure

One command produces the canonical scorecard:

```bash
PYTHONPATH=. python scripts/frontier_benchmark.py [--ckpt artifacts/neurosym_model.pt]
```

It loads any slotted world-model checkpoint (CPU only — a GPU/MPS training job may be
running), builds an in-distribution eval set from the checkpoint's own training spec and a
magnitude-OOD set (`replace(spec, max_const=400, max_input_val=400)`, ~300 episodes each),
and prints one scorecard table plus the three "numbers to beat" read live from the run.

Useful flags: `--n` (episodes per split, default 300), `--seed`, `--max-len` (episode
length cap for the single-step breakdown, default 18).

Scorecard columns (in-dist vs OOD):

| metric | meaning |
|---|---|
| `em_learned` | whole-state exact match, every field decoded by the net (collapses to ~0 OOD) |
| `em_digits_oracle` | **headline** — same predictions, digit payload supplied by a perfect ALU; isolates structural correctness |
| `pc` | next-pc accuracy (single step) |
| `cmp_result` | correctness of comparison-op results (value-dependent control) |
| `written_sign` | sign of the written register (a magnitude-invariant direction) |
| `full_traj_success` | executor: fraction of whole programs exact end-to-end (net-control + ALU-values) |
| `control_acc` | executor: mean per-step next-pc accuracy over the rollout |

To move the frontier: train your variant on the **same small-magnitude spec**, save a
checkpoint, and run the benchmark against it. A win is a strictly higher OOD number with
no large-magnitude training.

## The numbers to beat

Read live from the shipped checkpoint by `scripts/frontier_benchmark.py` (so always
re-confirm from your own run — these are the current reference values):

| OOD metric | current |
|---|---|
| `em_digits_oracle` (headline — arithmetic offloaded) | **~0.79** |
| `cmp_result` (comparison-result correctness) | **~0.63** |
| executor `full_trajectory_success` (whole-program exactness) | **~0.39** |

(In-distribution these are ~0.90 / ~0.79 / ~0.65 respectively, so each is the slack a
better model could recover — and the in-dist numbers are themselves ceilings worth
raising.)

## Where to attack (from `FINDINGS_FRONTIER.md`)

The findings localize the OOD failure precisely, which points at three concrete levers:

1. **Magnitude-invariant operand encoding (highest leverage).** The root cause behind
   both the order-probe degradation and the wrong-pc errors on plain ADD/SUB steps is the
   **encoder's** representation of large *inputs*. Large *outputs* only hurt the
   (offloadable) digit decode; large *inputs* corrupt the latent enough to nick control
   itself. Give the encoder a value encoding whose order/comparison structure is robust
   beyond the trained magnitude range **by architectural prior** (e.g. an explicit
   MSB-first digit comparator), not by data. This should lift `cmp_result`, `pc`, and the
   executor's `control_acc` / `full_trajectory_success` together.

2. **A stronger comparison readout.** A frozen linear *order* probe on the latent scores
   ~0.82 OOD while the trained `cmp_result` head sits at ~0.63 — so there is recoverable
   order signal the current head leaves on the table. A better comparison head (or one
   that consumes the order direction the probe finds) should narrow that gap.

3. **Offload comparison** the way arithmetic is offloaded. Clean and effective, but it
   moves more work onto the symbolic side (and toward "you are running the VM"), so it is
   the least preferred of the three — report it honestly if you take this route.

Be precise and honest in any results you report: state the training spec (to show no
large-magnitude leakage), the split definitions, the seed/`--n`, and grade against the VM
oracle exactly as the benchmark does.
