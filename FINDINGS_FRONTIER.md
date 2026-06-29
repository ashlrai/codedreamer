# The frontier: what is, and isn't, learnable in execution

`FINDINGS_NEUROSYM.md` established the headline — offload arithmetic and the magnitude
wall vanishes (OOD exact-match 0.00 → 0.79), because control flow is (mostly)
magnitude-invariant. This document pins down the *honest* boundary of that claim with
four follow-up experiments (three interpretability/diagnostic analyses on the shipped
checkpoint, one new training run). The picture that emerges is sharper than v1 and, in
one place, corrects it.

## The one-paragraph synthesis

The symbolic ALU offload is **flawless** — across thousands of OOD transitions it never
causes a single state divergence. What degrades out of distribution splits cleanly by
*where the magnitude lives*: **large output magnitude** (a computed result is huge) hurts
**only the digit decode**, which is exactly the offloadable part — so control stays
essentially perfect; **large input magnitude** (operands fed into a step are huge)
degrades the **encoder's** representation, which then nicks control itself. Sign is a
perfectly magnitude-invariant linear direction in the latent; *order/magnitude* is not.
So the frontier is precise: **make the encoder's representation of large operands
magnitude-robust** (and/or offload comparison the way we offload arithmetic). Magnitude-
invariant *comparison* cannot be fully learned from small-magnitude data — that
distribution carries no signal about how values in the hundreds order — so it needs an
architectural prior or a symbolic comparator, not more of the same data.

## 1. Multiplication: the cleanest cut (new training run)

`scripts/neurosym_mul.py` — one model trained on small ADD/SUB/**MUL** programs
(inputs ≤4, products ≤16), wide codec, read out in-distribution vs magnitude-OOD
(inputs ≤40, products ≤1600 — large *outputs*, modest *inputs*):

| split | EM learned | EM digits-oracle | pc acc | written digits | arith digits | cmp result |
|---|---|---|---|---|---|---|
| in-distribution | 0.695 | 0.928 | 0.999 | 0.690 | 0.528 | 0.774 |
| **magnitude-OOD** | **0.056** | **0.892** | **0.998** | 0.363 | 0.165 | 0.706 |

This is the sharpest demonstration of the thesis. With multiplication in the mix the
learned readout collapses *harder* (EM 0.70 → **0.056**; multiplication digits are even
further out of distribution), yet **control flow is essentially untouched (pc 0.999 →
0.998)** and offloading arithmetic recovers almost everything (**0.928 → 0.892**). The
reason control is *more* invariant here than in the ADD/SUB magnitude run (where inputs
reached ~400 and pc fell to 0.986) is the key insight of §3: here the large numbers are
*outputs*, not *inputs*.

## 2. Order-relation probe: sign is free, order is not (`scripts/analysis_order_probe.py`)

Frozen linear probes on the latent, fit on in-distribution states only, scored on
held-out in-dist and magnitude-OOD (`docs/finding_order_probe.md`):

| probe | in-dist | OOD | chance |
|---|---|---|---|
| sign (value < 0) | 1.000 | 1.000 | ~0.74 |
| order (value_i < value_j) | 0.949 | 0.818 | ~0.56 |

**Sign is a perfectly magnitude-invariant linear direction** — so the end-task sign
degradation is a *readout* failure, not a missing representation. **Order degrades (0.95
→ 0.82) but does not collapse**, and it stays above the model's own comparison readout
(0.63) — so the latent carries more usable order structure than the trained comparison
head exploits. Two levers fall out: a stronger comparison head (recover the slack), and
a magnitude-robust encoding (raise the ceiling).

## 3. Divergence cause: input-magnitude corrupts control (`scripts/analysis_divergence_cause.py`)

Running the multi-step executor on 300 OOD programs and classifying the *first* step
each one breaks (`docs/finding_divergence_cause.md`):

- **100% of divergences are wrong-next-pc (control) errors; 0% are value errors.** The
  ALU offload never breaks a single program.
- **In-distribution**, ~100% of failures are at comparison/branch ops (value-dependent
  control) — as expected.
- **Out of distribution, only 50.6% are comparison/branch — 47.2% are wrong-pc errors on
  plain ADD/SUB steps**, a mode that barely exists in-distribution. Large *input*
  operands corrupt the encoding enough that the pc head mispredicts even the trivial
  `pc+1` advance.

This **corrects** the v1 claim that "control is magnitude-invariant." It is invariant to
large *outputs* (§1, pc 0.998) but not fully to large *inputs* (the encoder is the weak
link). The single-step pc=0.986 OOD number hid this: that 1.4% residual concentrates on
high-input-magnitude steps and compounds over a rollout.

## 4. Other OOD axes: "only arithmetic fails" is magnitude-specific (`scripts/analysis_ood_axes.py`)

Holding magnitude small and pushing nesting depth and trace length OOD instead
(`docs/finding_ood_axes.md`):

| split | em_learned | em_digits_oracle | pc | cmp_result |
|---|---|---|---|---|
| in-distribution | 0.690 | 0.900 | 0.986 | 0.681 |
| depth-OOD (nesting 3–4) | 0.643 | 0.827 | 0.901 | 0.704 |
| length-OOD (trace 105–256) | 0.619 | 0.830 | 0.919 | 0.793 |

The structure-vs-digits decomposition holds *directionally* (em_digits_oracle stays high,
~0.83), but the failure mode is different: with magnitude small the digit head doesn't
collapse, so the modest damage lands on **control** (pc 0.99 → ~0.91), not digits. So
"only the arithmetic payload fails" is a statement about the *magnitude* axis
specifically. (Honest confounds: the model was never specialized for these axes, and the
length split leaks some magnitude — read as suggestive, not definitive.)

## 5. A free fix: hand pc-advance back to the ISA (`scripts/structural_pc_executor.py`)

§3 found that ~half the executor's OOD failures were wrong-pc errors on *non-branch*
instructions — but for any non-jump op the next pc is deterministically `pc+1` (the ISA),
not something to predict. So a "structural-pc" executor advances pc structurally for
non-control ops (and to the instruction's target for `JMP`), and reduces the net to the
*one* value-dependent control decision: taken/not-taken on `JZ`/`JNZ`. Values still come
from the ALU (`docs/finding_structural_pc.md`):

| executor | OOD full-program success | OOD per-step exact |
|---|---|---|
| net predicts every pc (current) | 0.400 | 0.641 |
| **structural pc-advance + net decides only branches** | **0.550** | **0.798** |

That recovers ~72% of the OOD per-step degradation **for free** — it was misattributed
control error on steps whose pc was never in question. The residual is genuine and lands
exactly on the frontier: OOD branch-decision accuracy is 0.86 (vs 0.91 in-dist), and
snapping the net's pc to {pc+1, target} barely helps (+0.02), because when the net gets
taken/not-taken wrong it's failing to compare a value in the hundreds to zero — the §2
order wall, not structure.

## 6. Measuring progress (`scripts/frontier_benchmark.py`, `docs/FRONTIER_CHALLENGE.md`)

One command scores any checkpoint on the canonical frontier metrics (in-dist vs
magnitude-OOD) and prints the three "numbers to beat": OOD `em_digits_oracle` ≈ 0.79,
OOD `cmp_result` ≈ 0.63, OOD executor `full_trajectory_success` ≈ 0.40 (now 0.55 with the
structural-pc executor). The rule: improve these *without* training on large magnitude
(that would be cheating — the model must learn from small-magnitude data and generalize).

(Engineering note: the executor's grading is now robust to post-divergence ALU values
that fall outside the codec range — they score as non-matches instead of raising
`EncodeError`, `execwm/eval/neurosym_exec.py:_exact`.)

## The v2 target (the contributor challenge, now precise)

1. **Magnitude-invariant operand encoding.** The encoder's representation of large
   *inputs* is the single root cause behind both the order-probe degradation (§2) and the
   arithmetic-step control errors (§3). A value encoding whose order/comparison structure
   is robust beyond the trained magnitude range — by architectural prior (e.g. an explicit
   MSB-first digit comparator), not by data — is the highest-leverage fix.
2. **A stronger comparison readout**, which the probe (§2) shows is leaving recoverable
   order signal on the table.
3. **Or offload comparison** the way arithmetic is offloaded — clean, but it moves more
   work to the symbolic side (and toward "you are running the VM").

## 7. v2 attempt: a fixed positional encoding does NOT close the gap (clean negative)

We tested the simplest form of target #1 directly. Hypothesis: with *learned* per-position
digit embeddings, the high-order positions are always zero in small-magnitude training and
so are undertrained — making the encoding of large values OOD. Swap them for a **fixed
sinusoidal position encoding** (never OOD) and the encoding of a large value becomes a
clean composition of (well-trained digit) + (fixed position). `scripts/neurosym_v2_encoding.py`
trains a `fixed_pos=True` model on the *same* small-magnitude slice:

| OOD metric | baseline (learned pos) | v2 (fixed pos) | Δ |
|---|---|---|---|
| em_digits_oracle | 0.790 | 0.742 | **−0.048** |
| cmp_result | 0.626 | 0.576 | **−0.050** |
| written_sign | 0.798 | 0.790 | −0.008 |
| pc | 0.986 | 0.993 | +0.007 |

**It does not help — it slightly hurts.** This is an important negative: the gap is *not*
primarily an undertrained-position-embedding artifact. The deeper cause stands — the
*dynamics* never observes a nonzero high-order digit during small-magnitude training, so it
never learns what large-magnitude values *mean* for comparison. Fixing the static encoding
of positions cannot supply that missing training signal. **Magnitude-invariant comparison
is not learnable from small-magnitude data by re-encoding alone.**

So the live options narrow to: a **structural** comparator (compute order MSB-first as a
fixed differentiable circuit — magnitude-invariant *by construction*, not learned), a
stronger comparison **readout** (target #2 — recovers the slack the probe in §2 shows, but
cannot raise the ceiling), or **offloading comparison** to the VM the way arithmetic is
offloaded (target #3 — clean, moves work to the symbolic side). Target #1 in its *learned*
form is now ruled out.

Every number here is graded against the VM oracle; the v1 correction in §3 and this v2
negative are reported as such.

## 8. The prior that works: an MSB-first comparator is magnitude-invariant (POSITIVE)

§7 ruled out re-encoding; the remaining shot was a *structural* comparator. We built one
(`execwm/model/comparator.py`) and tested it in isolation (`scripts/comparator_probe.py`):
predict order(a, b) ∈ {<, =, >} from (sign, MSB-first digits), trained ONLY on |value| ≤ 30,
tested on |value| ∈ [300, 800] — same codec width.

| comparator | in-dist acc | OOD acc |
|---|---|---|
| **`DigitComparator` (MSB-first prior)** | 1.000 | **1.000** |
| `PlainComparator` (MLP, no prior) | 0.995 | 0.748 |

**The frontier is closable with a learned model.** A small *learned* per-position cell
(scores `<`/`=`/`>` for one digit pair) plus a *fixed* lexicographic combiner (the most-
significant non-equal position decides) is **perfectly magnitude-invariant** — 1.00 → 1.00
across an 10–25× shift — while a plain MLP collapses to 0.748 (right where the world model's
own order/comparison sits, §2). The invariance is not memorized: it comes from
**weight-sharing across digit positions** (the cell is trained on the always-populated
low-order positions and transfers verbatim to the high-order positions that only light up
out of distribution) plus the fixed reduction. This is exactly the "architectural prior, not
data" that §2/§7 pointed to — and it is *learned*, not offloaded (the cell could fail to
learn `<`/`=`/`>`, but it learns it from low-magnitude digits and generalizes).

**Honest scope.** This solves the comparison *sub-problem* in isolation. The clear next step
(deliberately not rushed — it is a multi-call-site change to the model's forward) is to wire
`DigitComparator` into the world model's dynamics for comparison/branch instructions and
measure the end-to-end lift on OOD `cmp_result`, branch accuracy, and executor full-program
success. The mechanism is now proven; the integration is a well-scoped contributor task
(`docs/FRONTIER_CHALLENGE.md`).
