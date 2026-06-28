# Why the neurosymbolic executor degrades OOD: it is 100% control-flow, and the OOD delta is sequential-pc failure on arithmetic steps — *not* branch errors

**One-line result:** Every first-divergence of the free-running neurosymbolic executor
is a **wrong-next-pc (control) error** — the symbolic ALU never causes a divergence.
But the *hypothesis* that OOD failures concentrate on **comparison/branch** instructions
is **refuted**: in-distribution they do (100% branch), yet the entire in-dist→OOD
degradation is driven by a **new** failure mode — the net mispredicting the trivial
`pc+1` advance on **plain arithmetic (ADD/SUB) instructions** when operand magnitudes
are out of distribution. Branch errors stay essentially flat (106 → 91); arithmetic-step
control errors go 0 → 85.

## Setup

- Existing checkpoint `artifacts/neurosym_model.pt`, loaded on **CPU** (GPU was in use).
- In-distribution set = the checkpoint's own `spec` (`max_const=5, max_input_val=5`,
  values ≤ ~30). Magnitude-OOD set = `replace(spec, max_const=400, max_input_val=400)`
  (values ~300–800). 300 programs each, graded against the VM oracle, exactly as the
  executor already is.
- For every program that is not `full_exact`, take the **first** `StepRecord` with
  `state_exact == False`, identify the op of the instruction executed at that step
  (`ex.trace.program[record.pc].op`), classify it, and split control-error
  (`control_ok == False`) vs value/flag-error (`control_ok == True` but state wrong).
- Script: `scripts/analysis_divergence_cause.py` (runnable, deterministic — argmax + fixed
  seeds). Full-program success was 64.7% in-dist / 40.0% OOD, consistent with the
  0.70 / 0.39 in `FINDINGS_NEUROSYM.md`.

> Implementation note: the script carries a faithful copy of `neurosym_execute` whose
> only change is to treat a post-divergence `EncodeError` (the ALU computing a value
> beyond the 4-digit codec range *after* control has already gone wrong) as
> `state_exact = False` rather than aborting the rollout. No existing file was modified;
> this changes no grading outcome (an out-of-range state can never match in-range ground
> truth) — it only lets an already-diverged program finish so its first divergence is recorded.

## First-divergence op category (share of failures)

| category    | in-dist (106 fail) | OOD (180 fail) |
|-------------|-------------------:|---------------:|
| comparison  |  0.9% (1)          |  1.1% (2)      |
| jump/branch | 99.1% (105)        | 49.4% (89)     |
| arithmetic  |  0.0% (0)          | 47.2% (85)     |
| movement    |  0.0% (0)          |  1.1% (2)      |
| heap        |  0.0% (0)          |  1.1% (2)      |
| other       |  0.0% (0)          |  0.0% (0)      |
| **cmp+branch** | **100.0% (106)** | **50.6% (91)** |

Raw op at first divergence — **in-dist:** JZ 65.1%, JMP 34.0%, EQ 0.9%.
**OOD:** JZ 37.8%, ADD 25.0%, SUB 22.2%, JMP 11.7%, CONST/STORE/GT/GE ≤ 1.1% each.

## Control-vs-value split

| | in-dist | OOD |
|---|---:|---:|
| control error (wrong next pc) | **100.0%** | **100.0%** |
| value/flag error (pc ok, state wrong) | 0.0% | 0.0% |

**100% of divergences in both splits are wrong-next-pc errors.** Because the symbolic ALU
supplies exact register/heap values and types, the only thing the net can get wrong is
`pc`/`halted`/`error`, and in practice it is always `pc`. The ALU offload is perfect —
zero divergences are caused by arithmetic value error or by a mispredicted flag.

## Conclusion (honest)

**Does the comparison/branch hypothesis hold? Partially, then no.**

1. **Confirmed — the failure is control, never value.** Every first divergence is the
   net picking the wrong next pc; the ALU never produces a wrong value or flag. The
   executor's residual error is entirely in the learned next-pc head.

2. **Confirmed in-distribution.** When the model fails on in-dist programs, it fails
   essentially 100% at jump/branch instructions (JZ/JMP) — classic value-dependent
   control. Straight-line `pc+1` advancement is never wrong in-dist.

3. **Refuted for the OOD degradation.** At magnitude-OOD, comparison+branch accounts for
   only **50.6%** of first divergences, and an equally large **47.2%** are control errors
   on **plain arithmetic (ADD/SUB)** instructions — a category the hypothesis explicitly
   excludes. In absolute terms branch errors are flat-to-slightly-down (106 → 91) while
   arithmetic-step control errors rise 0 → 85, so the **entire** in-dist→OOD increase in
   failures (+74) is the new arithmetic-step mode. The OOD degradation is therefore
   **not** value-dependent control at comparisons/branches; it is the next-pc head losing
   its grip on **sequential** advancement when fed OOD-magnitude operands.

4. **Mechanism.** Large operand values push the latent off-distribution enough that the
   pc head mispredicts even the trivial `pc+1` after an arithmetic op. This nuances
   `FINDINGS_NEUROSYM.md`'s "control is magnitude-invariant (next-pc 0.999 → 0.986)":
   that ~1.4% per-step OOD pc-error is real, lands disproportionately on arithmetic
   steps, and under free-running compounding it — not branch error — drives most of the
   multi-step collapse.

**Implication for v2.** The contributor target sharpens from "magnitude-invariant value
*comparison*" to "**magnitude-invariant latent encoding of operands**": the pc head is
fine on in-dist latents but is corrupted by OOD-magnitude inputs even on non-branching
instructions. Making the *encoder's* representation of large values magnitude-invariant
(so straight-line control survives) is at least as important as fixing branch/comparison
prediction.

## Reproduce

```
PYTHONPATH=. CUDA_VISIBLE_DEVICES="" python scripts/analysis_divergence_cause.py --n 300
```
