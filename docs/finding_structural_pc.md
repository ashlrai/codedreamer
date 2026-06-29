# Finding: the structural-pc executor — most of the OOD control gap was spurious

`scripts/structural_pc_executor.py` tests the §3 prediction from `FINDINGS_FRONTIER.md`
head-on. §3 showed that out of distribution, ~47% of the neurosymbolic executor's
first-divergences are *wrong-next-pc errors on plain ADD/SUB steps* — a failure mode that
barely exists in-distribution. Those errors are **spurious**: by the ISA
(`execwm/substrate/vm.py`), every non-jump instruction deterministically advances
`pc → pc+1`. There is nothing to predict; the net only mispredicts it because large input
operands corrupt the encoder. So we should never have asked the net for it.

## The variant

Same frozen checkpoint (`artifacts/neurosym_model.pt`), same symbolic ALU for values. The
only change is *who decides the next pc*:

| instruction class | current executor | **structural-pc executor** |
|---|---|---|
| non-control (CONST/MOV/ADD/SUB/…/LOAD/STORE/HALT) | net predicts pc | **ISA: `pc+1` (or halt/trap), net not consulted** |
| JMP | net predicts pc | **target from instruction, net not consulted** |
| JZ / JNZ (value-dependent) | net predicts pc | **net decides branch; argmax snapped to nearer of {pc+1, target}** |

So the net is reduced to the one genuinely-learned thing: the taken/not-taken call on
`JZ`/`JNZ`, which hinges on comparing a value to zero — the value-comparison frontier.

## Results (300 programs/split, CPU, graded against the VM oracle via `scodec.exact_match`)

Branch-decision accuracy is measured **teacher-forced** at every true `JZ`/`JNZ` step
(1893 in-dist / 1918 OOD), so the free-argmax rule (current) and the snap rule
(structural) are compared on identical steps, uncontaminated by rollout divergence.
Full-trajectory success and per-step exact come from the free-running rollouts.

### Current executor (net predicts the next pc at every step)

| split | full-traj success | per-step exact | branch-decision acc | mean horizon | n |
|---|---|---|---|---|---|
| in-distribution (val ≤ ~30) | 0.647 | 0.860 | 0.890 | 37.5 | 300 |
| magnitude-OOD (val ~300–800) | 0.400 | 0.641 | 0.842 | 28.9 | 300 |

### Structural-pc executor (ISA advances non-branch pc; net only decides JZ/JNZ)

| split | full-traj success | per-step exact | branch-decision acc | mean horizon | n |
|---|---|---|---|---|---|
| in-distribution (val ≤ ~30) | 0.713 | 0.889 | 0.908 | 39.0 | 300 |
| magnitude-OOD (val ~300–800) | 0.550 | 0.798 | 0.861 | 36.0 | 300 |

### OOD deltas (structural − current)

| metric | current OOD | structural OOD | delta |
|---|---|---|---|
| full-trajectory success | 0.400 | 0.550 | **+0.150** |
| per-step exact | 0.641 | 0.798 | **+0.157** |
| branch-decision acc (free → snap) | 0.842 | 0.861 | +0.019 |
| mean exact horizon | 28.9 | 36.0 | +7.1 |

## Conclusion (honest)

Most of the executor's out-of-distribution control gap was **spurious** and is recovered
for free by handing the deterministic pc-advance back to the ISA. The current executor's
per-step exactness fell 0.860 → 0.641 from in-dist to OOD (a 0.219 drop); structural
pc-advance recovers 0.157 of that — **~72% of the OOD per-step degradation was wrong-pc
error on instructions whose next pc was never in question.** Full-trajectory success at
OOD rises 0.400 → 0.550 (+0.150, ~61% of the current executor's in-dist→OOD full-traj
drop) and the mean exact horizon lengthens by ~7 steps. This is a direct, quantitative
confirmation of `FINDINGS_FRONTIER.md` §3: the OOD encoder corruption was leaking into the
trivial `pc+1` prediction, and that leak is pure structure that the ISA already pins down.

The residual is exactly where it should be: the **value-comparison frontier**. Structural
advance does not, and should not, touch the `JZ`/`JNZ` decision — and that decision is what
still degrades OOD. Branch-decision accuracy drops 0.908 → 0.861 (−0.047) from in-dist to
OOD, and snapping the net's prediction to the legal branch targets barely helps it
(free 0.842 → snap 0.861, only +0.019 OOD). That small snap gain is telling: it means most
of the current executor's OOD branch errors were *not* the net landing near the wrong
branch — when the net gets the taken/not-taken call wrong, snapping faithfully reproduces
the wrong branch. So the surviving ~14% OOD branch error is genuine: the net cannot
reliably compare values in the hundreds to zero, having only ever seen values ≤ ~30. This
is the same wall as §2 (magnitude-invariant comparison is not learnable from
small-magnitude data) and bounds the structural executor: even with perfect structure and
a perfect ALU, full-trajectory success at OOD is capped at 0.550 because branch errors
compound over the rollout (note the structural executor's own in-dist→OOD full-traj drop,
0.713 → 0.550, is now almost entirely branch-driven).

**Net:** offloading the ISA's pc-advance is a clean, free win that removes a real and
previously-misattributed chunk of the OOD gap (~72% of per-step error). What remains is
not spurious — it is the value comparison inside `JZ`/`JNZ`, which needs an architectural
prior or a symbolic comparator (`FINDINGS_FRONTIER.md` §2/v2-target), not more structure.
