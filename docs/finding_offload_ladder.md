# The offload ladder: the OOD gap decomposes entirely into offloadable pieces

`scripts/offload_ladder.py` (run in-process, CPU only — a GPU/MPS training job was live)

## What this is

FINDINGS_FRONTIER.md §5 took the neurosymbolic executor from "net predicts every pc" to
"structural pc-advance, net decides only the JZ/JNZ branch," and recovered most of the
magnitude-OOD degradation for free. This finding walks **one more rung** to the bottom of
the decomposition: it also resolves the branch *structurally*, leaving the net responsible
for nothing, and measures the same in-distribution and magnitude-OOD program sets (300
each; OOD = `replace(spec, max_const=400, max_input_val=400)`).

Three executor variants, identical grading (every produced state vs the VM oracle via
`scodec.exact_match`, through the robust `_exact` guard):

1. **baseline** — the net predicts the next pc every step (shipped `neurosym_execute`);
   values from the symbolic ALU.
2. **structural-pc** — the ISA advances the pc for every non-branch op (and to the JMP
   target); the net is consulted only for the JZ/JNZ taken/not-taken decision. (Reused
   `structural_pc_execute` from `scripts/structural_pc_executor.py`.)
3. **structural-pc + structural-branch (NEW)** — additionally resolve JZ/JNZ from the
   concrete ALU operand value: JZ taken iff `operand == 0`, JNZ iff `operand != 0`,
   computed from the current concrete state. This comparison is magnitude-invariant, so
   **the net is consulted for nothing**. Values still come from the ALU.

## Results (actual numbers, n=300 per split, `artifacts/neurosym_model.pt`)

In-distribution (operand values ≤ ~5):

| rung | full-traj success | per-step exact | mean horizon | n |
|---|---|---|---|---|
| 1. baseline (net predicts every pc) | 0.647 | 0.860 | 37.5 | 300 |
| 2. structural-pc (net decides JZ/JNZ only) | 0.713 | 0.889 | 39.0 | 300 |
| 3. structural-pc + structural-branch (no net) | **1.000** | **1.000** | 43.9 | 300 |

Magnitude-OOD (operand values ~300–800):

| rung | full-traj success | per-step exact | mean horizon | n |
|---|---|---|---|---|
| 1. baseline (net predicts every pc) | 0.400 | 0.641 | 28.9 | 300 |
| 2. structural-pc (net decides JZ/JNZ only) | 0.550 | 0.798 | 36.0 | 300 |
| 3. structural-pc + structural-branch (no net) | **1.000** | **1.000** | 45.2 | 300 |

OOD full-trajectory success climbs `0.400 → 0.550 → 1.000`: the structural pc-advance buys
`+0.150` (the §5 result, reproduced), and resolving the branch structurally buys the
remaining `+0.450`, closing the gap completely. (Rungs 1–2 match §5's `0.400 / 0.550` and
`0.641 / 0.798` exactly; one OOD program tripped an out-of-range encode in the shipped
engine and was scored via the guarded replica, which changes no grading outcome.)

## The honest conclusion

Rung 3 is, **by construction, VM-equivalent**. It offloads all three pieces of a step:
arithmetic (the ALU), the structural pc-advance (the ISA — `pc+1` / JMP target / halt-trap),
and the one value-dependent control decision (a single `operand == 0` comparison on the
concrete state). Nothing is left for the net to do, so it scores 1.000 on both splits. That
is **not a result and not a failure — it is the endpoint of the decomposition.**

The point it makes is the decomposition itself: the executor's entire magnitude-OOD gap
attributes cleanly onto pieces that are *offloadable*, i.e. things we can compute exactly
from the concrete state without learning anything. Walking the ladder shows there is no
residual "learned execution" hiding in the gap — once value access, the ISA's pc rule, and
the operand-vs-zero comparison are handed off, the trajectory is exact at any magnitude.

The blunt implication: **given value access, the net's irreducible *learnable* contribution
to step-by-step execution is ~zero.** Every part of "running the program" that the net was
doing is either trivially structural (pc) or a comparison that is exact regardless of
magnitude (branch) or arithmetic we already offload (ALU). The net is not adding execution
skill on top of the symbolic machinery; it is, at best, an imperfect re-derivation of it
that degrades out of distribution.

So the world model's real value cannot be in *executing* programs whose concrete state you
hold — for that you would just run the VM (which is exactly what rung 3 amounts to). Its
value is confined to regimes where you **can't or won't** run the VM: planning and reasoning
over many candidate edits without instantiating each, reasoning about partial or
uninstantiated programs, or amortizing over distributions of inputs where concrete
execution is unavailable. That — not single-trajectory execution accuracy — is where the
learned model has to earn its keep.
