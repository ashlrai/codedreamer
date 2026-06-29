# A 10-million-parameter net runs your code in its head — and the reason it couldn't do arithmetic turned out to be the interesting part

**Repo:** https://github.com/ashlrai/codedreamer · **Demo (the magnitude slider, runs on CPU):** `PYTHONPATH=. python demo/app.py`

---

We built a small neural network — about 10 million parameters, the kind of thing that fits on a laptop — and gave it one job: watch a program execute and predict what happens next. Not the source code, the *execution*. Given the live machine state (registers, heap, program counter, flags) and the next instruction, predict the resulting state. It learns to do this entirely in a **learned latent space**, with every single step checked against the exact output of a real interpreter. You can attach a linear probe and read its internal state. You can reach in, edit a register, and watch its predictions fork. It is, in spirit, *Othello-GPT for a running CPU*.

It was *good*. Frozen linear probes decode every state field at ≥0.99. It tracked control flow beautifully — which instruction runs next, how branches resolve. And it was hopeless at arithmetic. Multi-digit results were a disaster. The natural reaction is the famous one: "neural nets can't do multi-digit arithmetic, everyone knows this." We chased the wall anyway, and chasing it produced the result this whole repo is actually about.

## The reframe: it's a factoring problem, not a ceiling

The standard reading of arithmetic failure — Faith-and-Fate, the length-generalization literature — is that it's a *capability ceiling*. Networks just can't carry digits reliably out of distribution, full stop.

Our claim is narrower and, we think, more useful: in a world model of execution, the arithmetic failure is a **factoring** problem. You are asking one network to do two very different jobs at once — predict the *structure* of a computation (what changes, where control goes) and *compute the digits* of the result — and then grading it on getting both right simultaneously. The structure is learnable and, crucially, magnitude-invariant. The digits are not. So stop asking the net to be a calculator.

That reframe is only available because of how the model is built. It's a **grounded latent world model**: a *slotted latent* (one latent slot per piece of machine state), *shallow grounding heads* that decode each slot back to a concrete field, and the whole thing **anchored to a real VM** — a custom DSL compiled to bytecode, with a per-instruction tracer that hands us exact ground truth for free. Ground truth is always one interpreter run away. Because the abstract operation lives in latent space, you can carry "an addition happened here" without committing to the *value* of the sum — and defer that value to the interpreter's ALU. In token space, where every prior execution predictor lives, you can't: you have to emit the digits.

## The headline experiment: one frozen model, two readouts

Here's the clean version. Train **one** model (d=256, ~10M params) on **small** programs — values ≤ ~30 — but give it a codec wide enough to represent big numbers (4 digits, up to 9999). Freeze it. Now read the *same weights* out two ways, on an in-distribution split and a magnitude-OOD split with values 10–25× larger:

- **EM (learned):** the net decodes every field itself, digits included. The status quo.
- **EM (offloaded):** the net predicts everything — pc, type, sign, flags, which slot changed, comparison outcomes — but a perfect ALU supplies the numeric *digit payload*. This is not "just running the VM"; it isolates whether the net's *structural* prediction is right, with arithmetic and only arithmetic handed off.

The result, over n ≈ 10k transitions per split:

| split | EM (learned) | EM (offloaded) | next-pc acc | branch acc | written-digits |
|---|---|---|---|---|---|
| in-distribution (val ≤ 30) | 0.721 | 0.904 | 0.999 | 0.996 | 0.773 |
| **magnitude-OOD (val 300–800)** | **0.000** | **0.790** | **0.986** | **0.989** | 0.254 |

Read it slowly. Whole-state exact-match collapses to **0.000** out of distribution — total failure, the wall everyone expects. But offload arithmetic and the *same frozen model* jumps to **0.790**. And control flow barely moves: next-pc **0.999 → 0.986**, branches **0.996 → 0.989**. The only thing that craters is the digit payload (**0.773 → 0.254**) — exactly, and only, what a symbolic ALU computes for free.

The slider demo makes this visceral. Drag magnitude past anything the model trained on and the pure-net readout turns red while the neurosymbolic one stays green, on one model:

| input magnitude | pure-net exact | neurosymbolic exact |
|---|---|---|
| ≈5 (trained here) | 61.9% | 93.3% |
| ≈20 | 26.2% | 91.5% |
| ≈60 (OOD) | **0.0%** | 96.2% |
| ≈150 | **0.0%** | 90.8% |
| ≈400 (80× training scale) | **0.0%** | 93.3% |

And it isn't just single steps. A neurosymbolic *executor* — net drives control flow, ALU computes values — runs whole programs at 0.39 full-program success OOD (per-step exact 0.612, mean exact horizon 27.5 steps) versus **0.00** for the pure-net executor, which can't survive step one.

**Execution = (learnable, magnitude-invariant control) + (offloadable arithmetic).**

## Multiplication: the cleanest cut

If the thesis is right, the sharpest test is an operation whose *outputs* blow up while its *inputs* stay small — because then the only thing going out of distribution is the digit payload. Multiplication. Train on inputs ≤4, products ≤16; test on inputs ≤40, products ≤1600 (big outputs, modest inputs):

| split | EM (learned) | EM (offloaded) | next-pc acc |
|---|---|---|---|
| in-distribution | 0.695 | 0.928 | 0.999 |
| **magnitude-OOD** | **0.056** | **0.892** | **0.998** |

This is the demonstration in its purest form. The learned readout collapses *harder* than the ADD/SUB case (0.695 → **0.056** — multiplication digits are even further out of distribution), yet **control flow is essentially untouched: 0.999 → 0.998**, and offloading arithmetic recovers almost everything (0.928 → 0.892). Control is *more* invariant here than in the ADD/SUB magnitude run precisely because the large numbers are outputs, not inputs — which is the hinge for the next part.

## The honest turn

We could have stopped there. We didn't, and four follow-up experiments both sharpened the story and corrected it — which is worth being candid about, because the correction is part of what makes the result trustworthy.

**First, a real correction.** We had claimed, flatly, that "control is magnitude-invariant." That's true for large *outputs* (multiplication, pc 0.998). It is **not** fully true for large *inputs*. Classifying the *first* step each OOD program breaks: 100% of divergences are wrong-next-pc errors (the ALU offload never causes a single divergence). In-distribution, ~100% of those are at comparison/branch instructions — classic value-dependent control. But out of distribution, only 50.6% are comparison/branch; **47.2% are wrong-pc errors on plain ADD/SUB steps** — a failure mode that barely exists in-distribution (branch errors stay flat, 106 → 91; arithmetic-step control errors go 0 → 85). Large *input* operands push the latent off-distribution enough that the pc head fumbles even the trivial `pc+1` advance. The single-step 0.986 pc number hid this; that 1.4% concentrates on high-input steps and compounds over a rollout. So: control is invariant to big results, not to big operands. The encoder is the weak link.

**Second, where exactly the encoder is weak.** A frozen linear probe on the latent, fit on in-distribution states only: *sign* (value < 0) is recovered at **1.000 in-dist and 1.000 OOD** — a perfectly magnitude-invariant linear direction. *Order* (value_i < value_j) holds at 0.949 in-dist but drops to 0.818 OOD. It degrades but doesn't collapse — and notably, OOD order (0.818) sits *above* the model's own comparison readout (0.626), meaning the latent carries more usable order structure than the trained head exploits. So the residual gap from the 0.790 headline isn't digits; it's value-derived predicates (comparison 0.788 → 0.626, sign readout 0.881 → 0.798) that depend on *relative* magnitude.

**Third, it's magnitude-specific.** Push *depth* and *length* OOD instead while holding magnitude small, and the decomposition holds directionally (offloaded EM stays ~0.83 vs 0.90) — but the damage lands on control (pc 0.986 → ~0.91), not digits, because the digit head never trips. "Only the arithmetic payload fails" is a statement about the *magnitude* axis specifically. (Honest confound: the length split leaks some incidental magnitude, so read it as suggestive.)

## Why it matters, and the open challenge

Every prior execution predictor we know of — Learning-to-Execute, CodeExecutor, SemCoder, CRUXEval, and Meta's 32B **Code World Model** — predicts traces in **token space**. A token-space model *must* emit the digits and *must* eat the arithmetic error. A grounded *latent* model can carry an abstract operation and defer the numbers to the ALU. Nobody had built a grounded latent world model of execution; this is one, the factoring is natural in it, and that's the moat. A 10M-param laptop model generalizes 10–80× past its training magnitude on structure precisely because it never tried to be the calculator.

The frontier is now precise. The offloaded number is **0.79, not 1.0**, and we know why: the encoder's representation of large *input* operands. Sign transfers perfectly; order doesn't. And here's the catch — magnitude-invariant comparison *cannot be learned from small-magnitude data alone*, because that distribution carries zero signal about how values in the hundreds order. It needs an architectural prior (e.g. an explicit MSB-first digit comparator), a stronger comparison head to recover the slack the probe says is on the table, or to offload comparison the way we offload arithmetic.

That is a real, well-localized, laptop-sized research problem. Everything is graded against the VM oracle, negative results are reported as such, and there are 111 tests in the box. If you want to push the frontier — **magnitude-invariant operand encoding** — clone the repo, drag the slider, and start with `CONTRIBUTING.md`. The interpreter is always one run away to tell you whether you're right.
