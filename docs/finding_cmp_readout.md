# Is the OOD comparison gap readout-recoverable? Linear vs MLP readouts on the frozen latent

**FINDINGS_FRONTIER target #2** claims "a stronger comparison readout leaves
recoverable order signal on the table" — i.e. the model's own *linear* comparison
head underexploits the latent, and a higher-capacity readout would recover OOD
comparison accuracy. This experiment tests that directly, read-only, on the
existing frozen checkpoint `artifacts/neurosym_model.pt` (CPU only — no retraining,
no architecture change).

## Method

For every **comparison-op transition** (op ∈ {LT, LE, EQ, NE, GT, GE} with a
written dst register) we run the model's *own* pipeline —
`z = encode(state)`, then `zn = dynamics(z, action(a))` (the **predicted next
latent**) — and take the **written register's latent slot** `zn[:, dst]` (256-d)
as the probe feature. The label is the ground-truth BOOL comparison outcome (the
dst register's value in the true next state, 0/1; this is exactly the `cmp_result`
target in `execwm/eval/neurosym.py`).

Two readouts are trained **on in-distribution data only** and scored on held-out
in-dist (20% split) and on magnitude-OOD (`replace(spec, max_const=400,
max_input_val=400)`):

- **(a) LINEAR** — a single `nn.Linear` (baseline-equivalent; the model's own
  digit head is itself linear).
- **(b) MLP** — a 2-hidden-layer MLP (256→256→256, GELU), higher capacity.

We also record the model's **own** readout on the *same* transitions, both the
strict `cmp_result` (reg_type + exact digits) and a bool-value-only accuracy, for
an apples-to-apples comparison. (sklearn is numpy-ABI-incompatible in this env, so
both probes use a torch `nn.Linear`/MLP + logistic loss — same fallback the order
probe used.)

Reproduce: `PYTHONPATH=. python scripts/analysis_cmp_readout.py --episodes 1500`
(script: `scripts/analysis_cmp_readout.py`).

## Results

Primary run (`--episodes 1500 --seed 0`): in-dist = 4 424 comparison transitions,
OOD = 4 244; base-rate(result = True) = 0.681 in-dist, 0.680 OOD.

| readout | in-dist acc | OOD acc | OOD drop |
|---|---|---|---|
| model own readout: `cmp_result` (type+digits) | 0.759 | 0.574 | +0.185 |
| model own readout: bool value only            | 0.759 | 0.713 | +0.046 |
| probe (a) **LINEAR**                           | 0.766 | 0.709 | +0.057 |
| probe (b) **MLP** (2 hidden layers)            | 0.775 | 0.684 | +0.091 |
| majority baseline                              | 0.686 | 0.680 | — |

Robustness (`--seed 1`, 1 500 episodes): in-dist LINEAR 0.759 / MLP 0.781;
OOD LINEAR 0.720 / MLP 0.730 / model-bool 0.726 / majority 0.680. Across the two
seeds the MLP's OOD accuracy is {0.684, 0.730} and the linear probe's is
{0.709, 0.720} — **the linear-vs-MLP OOD ordering flips sign between seeds**, so
that difference is at the noise floor.

## Conclusion — the gap is representational, not readout-recoverable

**Extra readout capacity does not recover OOD comparison.**

1. **Capacity buys in-dist accuracy that does not transfer.** The MLP beats the
   linear probe in-dist in both seeds (≈ +1–2 pts: 0.775 vs 0.766; 0.781 vs
   0.759), confirming the MLP genuinely has more capacity and uses it. But that
   gain **does not transfer OOD** — the MLP's OOD accuracy is statistically
   indistinguishable from the linear probe's (sign of the difference flips across
   seeds). This is the textbook signature of a representational ceiling: more
   capacity fits the training-magnitude regime better and generalizes no better.

2. **The model's own linear head is already near the linear ceiling.** The
   linear probe's OOD accuracy (0.709 / 0.720) matches the model's own
   bool-value readout (0.713 / 0.726) almost exactly. There is essentially **no
   linearly-recoverable comparison signal left on the table** in the written
   register slot that the model's existing head isn't already taking.

3. **Almost nothing is recoverable above chance at OOD.** Every readout — model,
   linear, MLP — sits only ~3–5 points above the OOD majority baseline (0.680).
   The predicted-next latent of the *result* slot retains very little
   magnitude-invariant comparison signal beyond the base rate.

**How much of the cmp gap is readout vs representational?** Effectively all
**representational**. Going from the model's own linear head to a 2-layer MLP on
the identical frozen latent moves OOD comparison by 0 points (within noise). The
recoverable-readout share of the gap is ≈ 0; the ceiling is set by the latent.

**Why this looks different from the ORDER probe — and why it's consistent.** The
order probe (`docs/finding_order_probe.md`) found *operand*-register latents
encode pairwise order strongly (0.949 in-dist, 0.818 OOD). Here the *result* slot
— where dynamics has decided to deposit the boolean — encodes the outcome only
weakly even in-dist (≈ 0.77) and barely above chance OOD. So the order signal is
present in the **operand** representation, but the model's dynamics does not
cleanly transcribe it into a magnitude-invariant boolean in the **written**
slot, and no readout reading that written slot can undo a write that already lost
the information.

**Implication for the thesis.** This refutes the "stronger readout recovers it"
framing of target #2 *for the comparison-result slot*: the OOD comparison wall is
in the latent/dynamics, not the readout head. Closing it needs a better
representation/prior (e.g. a monotone value readout or fixed-position value
embedding so large-magnitude operands aren't themselves OOD) or, in line with the
neurosymbolic thesis, **offloading the comparison to a symbolic comparator** that
reads the (well-recovered, per the order probe) operand order rather than trusting
the learned boolean write.
