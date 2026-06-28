# Where the OOD comparison failure lives: SIGN vs ORDER linear probes

**Method.** Othello-GPT-style read-only linear probing of the *existing* frozen
checkpoint `artifacts/neurosym_model.pt` (CPU only). For every state in ~300
in-distribution episodes (the saved spec, `max_const = max_input_val = 5`,
`|value| <= 34`) and ~300 magnitude-OOD episodes
(`replace(spec, max_const=400, max_input_val=400)`, `|value|` up to ~6.3k), we
take the per-register latent `z_i = encode(state)[:, i]` and ground-truth values
from the *same* state's codec labels. Two logistic probes
(`nn.Linear` + logistic loss, frozen encoder) are fit **on in-distribution
latents only** over valued INT registers, then scored on held-out in-dist and on
OOD:

- **SIGN** — predict `value_i < 0` from a single register's latent `z_i`.
- **ORDER** — predict `value_i < value_j` from the concat `[z_i, z_j]` for pairs
  of INT registers (ties dropped, <=8 pairs/state).

Reproduce: `PYTHONPATH=. python scripts/analysis_order_probe.py`
(script: `scripts/analysis_order_probe.py`).

> Note: sklearn is binary-incompatible with numpy in this env, so the script's
> torch `nn.Linear` logistic-regression fallback was used — the spec's primary
> probe option. Training sets were ~31k examples each; test/OOD sets 8k–42k, so
> the numbers are stable.

## Results

| probe | in-dist acc | OOD acc | in-dist majority | OOD majority |
|---|---|---|---|---|
| SIGN  (`value_i < 0`)        | **1.000** | **1.000** | 0.746 | 0.737 |
| ORDER (`value_i < value_j`)  | **0.949** | **0.818** | 0.560 | 0.555 |

(majority = always-predict-the-majority-class baseline on that split.)

## Interpretation

**The failure localizes to pairwise ORDER, not sign.**

1. **Sign is fully magnitude-invariant.** The linear sign axis is recovered
   perfectly in-dist *and* OOD (1.000 → 1.000). Whatever makes the end-task
   "written sign" degrade OOD (0.881 → 0.798 in `FINDINGS_NEUROSYM.md`), it is
   **not** that the sign bit is un-decodable from the latent — sign is cleanly,
   linearly, and invariantly present at every magnitude. That degradation must
   come from elsewhere in the readout pipeline (e.g. which-register-changed or
   the digit-oracle confound), not from a missing sign direction.

2. **Order is the locus.** The pairwise-order axis is strongly linear in-dist
   (0.949) but drops ~13 points OOD (0.818). This is the latent-level signature
   of the known end-task comparison wall (`cmp_result` 0.788 → 0.626). The probe
   sits squarely on the hypothesis: *the latent linearly encodes register
   ordering in-distribution, and that linear structure degrades at OOD
   magnitude.*

3. **It degrades, but does not collapse.** OOD order accuracy (0.818) stays well
   above chance (0.555) and notably *above* the model's own OOD comparison
   readout (0.626). So a large share of ordering structure does transfer to
   unseen magnitudes — plausibly because the compositional digit
   `ValueEmbedding` gives partial magnitude generalization — and the trained
   comparison head fails to fully exploit even the structure the latent retains.

**The deeper point (supported, with a caveat).** A linear order boundary fit on
`|value| <= 34` is given *no signal* about how to separate values in the
hundreds: small-magnitude training cannot constrain the large-magnitude decision
surface. The 13-point residual is exactly the part that **cannot be learned from
this distribution** — magnitude-invariant comparison would need an architectural
prior (e.g. a monotone value readout / abstract interpreter) or to be **offloaded**
to a symbolic comparator, mirroring the project's neurosymbolic thesis that
arithmetic should be offloaded. The honest caveat: ORDER degrades partially
rather than collapsing, so the latent is closer to magnitude-invariant ordering
than the end-task `cmp_result` number alone suggests — the comparison readout
head is itself part of the problem, not only the latent geometry.
