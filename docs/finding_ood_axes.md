# Does execution STRUCTURE generalize off the magnitude axis?

**Question.** The CodeDreamer magnitude finding is that the frozen net predicts
transition *structure* robustly (which slot changes, op type, sign, next pc,
branch/compare outcomes) and only the arithmetic *digit payload* collapses
out-of-distribution. Operationally this shows up as `em_digits_oracle` (digits
supplied by a perfect ALU; everything else still the net's job) staying high while
`em_learned` (net decodes digits too) falls toward zero on magnitude-OOD.

This note asks whether that same structure robustness holds on **two non-magnitude
OOD axes** — **nesting depth** and **trace length** — using the *same frozen
checkpoint* (`artifacts/neurosym_model.pt`), CPU only, no retraining. Magnitude is
held small on the in-dist and depth splits so the moving part is structure, not
numbers.

**Setup.** Model trained on the small-magnitude spec (`max_depth=2`, `num_stmts=5`,
`max_const=max_input_val=5`, `max_loop_count=3`, `max_steps=128`; training rollout
`max_len=18`). Three eval sets of ~300 episodes each, scored by
`field_breakdown(..., max_len=48)`:

- **in-dist** — the exact training spec.
- **depth-OOD** — `replace(spec, max_depth=4)`, kept only where realized nesting >= 3
  (training never exceeds nesting 2).
- **length-OOD** — `replace(spec, num_stmts=10, max_loop_count=5, max_steps=256)`,
  kept only where realized trace length > the in-dist 95th percentile (104 steps).

Reproduce: `PYTHONPATH=. python scripts/analysis_ood_axes.py`

## Realized axis ranges per split (min..max)

| split | trace_len | nesting | \|val\| (magnitude) | n (transitions) |
|---|---|---|---|---|
| in-dist | 6..128 | 0..2 | 2..54 | 10142 |
| depth-OOD | 10..128 | 3..4 | 3..272 | 12169 |
| length-OOD | 105..256 | 2..2 | 5..7642 | 14400 |

## Results

| split | em_learned | em_digits_oracle | pc | cmp_result | written_digits |
|---|---|---|---|---|---|
| in-dist | 0.690 | 0.900 | 0.986 | 0.681 | 0.729 |
| depth-OOD | 0.643 | 0.827 | 0.901 | 0.704 | 0.737 |
| length-OOD | 0.619 | 0.830 | 0.919 | 0.793 | 0.722 |

(`em_digits_oracle` = structure-only exact match; `pc` and `cmp_result` are pure
structure signals; `written_digits` is the arithmetic payload.)

## Interpretation (honest)

**Headline: structure mostly survives both axes, but it is NOT as clean as the
magnitude story — `pc` itself degrades, which it does not on magnitude.**

1. **Structure largely generalizes.** `em_digits_oracle` stays high on both OOD axes
   (0.83 vs 0.90 in-dist) — only ~7-8 points down. So with arithmetic offloaded to an
   ALU, the net's structural prediction of the next state is still right ~83% of the
   time on programs deeper and longer than anything it trained on. That is the same
   qualitative shape as the magnitude finding: the structure channel is the durable
   one. `cmp_result` (a pure control signal) actually *rises* OOD (0.68 -> 0.70/0.79),
   consistent with structure being robust.

2. **But `pc` degrades on these axes, unlike magnitude.** Next-pc accuracy drops from
   0.986 in-dist to 0.901 (depth) and 0.919 (length). On the magnitude axis pc is
   essentially perfect because magnitude never touches control flow; here it does. So
   depth/length DO hurt the structure prediction itself — modestly, via control flow —
   whereas magnitude leaves structure untouched. The ~7-point drop in
   `em_digits_oracle` is mostly this pc/control degradation, not digits.

3. **`em_learned` barely moves (0.69 -> 0.64/0.62).** This is the most important
   honesty caveat and it CONFOUNDS the clean magnitude narrative. On magnitude,
   `em_learned` collapses toward ~0 because magnitude directly breaks the digit head.
   Here it does not collapse — because **magnitude was held small on these splits**, so
   the digit head still works (`written_digits` ~0.72 everywhere, same as in-dist). In
   other words, depth and length do NOT trigger the digit wall at all; they nibble at
   control structure instead. The gap between `em_learned` and `em_digits_oracle`
   (~0.21 in every row) is roughly constant and reflects the standing in-dist digit
   error, not an OOD digit collapse.

4. **Confounds, stated plainly.**
   - The model was never specialized for depth or length; it was trained at
     `max_depth=2` and rollout `max_len=18`. These are genuine OOD probes, but small ones.
   - **Magnitude leaks into the length split.** Holding `max_const/max_input_val=5`
     does not bound *computed* magnitudes: with `max_steps=256` and 10 statements, loops
     accumulate values up to |val|=7642 (and depth-OOD reaches 272). So the length and
     depth numbers are partly contaminated by incidental magnitude OOD — we cannot fully
     separate "long trace" from "big intermediate value." `written_digits` staying flat
     at 0.72 suggests the contamination is mild at the per-write level, but it is real.
   - `max_len=48` truncates length-OOD episodes (realized 105..256 steps) to their
     first 48 transitions, so these metrics describe the early-to-mid portion of long
     executions, not their tails.
   - depth-OOD only reaches nesting 3-4 (the spec caps at 4 and small `num_stmts`
     makes depth 4 rare); this is a shallow excursion off the axis.

**Bottom line.** The structure-vs-digits decomposition is directionally confirmed off
the magnitude axis — `em_digits_oracle` stays high (~0.83) on both depth and length
OOD, so the latent keeps encoding most of the transition structure. But the failure
mode is different and the result is weaker than magnitude: depth/length do not trip
the digit wall (magnitude was held small), and instead cause a real, modest erosion
of *control structure* itself (`pc` 0.99 -> ~0.91). So "only the digit payload fails"
is a magnitude-specific statement; on depth/length the small damage that exists lands
on pc/control, not on arithmetic. With magnitude leaking into the length split, these
should be read as suggestive, not as a clean isolation of the non-magnitude axes.
