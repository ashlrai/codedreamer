# M2 Findings — ExecWM-Bench: the unified benchmark

M1/M1.5/M1.6 produced a working slotted-latent world model, isolated the binding
constraint (single-step arithmetic), and built three eval modules in parallel
(OOD, interpretability, counterfactual). M2 wires them into a **single benchmark
that grades any trained world model and runs the headline latent-vs-token-space
comparison from one command**.

## What the harness is

`execwm/eval/execwm_bench.py` exposes `run_bench(model, scodec, acodec, spec,
codec_cfg, *, families=...)` → a `BenchReport`. It runs four metric families, each
wrapped in `_safe()` so one failing family records its error and the rest still
produce numbers:

| Family | What it measures | Source module |
|---|---|---|
| `core` | single-step exact-match, per-var accuracy, rollout-horizon curve (in-dist) | `train_m1.evaluate` / `rollout_horizon` |
| `ood` | in-dist vs each of the 5 OOD axes (skips shape-changing axes) | `eval/ood_eval.py` |
| `interp` | frozen-encoder linear-probe accuracy per field + causal-intervention flip-rate | `eval/probes.py` |
| `counterfactual` | `do(register=v')` and `do(swap action)` accuracy vs the VM oracle, vs the identity baseline | `eval/counterfactual.py` |

`execwm/eval/report.py` defines the `BenchReport` dataclass (JSON round-trips via
`to_json`/`from_json`), a human `to_markdown()`, a **scorecard** that grades the
report against the M1/M2 success criteria, and `compare_reports(latent, baseline)`
for the head-to-head. `execwm/eval/checkpoint.py` saves/loads any of the three
model classes (registry over `GroundedLatentWM`/`ArithWM`/`DeltaWM`) with its
codec + spec so a report is reproducible from the artifact.

## The headline comparison (latent vs token-space)

The thesis: a **grounded latent** predictor matches a **token-space** trace
predictor on exactness while being cheaper to roll out — and nobody has built the
latent version (every prior system, incl. Meta's 32B CWM, predicts the trace as
text). `execwm/model/token_baseline.py` is that matched control: a small
seq2seq Transformer that serializes `(state, action)` → next-state token string,
trained on the **same data and step budget** as the latent model via the shared
`TrainConfig`. `scripts/run_execwm_bench.py` trains both, grades them on the
**same held-out eval set**, and prints `compare_reports`.

> **Matched-budget caveat, stated up front:** "matched budget" here means
> identical training data and optimizer steps (latent 10.4M params, token 3.4M).
> It is a fair compute/data control, *not* a claim of equal parameter count (the
> architectures differ inherently). All numbers below come from a single laptop run
> (latent on MPS, token baseline on CPU for memory reasons); a clean GPU run at
> larger budget at equal hardware is the natural next step.

### Results (first matched laptop run — indicative, not decisive)

Both models trained on the **same data, 800 steps**; latent = slotted `GroundedLatentWM`
(10.4M params, MPS), baseline = decoder-only token predictor (3.4M params, CPU — see
the memory note below). Graded on the **same 40 held-out programs**.

| Metric | Latent | Token-space | Δ (latent − token) |
|---|---|---|---|
| single-step **exact-match** (whole next state) | **0.396** | 0.326 | **+0.070** |
| per-variable accuracy (per field) | 0.882 | **0.940** | −0.058 |

**The split is the interesting part.** The token model's *teacher-forced* token accuracy
hits 0.995, and its per-field accuracy is higher (0.940) — but under **greedy
autoregressive decode** its errors scatter across fields so fewer *whole* states come
out exactly right (0.326). The latent model predicts all slots jointly in one shot, so
it lands the **entire** next state more often (0.396) despite slightly lower per-field
accuracy. Exact-match is the metric that matters for plannability — you need the whole
state right to roll forward — and the latent leads it by +7 points at matched budget.

**Honest caveats (this is a laptop run, not the decisive experiment):**
- Tiny budget (800 steps, 3–10M params). Neither model is near the ≥0.99 exact-match
  bar; the absolute numbers are low for both. This is indicative, not a verdict.
- The two paths count transitions slightly differently (the latent path truncates
  episodes at `max_len`; the token path flattens all transitions), so n differs
  (608 vs 1002) on the same 40 source programs — close but not transition-identical.
- The token baseline ran on **CPU** for memory reasons (below); the latent on MPS.
  Matched on data/steps, not on wall-clock or hardware.
- The comparison here covers **core metrics only**. The latent model additionally
  passes the interpretability (frozen-probe 0.99, flip-rate 1.0) and causal
  counterfactual (`do(register)` 0.255 vs 0.0 identity) families — which the token
  baseline has no analogue for. Those properties, not raw exact-match, are the
  grounded latent's real differentiator.

**Conclusion:** at matched laptop budget the two are roughly comparable on raw
exactness, with the latent ahead on whole-state exact-match and uniquely carrying
interpretability + causal structure. The clean win/loss on exactness is a
**GPU-scale matched run** away — exactly the experiment the plan always reserved for
real compute.

### Causal axis — the thesis test (null result at laptop budget)

The central claim is that a *grounded latent* buys **causal** generalization a
token-space predictor lacks. `scripts/compare_causal.py` grades both saved
checkpoints on ONE identical set of intervention pairs (200 `do(register)`, 200
`do(action)`), via `eval/counterfactual.py` (latent) and the new
`eval/token_eval.py:evaluate_counterfactual_token` (token), against the VM oracle:

| Model | `do(register)` EM | `do(action)` EM | identity baseline |
|---|---|---|---|
| grounded latent | 0.265 | 0.045 | 0.000 |
| token-space | 0.285 | 0.035 | 0.000 |

**Honest reading: a tie.** Both models crush the identity ("predict no change")
baseline of 0.0 — so *both* learn genuine causal structure, not copying. But the
latent shows **no causal-accuracy advantage** (−0.02 register, +0.01 action — within
noise at n=200). **At laptop budget the causal-superiority thesis is NOT supported.**
We record this as a null result. `do(action)` is hard for both (~0.04): swapping the
instruction is the strongest intervention and the regime where exact arithmetic +
control-flow prediction both bite — consistent with the M1.6 finding that single-step
arithmetic is the binding constraint for everyone.

What *did* differentiate the grounded latent, and remains true:
- **Interpretability** — frozen linear probes 0.99 / causal-intervention flip-rate
  1.0. The token model has no analogue (you cannot linearly probe or causally edit a
  token string's "internal state").
- **Representational headroom** — the token serialization has a hard ceiling: on the
  `trace_length` OOD axis it emits jump targets beyond `max_pc` that index past its
  vocab, so `evaluate_ood_token` must SKIP that axis; the latent's action encoder
  handles it. A token-specific limit, surfaced honestly rather than clamped.
- **Efficiency** — the latent (transformer over ~30 slots) is dramatically cheaper in
  memory and compute than the token path (375-token sequences + autoregressive decode
  that OOM'd a 128 GB machine). This is the JEPA-efficiency half of the thesis, and it
  held decisively.

**Net M2 verdict (laptop):** the grounded latent is *competitive* on exactness,
*tied* on causal accuracy, and *clearly ahead* on interpretability, representational
headroom, and efficiency — but the causal-generalization claim is unproven and needs
the GPU-scale matched run the plan reserved for it. No overclaiming.

### Memory note (laptop MPS)

The token baseline (375-token serialized sequences + autoregressive greedy decode) is
far more memory- and compute-hungry than the latent model (a transformer over ~30
slots). On this 128 GB Mac, Ollama's `llama-server` parks ~54 GB resident; PyTorch-MPS
shares that unified pool and its allocator never released the token model's cache, so
runs OOM'd / froze the machine. Fixes now in-tree: token baseline trains+evals on
**CPU**, `torch.mps.empty_cache()` in the MPS train/eval loops, both
`PYTORCH_MPS_*_WATERMARK_RATIO` caps + `caffeinate -i` on launch, and the token model
is checkpointed *before* the heavy eval (`--reuse-token` to skip retraining).

## How to run

```bash
# full benchmark + matched token-space baseline (latent M1 model)
PYTHONPATH=. python scripts/run_execwm_bench.py --steps 800 --n-eval 400

# quick (core + counterfactual only), or the carry-aware ArithWM latent
PYTHONPATH=. python scripts/run_execwm_bench.py --quick
PYTHONPATH=. python scripts/run_execwm_bench.py --arith
```

## Status

51 tests green (M0 substrate, M1 slotted model, M1.6 arith/delta, the three eval
modules, the report/checkpoint/token-baseline harness). The project is now a
single-command benchmark over a grounded latent world model of computation, with
a matched token-space control — the seed of the publishable `ExecWM-Bench`
artifact. The open scientific question (does the latent buy causal/OOD
generalization the token model lacks, at scale) is a GPU-budget run away.
