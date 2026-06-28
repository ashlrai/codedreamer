"""ExecWM-Bench — the unified benchmark over a trained world model.

Runs every metric family and assembles a :class:`~execwm.eval.report.BenchReport`:
  core            single-step / rollout exact-match + per-var (in-distribution)
  ood             in-dist vs each out-of-distribution axis
  interpretability frozen-encoder linear probes + causal intervention
  counterfactual  causal-intervention accuracy vs the VM oracle (the headline)

Each family is wrapped so one failing block records an error and the rest still
run. Use :func:`run_bench` on a live model, or the ``scripts/run_execwm_bench.py``
CLI to train (or load) a latent model and a token-space baseline and print the
comparison.
"""

from __future__ import annotations

import random
import traceback

import torch
from torch.utils.data import DataLoader

from ..data.dataset import collect_examples
from ..data.state_codec import CodecConfig, StateCodec
from ..data.action_codec import ActionCodec
from ..data.torch_data import EpisodeDataset, collate_episodes
from ..substrate.generators import GenSpec, default_axes
from ..train.train_m1 import evaluate, pick_device, rollout_horizon
from ..plan.goal_tasks import make_goal_task
from ..plan.metrics import evaluate_planning
from ..plan.planner import cheap_scorer
from . import counterfactual as cf
from . import ood_eval, probes
from .report import BenchReport


def _safe(fn, default, label):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - a bad family shouldn't kill the bench
        print(f"[bench] {label} failed: {e}")
        traceback.print_exc()
        return default


def core_metrics(model, scodec, acodec, examples, device, max_len=24) -> dict:
    ds = EpisodeDataset(examples, scodec, acodec, max_len=max_len)
    loader = DataLoader(ds, batch_size=48, shuffle=False, collate_fn=collate_episodes)
    ev = evaluate(model, loader, device)
    horizon = rollout_horizon(model, loader, device, max_k=max_len)
    return {"single_step_exact_match": ev["step_exact_match"],
            "per_var_acc": ev["per_var_acc"], "rollout_horizon": horizon, "n": ev["n"]}


def ood_metrics(model, scodec, acodec, device, n=300) -> dict:
    out = {}
    for axis in default_axes():
        res = _safe(lambda a=axis: ood_eval.compare_indist_vs_ood(
            model, scodec, acodec, a, n=n, device=device), None, f"ood:{axis.name}")
        if res is None:
            out[axis.name] = {"skipped": True, "reason": "error"}
            continue
        if res.get("skipped"):
            out[axis.name] = {"skipped": True, "reason": res.get("reason", "shape mismatch")}
            continue
        ind, ood = res.get("indist", {}), res.get("ood", {})
        out[axis.name] = {
            "indist": {"exact_match": ind.get("step_exact_match", ind.get("exact_match")),
                       "per_var": ind.get("per_var_acc", ind.get("per_var"))},
            "ood": {"exact_match": ood.get("step_exact_match", ood.get("exact_match")),
                    "per_var": ood.get("per_var_acc", ood.get("per_var"))},
            "delta_exact_match": res.get("delta_step_exact_match", res.get("delta_exact_match")),
            "skipped": False, "reason": None,
        }
    return out


def interp_metrics(model, scodec, examples, device, max_states=3000, epochs=150) -> dict:
    state_dict = probes.collect_state_tensors(examples, scodec, max_states, device)
    fitted = probes.fit_linear_probes(model, state_dict, device, epochs=epochs)
    acc = probes.probe_accuracy(model, fitted, state_dict, device)
    interv = probes.causal_intervention(model, fitted, state_dict, device)
    return {"probe_accuracy": {k: v for k, v in acc.items()},
            "reg_composite": acc.get("reg_composite", acc.get("reg")),
            "intervention_flip_rate": interv.get("flip_rate")}


def cf_metrics(model, scodec, acodec, spec, codec_cfg, device, n=400, seed=0) -> dict:
    base = cf.sample_base_transitions(spec, n, seed, codec_cfg=codec_cfg)
    rng = random.Random(seed + 1)
    reg_pairs, act_pairs = [], []
    for st, instr in base:
        r = cf.intervene_register(st, instr, rng, value_range=(-10, 10))
        if r:
            reg_pairs.append(r)
        a = cf.intervene_action(st, instr, rng)
        if a:
            act_pairs.append(a)
    reg = cf.evaluate_counterfactual(model, scodec, acodec, reg_pairs, device)
    act = cf.evaluate_counterfactual(model, scodec, acodec, act_pairs, device)
    return {"register_do": {"exact_match": reg["exact_match"], "per_var": reg["per_var"]},
            "action_swap": {"exact_match": act["exact_match"], "per_var": act["per_var"]},
            "identity_baseline": cf.identity_baseline(reg_pairs)}


def planning_metrics(spec: GenSpec, codec_cfg: CodecConfig, *,
                     n_tasks=30, edit_budget=2, seed=0) -> dict:
    """R4 planning calibration: cheap symbolic scorer vs brute-force VM search.

    This family is model-INDEPENDENT: it takes no neural model. The "planner" is
    the symbolic re-encode beam search (``beam_plan`` + ``cheap_scorer``), and the
    control is brute-force ``vm_search``. It measures whether planning already
    saves *real VM executions* over brute force — the bar a learned edit-conditioned
    scorer (added separately) must clear. A scorer can be swapped into ``planner_kw``
    later without changing this family's shape.

    Builds ``n_tasks`` solvable goal tasks with :func:`make_goal_task` (skipping the
    rare construction failure) and aggregates via :func:`evaluate_planning`. Returns
    a clean summary dict (see ``BenchReport`` schema / ``report.planning``).
    """
    rng = random.Random(seed)
    tasks = []
    for _ in range(n_tasks):
        try:
            tasks.append(make_goal_task(rng, spec, codec_cfg, edit_budget=edit_budget))
        except RuntimeError:
            continue  # could not construct a task this draw; skip it honestly
    if not tasks:
        return {}

    res = evaluate_planning(
        tasks,
        # baseline: no-model brute-force VM search; planner: cheap symbolic scorer.
        # The baseline cap bounds the (combinatorial) brute-force cost per task; the
        # planner cap bounds verified candidates. Both are honest hard caps: hitting
        # them is recorded as an unsolved task, never a hidden cost.
        baseline_kw=dict(max_executions=2_000, strategy="bfs"),
        planner_kw=dict(scorer=cheap_scorer, beam_width=8,
                        max_depth=edit_budget, max_executions=200),
    )
    return {
        "baseline_success_rate": res["baseline_success_rate"],
        "planned_success_rate": res["planned_success_rate"],
        "baseline_mean_execs": res["baseline_mean_execs"],
        "planned_mean_execs": res["planned_mean_execs"],
        "baseline_mean_execs_solved": res["baseline_mean_execs_solved"],
        "planned_mean_execs_solved": res["planned_mean_execs_solved"],
        "mean_saved_frac": res["mean_saved_frac"],
        "n_tasks": res["num_tasks"],
        "n_both_solved": res["num_both_solved"],
        "scorer": "cheap_symbolic",  # no world model; swap in a learned scorer later
    }


def run_bench(model, scodec, acodec, spec: GenSpec, codec_cfg: CodecConfig, *,
              device=None, model_name="latent", n_eval=600, n_cf=400, seed=0,
              families=("core", "ood", "interp", "counterfactual", "planning"),
              examples=None, meta: dict | None = None) -> BenchReport:
    device = device or pick_device()
    model = model.to(device)
    model.eval()
    if examples is None:
        examples, _ = collect_examples(spec, n_eval, lambda e: True, seed + 7, scodec, acodec)

    report = BenchReport(model_name=model_name, meta=meta or {})
    if "core" in families:
        report.core = _safe(lambda: core_metrics(model, scodec, acodec, examples, device),
                            {}, "core")
    if "ood" in families:
        report.ood = _safe(lambda: ood_metrics(model, scodec, acodec, device), {}, "ood")
    if "interp" in families:
        report.interpretability = _safe(
            lambda: interp_metrics(model, scodec, examples, device), {}, "interp")
    if "counterfactual" in families:
        report.counterfactual = _safe(
            lambda: cf_metrics(model, scodec, acodec, spec, codec_cfg, device, n=n_cf),
            {}, "counterfactual")
    if "planning" in families:
        # Model-INDEPENDENT: the planning family uses a cheap symbolic scorer vs
        # brute-force VM search (R4 calibration), so it runs from spec + codec
        # alone and does NOT touch the neural model. A learned scorer can be
        # swapped into planning_metrics later without changing this wiring.
        #
        # edit_budget=1 here for tractability: the brute-force depth-2 baseline
        # re-executes many loopy candidate programs (each up to max_steps), giving
        # pathological per-task wall-time on some tasks. Depth-1 goal tasks are
        # fast (sub-second for n_tasks) and still a meaningful executions-saved
        # calibration. Call planning_metrics directly with edit_budget=2 for the
        # deeper (slower) calibration.
        report.planning = _safe(
            lambda: planning_metrics(spec, codec_cfg, edit_budget=1), {}, "planning")
    return report
