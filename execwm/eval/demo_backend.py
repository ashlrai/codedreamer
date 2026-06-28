"""Gradio-free backend for the CodeDreamer demo (so the demo logic is testable
without the heavy UI dependency). `demo/app.py` is thin Gradio wiring over this.

The `DemoEngine` loads a trained model once and, for a chosen input magnitude,
samples programs and reports how often the pure-net vs neurosymbolic readouts match
ground truth — plus a colored step-by-step HTML trace of one representative program.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch

from ..data.dataset import collect_examples
from ..substrate import vm as vmmod
from .checkpoint import load_checkpoint
from .neurosym_exec import demo_trace

LEVELS = [5, 20, 60, 150, 400]   # slider stops: in-distribution -> far OOD


class DemoEngine:
    def __init__(self, ckpt: str = "artifacts/neurosym_model.pt", device=None):
        self.device = device or torch.device("cpu")
        ck = load_checkpoint(ckpt, device=self.device)
        self.model = ck["model"].to(self.device).eval()
        self.scodec, self.acodec, self.spec = ck["scodec"], ck["acodec"], ck["spec"]

    def _sample(self, magnitude: int, n: int, seed: int):
        spec = replace(self.spec, max_const=magnitude, max_input_val=magnitude)
        try:
            ex, _ = collect_examples(spec, n, lambda e: True, seed, self.scodec, self.acodec)
        except RuntimeError:
            ex = []
        return ex

    def aggregate(self, magnitude: int, seed: int, n: int = 60) -> dict:
        pn = nx = tot = 0
        for ex in self._sample(magnitude, n, seed + 1000):
            d = demo_trace(self.model, self.scodec, self.acodec, ex, self.device)
            for s in d["steps"]:
                pn += int(s["pure_exact"]); nx += int(s["neurosym_exact"]); tot += 1
        return {"pure_net": pn / tot if tot else float("nan"),
                "neurosym": nx / tot if tot else float("nan"),
                "n_steps": tot}

    def one_trace(self, magnitude: int, seed: int) -> dict | None:
        ex = self._sample(magnitude, 1, seed)
        if not ex:
            return None
        return demo_trace(self.model, self.scodec, self.acodec, ex[0], self.device)

    def intervened_trace(self, magnitude: int, seed: int, override: int):
        """do(input := override): set the first INT input register to ``override``,
        re-run the program with the VM to get the TRUE counterfactual trace, and read
        the model against it. Demonstrates the latent tracks causal interventions —
        change an input (even to a far-OOD value) and the net re-derives the whole
        execution. Returns (target_reg, original_value, demo_trace_dict | None)."""
        ex = self._sample(magnitude, 1, seed)
        if not ex:
            return None, None, None
        ex = ex[0]
        program = ex.trace.program
        init = ex.trace.states[0].copy()
        target = next((n for n in self.scodec.reg_names
                       if init.types.get(n) is not None
                       and init.types[n].name == "INT"), None)
        if target is None:
            return None, None, None
        original = init.regs[target]
        init.regs[target] = int(override)
        try:
            new_trace = vmmod.run_traced(program, init, max_steps=self.spec.max_steps)
        except Exception:  # noqa: BLE001
            return target, original, None
        if len(new_trace.actions) == 0:
            return target, original, None
        shim = SimpleNamespace(trace=new_trace)
        return target, original, demo_trace(self.model, self.scodec, self.acodec,
                                            shim, self.device)


def _fmt_val(v) -> str:
    return "·" if v is None else str(v)


def _cell(v, ok: bool, highlight: bool) -> str:
    bg = "#163a1a" if ok else "#3a1616"
    fg = "#5fd97a" if ok else "#ff6b6b"
    border = "2px solid #d9a85f" if highlight else "1px solid #333"
    return (f'<td style="background:{bg};color:{fg};border:{border};'
            f'padding:3px 8px;text-align:center;font-variant-numeric:tabular-nums">'
            f'{_fmt_val(v)}</td>')


def render_trace_html(d: dict) -> str:
    regs = d["reg_names"]
    written = {s["dst"] for s in d["steps"] if s["dst"] is not None}
    show = [r for r in regs if r.startswith("v") or r in written][:8]
    head = "".join(f'<th style="padding:3px 8px;color:#9aa">{r}</th>' for r in show)
    rows = []
    for s in d["steps"]:
        gt = s["ground_truth"]
        instr = (f'<td style="padding:3px 8px;color:#ccc;font-family:monospace">'
                 f'{s["pc"]:>2}: {s["instr"]}</td>')

        def band(state, exact_flag, label):
            cells = "".join(_cell(state[r], state[r] == gt[r], r == s["dst"]) for r in show)
            tcol = "#5fd97a" if exact_flag else "#ff6b6b"
            tag = "✓" if exact_flag else "✗"
            return f'<td style="color:{tcol};padding:3px 8px">{tag} {label}</td>' + cells

        rows.append(f'<tr>{instr}{band(gt, True, "truth")}</tr>')
        rows.append(f'<tr><td></td>{band(s["pure_net"], s["pure_exact"], "pure-net")}</tr>')
        rows.append(f'<tr><td></td>{band(s["neurosym"], s["neurosym_exact"], "neurosym")}'
                    f'</tr><tr><td colspan="20" style="height:6px"></td></tr>')
    return (f'<div style="overflow:auto;max-height:520px">'
            f'<table style="border-collapse:collapse;font-size:13px">'
            f'<tr><th style="color:#9aa;padding:3px 8px">step: instr</th>'
            f'<th style="color:#9aa;padding:3px 8px">readout</th>{head}</tr>'
            f'{"".join(rows)}</table></div>')


def summary_md(magnitude: int, agg: dict) -> str:
    regime = "in-distribution (trained here)" if magnitude <= LEVELS[0] else \
        f"≈{magnitude // LEVELS[0]}× beyond training magnitude"
    return (
        f"### Inputs up to ≈{magnitude}  ({regime})\n\n"
        f"| readout | next-state exact-match |\n|---|---|\n"
        f"| 🔴 pure-net (decodes digits) | **{agg['pure_net']:.1%}** |\n"
        f"| 🟢 neurosymbolic (offloads arithmetic) | **{agg['neurosym']:.1%}** |\n\n"
        f"*Same frozen ~10M-param model, trained only on values ≤30. "
        f"The only difference is who computes the numbers.*")
