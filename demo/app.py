"""CodeDreamer — interactive demo. Watch a ~10M-param net run code in its head.

Tab 1 (the hero): the **magnitude slider**. One frozen model predicts each execution
step, read two ways — *pure-net* (decodes the digit values itself) and *neurosymbolic*
(net predicts control flow & structure; a symbolic ALU fills the arithmetic). Both were
trained ONLY on small numbers. Crank to large, out-of-distribution values: pure-net
collapses to red, neurosymbolic stays green. The wall is the digit head.

Tab 2: **causal intervention**. do(input := v) — override an input register to any value
(even far out of distribution); the net re-derives the entire execution against the true
counterfactual. The latent tracks the intervention.

    pip install gradio          # ideally a fresh venv (see demo/requirements.txt)
    PYTHONPATH=. python demo/app.py
"""
from __future__ import annotations

import gradio as gr

from execwm.eval.demo_backend import (LEVELS, DemoEngine, render_trace_html,
                                       summary_md)

print("[demo] loading model ...", flush=True)
ENGINE = DemoEngine()


def run(level_idx: int, seed: int):
    magnitude = LEVELS[int(level_idx)]
    agg = ENGINE.aggregate(magnitude, seed)
    d = ENGINE.one_trace(magnitude, seed)
    html = render_trace_html(d) if d else "<i>no program sampled at this magnitude</i>"
    return summary_md(magnitude, agg), html


def _new(level_idx, s):
    s = (s + 1) % 100000
    md, html = run(level_idx, s)
    return md, html, s


def intervene(level_idx: int, seed: int, override: float):
    magnitude = LEVELS[int(level_idx)]
    target, original, d = ENGINE.intervened_trace(magnitude, seed, int(override))
    if d is None:
        return "*(no program sampled — try a new program)*", ""
    s = d["summary"]
    cap = (f"### do({target} := {int(override)})  &nbsp; (was {original})\n\n"
           f"The net re-runs the whole program under your intervention, graded against "
           f"the VM's true counterfactual:\n\n"
           f"| readout | exact-match | \n|---|---|\n"
           f"| 🔴 pure-net | **{s['pure_net_exact_frac']:.1%}** |\n"
           f"| 🟢 neurosymbolic | **{s['neurosym_exact_frac']:.1%}** |\n\n"
           f"*Set the value far out of distribution — the green column barely moves.*")
    return cap, render_trace_html(d)


def _new_cf(level_idx, s, override):
    s = (s + 1) % 100000
    cap, html = intervene(level_idx, s, override)
    return cap, html, s


with gr.Blocks(title="CodeDreamer", theme=gr.themes.Base()) as demo:
    gr.Markdown(
        "# 🌀 CodeDreamer\n"
        "### A neural net runs your code in its head — and the magnitude wall is a "
        "design choice, not a limit.")

    with gr.Tab("Magnitude wall"):
        gr.Markdown(
            "Drag the slider to make the program's numbers bigger than anything the "
            "model saw in training. **pure-net** (tries to *compute* the digits) turns "
            "red; **neurosymbolic** (net predicts control flow & structure, a symbolic "
            "ALU fills the arithmetic) stays green at *any* magnitude. Yellow border = "
            "the register the instruction writes.")
        with gr.Row():
            level = gr.Slider(0, len(LEVELS) - 1, value=0, step=1,
                              label="input magnitude   (left = trained regime · right = far out-of-distribution)")
            newbtn = gr.Button("🎲 new program", scale=0)
        seed = gr.State(0)
        out_md = gr.Markdown()
        out_html = gr.HTML()
        level.change(run, [level, seed], [out_md, out_html])
        newbtn.click(_new, [level, seed], [out_md, out_html, seed])

    with gr.Tab("Causal intervention"):
        gr.Markdown(
            "**do(input := v)** — overwrite an input register and watch the net "
            "re-derive the entire execution against the VM's true counterfactual. "
            "Push the value far out of distribution; the neurosymbolic column holds.")
        with gr.Row():
            cf_level = gr.Slider(0, len(LEVELS) - 1, value=2, step=1,
                                 label="sampling magnitude")
            cf_val = gr.Number(value=250, label="override the first input register with…")
            cf_btn = gr.Button("apply intervention", scale=0)
            cf_new = gr.Button("🎲 new program", scale=0)
        cf_seed = gr.State(0)
        cf_md = gr.Markdown()
        cf_html = gr.HTML()
        cf_btn.click(intervene, [cf_level, cf_seed, cf_val], [cf_md, cf_html])
        cf_new.click(_new_cf, [cf_level, cf_seed, cf_val], [cf_md, cf_html, cf_seed])

    demo.load(run, [level, seed], [out_md, out_html])


if __name__ == "__main__":
    demo.launch()
