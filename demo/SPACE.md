# Deploying the CodeDreamer demo to Hugging Face Spaces

The demo (`demo/app.py`) is a Gradio app that loads the included ~10M-param checkpoint on
**CPU** and lets you drag a magnitude slider to watch the pure-net readout collapse while
the neurosymbolic readout stays correct. No GPU needed, so a free CPU Space works.

This guide assumes you are deploying from the CodeDreamer repo root.

---

## 1. The Space `README.md` frontmatter

A Hugging Face Space is configured by a YAML frontmatter block at the top of its
`README.md`. Use exactly this (tweak the cosmetic fields freely):

```yaml
---
title: CodeDreamer
emoji: 🌀
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: demo/app.py
pinned: false
---
```

Notes:
- `sdk_version` must be a real released Gradio 4.x version (4.44.0 is a safe pin). It
  should satisfy `gradio>=4.0` from `demo/requirements.txt`.
- `app_file: demo/app.py` points HF at the app. See §4 for the one caveat about imports.

---

## 2. What the Space must contain

The Space repo must include all three of these, or the app will not import / load:

1. **The `execwm/` package** — `demo/app.py` imports `from execwm.eval.demo_backend
   import (LEVELS, DemoEngine, render_trace_html, summary_md)`, which in turn pulls in the
   model, codecs, substrate, and eval code. Push the whole `execwm/` directory.
2. **`demo/app.py`** — the Gradio app itself (and `demo/requirements.txt`).
3. **`artifacts/neurosym_model.pt`** — the trained checkpoint the `DemoEngine` loads. This
   is a binary file; track it with **Git LFS** (HF Spaces support LFS; the file is a few
   MB for a ~10M-param model). Without it the demo cannot start.

The simplest correct mental model: push the repo root (with `execwm/`, `demo/`,
`artifacts/neurosym_model.pt`) plus a Space `README.md` and a top-level
`requirements.txt`.

---

## 3. requirements.txt for the Space

HF Spaces install from a `requirements.txt` at the **repo root**. Mirror the contents of
[`demo/requirements.txt`](requirements.txt):

```
torch>=2.2
numpy
gradio>=4.0
```

If you prefer to keep dependencies only in `demo/requirements.txt`, create a root
`requirements.txt` with the same three lines (HF does not read a nested requirements file
automatically). CPU-only torch is fine and smaller; you may pin `torch` to a CPU wheel if
you want faster, leaner builds.

---

## 4. Making the app importable (the one caveat)

Locally the app is launched with `PYTHONPATH=. python demo/app.py` so that `import execwm`
resolves from the repo root. On HF Spaces, `app_file: demo/app.py` is executed but the
repo root is not guaranteed to be on `sys.path` for the `execwm` import to resolve.

Two robust options:

**Option A — top-level `app.py` shim (recommended).** Add this file at the repo root and
point the Space at it instead (`app_file: app.py`):

```python
# app.py — HF Spaces entrypoint shim. Ensures the repo root is importable, then
# re-exports the Gradio Blocks object from the real app.
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo.app import demo  # noqa: E402  (the Gradio Blocks defined in demo/app.py)

if __name__ == "__main__":
    demo.launch()
```

`demo/app.py` defines its UI as `with gr.Blocks(...) as demo:` and calls `demo.launch()`
under `if __name__ == "__main__"`, so importing `demo` and launching it from the shim is
equivalent. (This shim is a *new* top-level file you add for deployment; it does not modify
`demo/app.py`.)

**Option B — keep `app_file: demo/app.py`.** This can work because HF runs the app from the
repo root, but it is less reliable across SDK versions. If you hit
`ModuleNotFoundError: No module named 'execwm'`, switch to Option A.

---

## 5. Step by step

1. Create a new Space at https://huggingface.co/new-space → SDK: **Gradio**, hardware:
   **CPU basic** (free) is enough.
2. Clone the empty Space repo locally:
   `git clone https://huggingface.co/spaces/<you>/codedreamer`.
3. Copy in the CodeDreamer files: the `execwm/` package, `demo/app.py`,
   `demo/requirements.txt`, and `artifacts/neurosym_model.pt`.
4. Add the Space `README.md` with the frontmatter from §1, and a root
   `requirements.txt` with the §3 contents.
5. Choose your entrypoint:
   - Option A: add the root `app.py` shim from §4 and set `app_file: app.py`.
   - Option B: leave `app_file: demo/app.py`.
6. Track the checkpoint with Git LFS before committing:
   `git lfs install && git lfs track "artifacts/*.pt"` (commit the generated
   `.gitattributes`).
7. Commit and push to the Space remote (`git add -A && git commit && git push`). HF will
   build the image, install `requirements.txt`, and launch the app.
8. Watch the build logs. The app prints `[demo] loading model ...` on startup; once it
   shows the local URL line internally, the Space is live. Drag the magnitude slider to
   confirm the pure-net readout turns red and the neurosymbolic readout stays green.

If the build fails on the model download/load, the most common cause is the checkpoint not
being LFS-tracked (so the file in the repo is an LFS pointer, not the weights) — re-track
with §6 and push again.
