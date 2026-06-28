"""Hugging Face Spaces entrypoint. HF Spaces expects a top-level `app.py`; this
re-exports the Gradio demo defined in `demo/app.py` so the Space can run it directly.

Locally you can still run either `python app.py` or `python demo/app.py`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from demo.app import demo  # noqa: E402

if __name__ == "__main__":
    demo.launch()
