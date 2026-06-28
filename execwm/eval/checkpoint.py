"""Checkpoint persistence for slotted world models.

Saves a trained slotted world model together with everything needed to rebuild
it cold: its ``ModelConfig`` (architecture), the ``CodecConfig`` (numeric
encoding), and the ``GenSpec`` (which fixes the VM register/heap shape via
``spec.config()``). With those three pieces the codecs and the model can be
reconstructed exactly, so the benchmark can run on a persisted checkpoint
instead of a live training run.

The three slotted models — :class:`GroundedLatentWM`, :class:`ArithWM`,
:class:`DeltaWM` — all share the single-``ModelConfig`` constructor signature,
so the class is recorded by name and looked up in :data:`MODEL_REGISTRY`.

Tricky (de)serialization: ``GenSpec.arith_ops`` / ``GenSpec.cmp_ops`` are tuples
of :class:`~execwm.substrate.vm.Op` enums, and ``GenSpec.forbidden_pairs`` is a
``frozenset`` of ``(context_str, Op)`` pairs. ``Op`` enum members are not JSON-
friendly and a frozenset is not order-stable, so those fields are stored as the
enum *names* (``Op.name``) in plain lists and rebuilt with ``Op[name]``.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from ..data.action_codec import ActionCodec
from ..data.state_codec import CodecConfig, StateCodec
from ..model.arith import ArithWM
from ..model.delta import DeltaWM
from ..model.world_model import GroundedLatentWM, ModelConfig
from ..substrate.generators import GenSpec
from ..substrate.vm import Op

# class name -> class, for the three slotted models (all take a single ModelConfig).
MODEL_REGISTRY: dict[str, type] = {
    "GroundedLatentWM": GroundedLatentWM,
    "ArithWM": ArithWM,
    "DeltaWM": DeltaWM,
}


# ---------------------------------------------------------------------------
# GenSpec (de)serialization — the Op enums + forbidden_pairs frozenset
# ---------------------------------------------------------------------------


def serialize_spec(spec: GenSpec) -> dict:
    """GenSpec -> a plain, picklable dict of constructor kwargs.

    ``asdict`` handles the scalar fields; the Op-bearing fields are overridden to
    store the enum *names* so they round-trip without depending on enum identity
    or set ordering.
    """
    d = asdict(spec)
    d["arith_ops"] = [op.name for op in spec.arith_ops]
    d["cmp_ops"] = [op.name for op in spec.cmp_ops]
    d["forbidden_pairs"] = [[ctx, op.name] for ctx, op in spec.forbidden_pairs]
    return d


def deserialize_spec(d: dict) -> GenSpec:
    """Inverse of :func:`serialize_spec`: rebuild a GenSpec from its kwargs dict."""
    d = dict(d)
    d["arith_ops"] = tuple(Op[name] for name in d["arith_ops"])
    d["cmp_ops"] = tuple(Op[name] for name in d["cmp_ops"])
    d["forbidden_pairs"] = frozenset(
        (ctx, Op[name]) for ctx, name in d["forbidden_pairs"])
    return GenSpec(**d)


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_checkpoint(path, model, *, model_cfg: ModelConfig,
                    codec_cfg: CodecConfig, spec: GenSpec,
                    meta: dict | None = None) -> None:
    """Persist a slotted world model + the configs needed to reload it.

    Args:
        path: file path to ``torch.save`` to.
        model: a trained model whose ``type(model).__name__`` is in
            :data:`MODEL_REGISTRY`.
        model_cfg: the architecture config used to build ``model``.
        codec_cfg: the numeric codec config.
        spec: the generator spec (fixes the VM register/heap shape).
        meta: optional free-form metadata (step count, metrics, ...).
    """
    model_class = type(model).__name__
    if model_class not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model class {model_class!r}; "
            f"expected one of {sorted(MODEL_REGISTRY)}")
    payload = {
        "model_class": model_class,
        "state_dict": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "codec_cfg": asdict(codec_cfg),
        "spec": serialize_spec(spec),
        "meta": meta or {},
    }
    torch.save(payload, str(path))


def load_checkpoint(path, device=None) -> dict:
    """Reload a checkpoint into a ready-to-eval model + its codecs.

    Returns a dict with keys: ``model`` (on ``device``, in eval mode),
    ``scodec``, ``acodec``, ``spec``, ``codec_cfg``, ``model_cfg``, ``meta``.
    """
    if not Path(path).exists():
        raise FileNotFoundError(path)
    # Our own trusted payload contains only plain python + tensors; load to CPU
    # first, then move to the requested device.
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # older torch without weights_only kwarg
        ckpt = torch.load(str(path), map_location="cpu")

    spec = deserialize_spec(ckpt["spec"])
    codec_cfg = CodecConfig(**ckpt["codec_cfg"])
    model_cfg = ModelConfig(**ckpt["model_cfg"])

    vm_cfg = spec.config()
    scodec = StateCodec(vm_cfg, codec_cfg)
    acodec = ActionCodec(vm_cfg, codec_cfg)

    model_class = ckpt["model_class"]
    if model_class not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown model class {model_class!r}; "
            f"expected one of {sorted(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[model_class](model_cfg)
    model.load_state_dict(ckpt["state_dict"])
    if device is not None:
        model.to(device)
    model.eval()

    return {
        "model": model,
        "scodec": scodec,
        "acodec": acodec,
        "spec": spec,
        "codec_cfg": codec_cfg,
        "model_cfg": model_cfg,
        "meta": ckpt.get("meta", {}),
    }
