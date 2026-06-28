"""Round-trip tests for checkpoint persistence of the slotted world models.

Builds a tiny model of each of the three slotted classes (mirroring how
``execwm/train/train_m1.py:build`` wires a model + codecs from a GenSpec and a
CodecConfig), saves a checkpoint, reloads it, and asserts the reload is faithful:
same class, identical weights, the rebuilt VM shape matches, and the loaded model
reproduces the *same* forward output on a fixed input (determinism after load).

Kept tiny (d_model=32, 1 layer, no training) so the whole module runs in <15s.
"""

import numpy as np
import pytest
import torch

from execwm.data.action_codec import ActionCodec
from execwm.data.state_codec import CodecConfig, StateCodec
from execwm.eval.checkpoint import (MODEL_REGISTRY, load_checkpoint,
                                    save_checkpoint)
from execwm.model.world_model import ModelConfig
from execwm.substrate.generators import GenSpec
from execwm.substrate.vm import Op

MODEL_CLASSES = ["GroundedLatentWM", "ArithWM", "DeltaWM"]


def _tiny_spec(**overrides) -> GenSpec:
    """A small GenSpec whose config() yields a compact VM (5 regs, 2 heap cells)."""
    kw = dict(num_vars=2, num_inputs=2, num_temps=3,
              num_lists=1, list_len=2, max_steps=64)
    kw.update(overrides)
    return GenSpec(**kw)


def _build(model_class: str, spec: GenSpec):
    """Mirror train_m1.build: codecs + ModelConfig.from_codec -> model."""
    codec_cfg = CodecConfig(max_digits=3, base=10, max_pc=32)
    vm_cfg = spec.config()
    scodec = StateCodec(vm_cfg, codec_cfg)
    acodec = ActionCodec(vm_cfg, codec_cfg)
    mcfg = ModelConfig.from_codec(
        len(vm_cfg.reg_names), scodec.num_cells, vm_cfg.num_lists, codec_cfg,
        d_model=32, n_heads=2, enc_layers=1, dyn_layers=1, ffn_mult=2)
    model = MODEL_REGISTRY[model_class](mcfg)
    model.eval()
    return model, scodec, acodec, mcfg, codec_cfg


def _fixed_input(scodec: StateCodec, spec: GenSpec) -> dict:
    """A single deterministic encoded state, as a (B=1) batch of long tensors."""
    vm_cfg = spec.config()
    state = vm_cfg.initial_state(regs={"v0": 3, "v1": -2},
                                 heap=[[1, 2]][: vm_cfg.num_lists])
    enc = scodec.encode(state).as_dict()
    batch = {}
    for k, v in enc.items():
        t = torch.as_tensor(np.asarray(v))
        batch[k] = t.reshape(1) if t.ndim == 0 else t.unsqueeze(0)
    return batch


def _forward(model, inp: dict) -> torch.Tensor:
    """A deterministic forward output exercising encoder + grounding heads.

    All three classes expose ``encode`` and a ``heads`` decoder callable with a
    single latent argument; ``reg_type`` logits are present in every variant.
    """
    with torch.no_grad():
        z = model.encode(inp)
        return model.heads(z)["reg_type"]


@pytest.mark.parametrize("model_class", MODEL_CLASSES)
def test_save_load_roundtrip(model_class, tmp_path):
    spec = _tiny_spec()
    model, scodec, _, mcfg, codec_cfg = _build(model_class, spec)
    inp = _fixed_input(scodec, spec)
    before = _forward(model, inp)

    path = tmp_path / f"{model_class}.pt"
    save_checkpoint(path, model, model_cfg=mcfg, codec_cfg=codec_cfg,
                    spec=spec, meta={"step": 42})
    out = load_checkpoint(path, device="cpu")

    # same class
    assert type(out["model"]).__name__ == model_class

    # identical weights
    sd_a, sd_b = model.state_dict(), out["model"].state_dict()
    assert sd_a.keys() == sd_b.keys()
    for k in sd_a:
        assert torch.allclose(sd_a[k], sd_b[k]), f"weight mismatch at {k}"

    # rebuilt VM shape matches
    assert out["spec"].config().reg_names == spec.config().reg_names
    assert out["scodec"].num_cells == scodec.num_cells

    # configs + meta round-trip
    assert out["model_cfg"] == mcfg
    assert out["codec_cfg"] == codec_cfg
    assert out["meta"] == {"step": 42}

    # determinism after load: same output on the same fixed input
    after = _forward(out["model"], inp)
    assert torch.allclose(before, after)


def test_spec_enum_and_frozenset_serialization(tmp_path):
    """A GenSpec with non-default arith_ops + forbidden_pairs must round-trip the
    Op enums and the (str, Op) frozenset exactly."""
    forbidden = frozenset({("loop", Op.MUL), ("if", Op.MOD)})
    spec = _tiny_spec(arith_ops=(Op.ADD, Op.SUB),
                      cmp_ops=(Op.LT, Op.EQ),
                      forbidden_pairs=forbidden)
    model, _, _, mcfg, codec_cfg = _build("GroundedLatentWM", spec)

    path = tmp_path / "spec.pt"
    save_checkpoint(path, model, model_cfg=mcfg, codec_cfg=codec_cfg, spec=spec)
    out = load_checkpoint(path, device="cpu")

    rspec = out["spec"]
    assert rspec.arith_ops == (Op.ADD, Op.SUB)
    assert rspec.cmp_ops == (Op.LT, Op.EQ)
    assert rspec.forbidden_pairs == forbidden
    # frozenset identity of members is preserved (enum singletons)
    assert ("loop", Op.MUL) in rspec.forbidden_pairs
    assert ("if", Op.MOD) in rspec.forbidden_pairs
