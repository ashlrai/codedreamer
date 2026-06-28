"""Tests for the edit-conditioned dynamics model + divergence head (M3 step-3).

Coverage:
* EditEncoder output shape.
* true_divergence_mask correctness on a hand-built base/edited trace pair
  (including the length-aware control-flow case).
* EditConditionedWM forward: p_div in [0, 1] and grounding logits of right shape.
* train_edit smoke: ~10 steps returns the right dict with metrics in [0, 1].
* divergence head overfits a single batch (loss drops).
"""

import random

import numpy as np
import torch

from execwm.data.state_codec import CodecConfig
from execwm.data.edit_dataset import make_edit_example
from execwm.model.edit_dynamics import (EditConditionedWM, edit_loss,
                                        true_divergence_mask)
from execwm.substrate.dsl import make_config
from execwm.substrate.edits import EditConfig
from execwm.substrate.generators import GenSpec
from execwm.substrate.vm import Trace, VType
from execwm.train.train_edit import (TrainConfig, build, collate_edit_episodes,
                                     EditEpisodeDataset, _run_batch, train_edit)

_SPEC = GenSpec(num_vars=3, num_temps=6, max_depth=1, num_stmts=3,
                max_const=4, max_input_val=4, max_loop_count=2)
_CODEC = CodecConfig(max_digits=4, base=10, max_pc=128)
_EDIT_CFG = EditConfig(max_program_len=128)
_DEVICE = torch.device("cpu")


def _tiny_model():
    return build(_SPEC, _CODEC, _EDIT_CFG, d_model=64, n_heads=4,
                 enc_layers=1, dyn_layers=1)


def _make_batch(n=4, seed=0, max_len=12):
    model, scodec, ecodec = _tiny_model()
    rng = random.Random(seed)
    examples = [make_edit_example(rng, _SPEC, _CODEC, _EDIT_CFG) for _ in range(n)]
    ds = EditEpisodeDataset(examples, scodec, ecodec, max_len=max_len)
    batch = collate_edit_episodes([ds[i] for i in range(len(ds))])
    return model, batch


# ---------------------------------------------------------------------------


def test_edit_encoder_shape():
    model, scodec, ecodec = _tiny_model()
    rng = random.Random(3)
    ex = make_edit_example(rng, _SPEC, _CODEC, _EDIT_CFG)
    edit = ecodec.encode(ex.edit).as_dict()
    B = 5
    edit_t = {k: torch.from_numpy(np.stack([np.asarray(v)] * B)).long()
              for k, v in edit.items()}
    emb = model.embed_edit(edit_t)
    assert emb.shape == (B, model.cfg.d_model)


def test_true_divergence_mask_hand_built():
    config = make_config(num_vars=2, num_temps=2)
    s0 = config.initial_state(regs={"v0": 1})
    s1 = s0.copy(); s1.regs["v1"] = 2; s1.types["v1"] = VType.INT; s1.pc = 1; s1.steps = 1
    s2 = s1.copy(); s2.regs["t0"] = 3; s2.types["t0"] = VType.INT; s2.pc = 2; s2.steps = 2
    s2e = s1.copy(); s2e.regs["t0"] = 99; s2e.types["t0"] = VType.INT; s2e.pc = 2; s2e.steps = 2

    base = Trace(program=[], states=[s0, s1, s2], actions=[None, None], terminated=True)
    # same length, diverges only at index 2
    edited = Trace(program=[], states=[s0.copy(), s1.copy(), s2e],
                   actions=[None, None], terminated=True)
    mask = true_divergence_mask(base, edited)
    assert mask.tolist() == [False, False, True]

    # length-aware: edited trace is shorter -> trailing base index counts diverged
    edited_short = Trace(program=[], states=[s0.copy(), s1.copy()],
                         actions=[None], terminated=True)
    mask2 = true_divergence_mask(base, edited_short)
    assert mask2.tolist() == [False, False, True]

    # codec path agrees with the dataclass-equality path here
    from execwm.data.state_codec import StateCodec
    sc = StateCodec(config, _CODEC)
    assert true_divergence_mask(base, edited, sc).tolist() == [False, False, True]


def test_forward_produces_prob_and_logits():
    model, batch = _make_batch()
    out, edited_tgt, div_target, base_valid, edit_valid, B, L = \
        _run_batch(model, batch, _DEVICE)
    N = B * L
    assert out["p_div"].shape == (N,)
    assert torch.all(out["p_div"] >= 0) and torch.all(out["p_div"] <= 1)
    assert out["logits"]["reg_type"].shape == (N, model.cfg.num_regs, len(VType))
    assert out["logits"]["reg_digits"].shape == (
        N, model.cfg.num_regs, model.cfg.max_digits, model.cfg.base)
    assert out["logits"]["pc"].shape == (N, model.cfg.max_pc + 1)


def test_train_edit_smoke():
    res = train_edit(spec=_SPEC, codec_cfg=_CODEC, edit_cfg=_EDIT_CFG,
                     tc=TrainConfig(steps=10, batch_size=8, max_len=12),
                     n_train=32, n_eval=16, device=_DEVICE,
                     d_model=64, n_heads=4, enc_layers=1, dyn_layers=1,
                     log_every=5)
    assert set(res) >= {"model", "scodec", "ecodec", "eval", "device"}
    assert isinstance(res["model"], EditConditionedWM)
    ev = res["eval"]
    for key in ("div_step_acc", "div_first_acc", "edited_exact_match",
                "edited_per_var_acc"):
        assert 0.0 <= ev[key] <= 1.0, f"{key}={ev[key]} out of [0,1]"


def test_overfit_single_batch():
    """A few episodes, many steps -> divergence + grounding loss should drop."""
    model, batch = _make_batch(n=4, seed=1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    def step_loss():
        out, edited_tgt, div_target, base_valid, edit_valid, B, L = \
            _run_batch(model, batch, _DEVICE)
        return edit_loss(out, div_target, edited_tgt, base_valid=base_valid,
                         edit_valid=edit_valid)

    with torch.no_grad():
        _, m0 = step_loss()
    for _ in range(80):
        loss, _ = step_loss()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        _, m1 = step_loss()

    assert m1["loss"] < m0["loss"] * 0.6, (m0["loss"], m1["loss"])
    assert m1["L_div"] < m0["L_div"] * 0.5, (m0["L_div"], m1["L_div"])
