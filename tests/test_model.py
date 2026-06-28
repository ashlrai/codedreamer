"""Shape + sanity tests for the slotted-latent world model.

The key behavioural test is "overfit a tiny batch": a correct model with enough
steps should drive single-step exact-match to ~1.0 on a handful of episodes. If
it can't, something is wrong with the encode/dynamics/decode wiring.
"""

import random

import torch

from execwm.data.action_codec import ActionCodec
from execwm.data.state_codec import CodecConfig, StateCodec
from execwm.data.torch_data import (EpisodeDataset, _STATE_KEYS,
                                     collate_episodes)
from execwm.model.world_model import (GroundedLatentWM, ModelConfig,
                                      exact_match, grounding_loss)
from execwm.substrate.generators import GenSpec, make_example


def _setup(seed=0, n=6):
    spec = GenSpec(num_vars=3, num_temps=6, max_depth=1, num_stmts=3,
                   max_const=4, max_input_val=4, max_loop_count=2)
    codec = CodecConfig(max_digits=4, base=10, max_pc=128)
    scodec = StateCodec(spec.config(), codec)
    acodec = ActionCodec(spec.config(), codec)
    rng = random.Random(seed)
    ex = []
    while len(ex) < n:
        e = make_example(rng, spec)
        if e.trace.terminated and len(e.trace) > 0:
            ex.append(e)
    ds = EpisodeDataset(ex, scodec, acodec, max_len=12)
    mcfg = ModelConfig.from_codec(len(spec.config().reg_names), scodec.num_cells,
                                  spec.config().num_lists, codec,
                                  d_model=128, n_heads=4, enc_layers=2, dyn_layers=2)
    return ds, GroundedLatentWM(mcfg)


def test_forward_shapes():
    ds, model = _setup()
    batch = collate_episodes([ds[i] for i in range(len(ds))])
    B, L = batch["valid"].shape
    s0 = {k: batch[f"s_{k}"][:, 0] for k in _STATE_KEYS}
    z = model.encode(s0)
    assert z.shape == (B, model.cfg.num_slots, model.cfg.d_model)
    a0 = {k: batch[f"a_{k}"][:, 0] for k in
          ("op", "dst", "a_kind", "a_reg", "a_sign", "a_digits",
           "b_kind", "b_reg", "b_sign", "b_digits", "list_id", "target")}
    zn = model.predict_next(z, a0)
    assert zn.shape == z.shape
    logits = model.heads(zn)
    assert logits["reg_type"].shape[:2] == (B, model.cfg.num_regs)
    assert logits["reg_digits"].shape == (B, model.cfg.num_regs,
                                          model.cfg.max_digits, model.cfg.base)


def test_overfit_single_step():
    """A few episodes, many steps -> single-step exact-match should approach 1."""
    ds, model = _setup(seed=1, n=4)
    batch = collate_episodes([ds[i] for i in range(len(ds))])
    B, L = batch["valid"].shape
    valid = batch["valid"].reshape(B * L)
    cur = {k: batch[f"s_{k}"][:, :L].reshape(-1, *batch[f"s_{k}"].shape[2:])
           for k in _STATE_KEYS}
    act = {k: batch[f"a_{k}"][:, :L].reshape(-1, *batch[f"a_{k}"].shape[2:])
           for k in ("op", "dst", "a_kind", "a_reg", "a_sign", "a_digits",
                     "b_kind", "b_reg", "b_sign", "b_digits", "list_id", "target")}
    tgt = {k: batch[f"s_{k}"][:, 1:L + 1].reshape(-1, *batch[f"s_{k}"].shape[2:])
           for k in _STATE_KEYS}
    cur = {k: v[valid] for k, v in cur.items()}
    act = {k: v[valid] for k, v in act.items()}
    tgt = {k: v[valid] for k, v in tgt.items()}

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for _ in range(400):
        z = model.encode(cur)
        zn = model.predict_next(z, act)
        loss = grounding_loss(model.heads(zn), tgt)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        em = exact_match(model.heads(model.predict_next(model.encode(cur), act)), tgt)
    assert em.float().mean().item() > 0.9, f"overfit exact-match {em.float().mean():.2f}"
