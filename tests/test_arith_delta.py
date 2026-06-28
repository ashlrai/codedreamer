"""Mechanical tests for the M1.6 model variants: the carry-aware arithmetic head
and the copy-vs-compute delta head. Fast (no world-model training)."""

import torch

from execwm.data.state_codec import CodecConfig
from execwm.model.arith import ArithDigitHead, ArithGroundingHeads, ArithWM
from execwm.model.delta import (DeltaWM, changed_masks, compose_next,
                                exact_match_labels)
from execwm.model.world_model import ModelConfig
from execwm.substrate.dsl import make_config


def _cfg(d=64):
    codec = CodecConfig(max_digits=4, base=10, max_pc=64)
    vmc = make_config(num_vars=3, num_temps=4, num_lists=1, list_len=2)
    return ModelConfig.from_codec(len(vmc.reg_names), 2, 1, codec,
                                  d_model=d, n_heads=4, enc_layers=2, dyn_layers=2)


def test_arith_digit_head_teacher_and_greedy():
    cfg = _cfg()
    head = ArithDigitHead(cfg.d_model, cfg.base, cfg.max_digits)
    slot = torch.randn(5, cfg.d_model)
    tgt = torch.randint(0, cfg.base, (5, cfg.max_digits))
    tf = head(slot, tgt)                # teacher-forced
    greedy = head(slot)                 # autoregressive greedy
    assert tf.shape == (5, cfg.max_digits, cfg.base)
    assert greedy.shape == (5, cfg.max_digits, cfg.base)


def test_arith_head_can_memorize_digits():
    """Teacher-forced, the carry-aware head should fit a fixed slot->digits map,
    and greedy decoding should then reproduce it (sanity that AR decode works)."""
    torch.manual_seed(0)
    cfg = _cfg(d=64)
    head = ArithDigitHead(cfg.d_model, cfg.base, cfg.max_digits)
    slots = torch.randn(8, cfg.d_model)
    tgt = torch.randint(0, cfg.base, (8, cfg.max_digits))
    opt = torch.optim.Adam(head.parameters(), lr=5e-3)
    for _ in range(300):
        logits = head(slots, tgt)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, cfg.base), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = head(slots).argmax(-1)   # greedy
    acc = (pred == tgt).float().mean().item()
    assert acc > 0.9, f"greedy AR decode only {acc:.2f}"


def test_arith_grounding_heads_shapes():
    cfg = _cfg()
    heads = ArithGroundingHeads(cfg)
    z = torch.randn(3, cfg.num_slots, cfg.d_model)
    out = heads(z)
    assert out["reg_digits"].shape == (3, cfg.num_regs, cfg.max_digits, cfg.base)
    assert out["heap_digits"].shape == (3, cfg.num_cells, cfg.max_digits, cfg.base)
    assert out["pc"].shape == (3, cfg.max_pc + 1)


def test_arith_wm_builds():
    cfg = _cfg()
    model = ArithWM(cfg)
    s = {"reg_type": torch.zeros(2, cfg.num_regs, dtype=torch.long),
         "reg_sign": torch.zeros(2, cfg.num_regs, dtype=torch.long),
         "reg_digits": torch.zeros(2, cfg.num_regs, cfg.max_digits, dtype=torch.long),
         "heap_sign": torch.zeros(2, cfg.num_cells, dtype=torch.long),
         "heap_digits": torch.zeros(2, cfg.num_cells, cfg.max_digits, dtype=torch.long),
         "pc": torch.zeros(2, dtype=torch.long),
         "halted": torch.zeros(2, dtype=torch.long),
         "error": torch.zeros(2, dtype=torch.long)}
    z = model.encode(s)
    assert z.shape == (2, cfg.num_slots, cfg.d_model)


def test_delta_compose_copies_unchanged():
    """With an all-'unchanged' gate, compose_next must reproduce cur exactly, so
    exact_match_labels(compose, cur) is all True."""
    cfg = _cfg()
    N, S = 4, cfg.num_slots
    R, C, D, base = cfg.num_regs, cfg.num_cells, cfg.max_digits, cfg.base

    def rand_labels():
        return {
            "reg_type": torch.randint(0, 3, (N, R)),
            "reg_sign": torch.randint(0, 2, (N, R)),
            "reg_digits": torch.randint(0, base, (N, R, D)),
            "heap_sign": torch.randint(0, 2, (N, C)),
            "heap_digits": torch.randint(0, base, (N, C, D)),
            "pc": torch.randint(0, cfg.max_pc, (N,)),
            "halted": torch.randint(0, 2, (N,)),
            "error": torch.randint(0, 2, (N,)),
        }
    cur = rand_labels()
    gate = torch.zeros(N, S, 2); gate[..., 0] = 10.0   # argmax -> 0 (unchanged)
    value = {  # junk values; must be ignored because gate says copy
        "reg_type": torch.randn(N, R, 3), "reg_sign": torch.randn(N, R, 2),
        "reg_digits": torch.randn(N, R, D, base),
        "heap_sign": torch.randn(N, C, 2), "heap_digits": torch.randn(N, C, D, base),
        "pc": torch.randn(N, cfg.max_pc + 1), "halted": torch.randn(N, 2),
        "error": torch.randn(N, 2),
    }
    pred = compose_next(gate, value, cur, cfg)
    assert exact_match_labels(pred, cur).all()


def test_changed_masks_detect_single_change():
    cfg = _cfg()
    N, R, C, D = 2, cfg.num_regs, cfg.num_cells, cfg.max_digits

    def base_labels():
        return {
            "reg_type": torch.ones(N, R, dtype=torch.long),       # all INT
            "reg_sign": torch.zeros(N, R, dtype=torch.long),
            "reg_digits": torch.zeros(N, R, D, dtype=torch.long),
            "heap_sign": torch.zeros(N, C, dtype=torch.long),
            "heap_digits": torch.zeros(N, C, D, dtype=torch.long),
            "pc": torch.zeros(N, dtype=torch.long),
            "halted": torch.zeros(N, dtype=torch.long),
            "error": torch.zeros(N, dtype=torch.long),
        }
    cur = base_labels()
    nxt = base_labels()
    nxt["reg_digits"][:, 0, -1] = 5      # change register 0's value
    nxt["pc"][:] = 1                     # pc advances
    slot_changed, reg, heap, pc, flags = changed_masks(cur, nxt, cfg)
    assert reg[:, 0].all() and not reg[:, 1].any()
    assert pc.all() and not flags.any()
