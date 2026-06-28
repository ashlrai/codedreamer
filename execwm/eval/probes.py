"""Frozen-encoder linear probes + a causal intervention check.

The interpretability thesis of this project is that the slotted latent ``z``
*linearly* encodes the symbolic machine state. This module operationalizes that
claim two ways, in the Othello-GPT tradition:

1. **Linear probing.** Freeze the encoder, take ``z = encode(state)`` once, and
   fit a *fresh* single ``nn.Linear`` per state field from the relevant slot
   vectors to the ground-truth labels (the codec output). If the latent encodes
   the state linearly, these probes recover it at high accuracy (target >=95%).
   This is read-only evidence: the information is *present* and *linearly
   decodable* from the latent, slot by slot.

2. **Causal intervention.** Take a probe's weight direction (the sign probe is a
   clean binary axis), add a scaled multiple of it to a single register's slot
   vector in ``z``, then decode. If the decoded value for *that* register flips
   while the others stay put, the latent axis the probe found is not merely
   correlational but causally wired into the readout. Because the grounding /
   probe heads are per-slot, editing one slot provably cannot move another's
   decode, so "others stay stable" is structural here -- the load-bearing
   measurement is the targeted flip-rate.

The probes mirror :class:`~execwm.model.world_model.GroundingHeads` exactly in
how they map slots -> fields, so a probe and the jointly-trained head are
directly comparable on the same ``z``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.torch_data import _STATE_KEYS
from ..model.world_model import valued_mask
from ..substrate.vm import VType

# Register sign/digit payloads are junk on UNDEF registers, so probe fitting and
# scoring of those fields is masked to valued (INT/BOOL) registers -- exactly the
# convention the grounding loss and the per-var metric use.
_MASKED_FIELDS = ("reg_sign", "reg_digits")


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


def collect_state_tensors(examples, scodec, max_states: int,
                          device) -> dict[str, torch.Tensor]:
    """Encode states from many example traces into one batched state-dict.

    Walks ``ex.trace.states`` across ``examples``, encodes each state with the
    codec (the ground-truth label arrays), stacks up to ``max_states`` of them,
    and returns torch int64 tensors on ``device`` keyed by ``_STATE_KEYS``
    (shapes ``(N, ...)`` matching what ``model.encode`` consumes).
    """
    buf: dict[str, list[np.ndarray]] = {k: [] for k in _STATE_KEYS}
    count = 0
    for ex in examples:
        for st in ex.trace.states:
            enc = scodec.encode(st).as_dict()
            for k in _STATE_KEYS:
                buf[k].append(enc[k])
            count += 1
            if count >= max_states:
                break
        if count >= max_states:
            break
    if count == 0:
        raise ValueError("no states collected (empty traces?)")
    return {k: torch.from_numpy(np.stack(buf[k])).to(device).long()
            for k in _STATE_KEYS}


# ---------------------------------------------------------------------------
# The probes
# ---------------------------------------------------------------------------


class LinearProbes(nn.Module):
    """One fresh ``nn.Linear`` per state field, mapping slot vectors -> labels.

    Slot routing mirrors ``GroundingHeads``: register fields read the register
    slots ``z[:, :R]`` (per-slot, shared weights), heap fields read the heap
    slots ``z[:, R:R+C]``, ``pc`` reads the pc slot, and ``halted``/``error``
    read the flags slot.
    """

    def __init__(self, cfg) -> None:
        super().__init__()
        d, D, base = cfg.d_model, cfg.max_digits, cfg.base
        self.cfg = cfg
        self.probes = nn.ModuleDict({
            "reg_type": nn.Linear(d, len(VType)),
            "reg_sign": nn.Linear(d, 2),
            "reg_digits": nn.Linear(d, D * base),
            "heap_sign": nn.Linear(d, 2),
            "heap_digits": nn.Linear(d, D * base),
            "pc": nn.Linear(d, cfg.max_pc + 1),
            "halted": nn.Linear(d, 2),
            "error": nn.Linear(d, 2),
        })

    def logits(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        R, C, D, base = cfg.num_regs, cfg.num_cells, cfg.max_digits, cfg.base
        N = z.shape[0]
        reg = z[:, :R]
        heap = z[:, R:R + C]
        pc_slot = z[:, R + C]
        flags_slot = z[:, R + C + 1]
        P = self.probes
        return {
            "reg_type": P["reg_type"](reg),
            "reg_sign": P["reg_sign"](reg),
            "reg_digits": P["reg_digits"](reg).view(N, R, D, base),
            "heap_sign": P["heap_sign"](heap),
            "heap_digits": P["heap_digits"](heap).view(N, C, D, base),
            "pc": P["pc"](pc_slot),
            "halted": P["halted"](flags_slot),
            "error": P["error"](flags_slot),
        }


def _ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           labels.reshape(-1))


def _ce_masked(logits: torch.Tensor, labels: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           labels.reshape(-1), reduction="none").view(labels.shape)
    m = mask
    while m.dim() < loss.dim():
        m = m.unsqueeze(-1)
    m = m.expand_as(loss).float()
    return (loss * m).sum() / m.sum().clamp_min(1.0)


def _probe_loss(logits: dict, state_dict: dict) -> torch.Tensor:
    mask = valued_mask(state_dict["reg_type"])  # (N, R)
    total = logits["reg_type"].new_zeros(())
    for field, lg in logits.items():
        lab = state_dict[field]
        if field in _MASKED_FIELDS:
            total = total + _ce_masked(lg, lab, mask)
        else:
            total = total + _ce(lg, lab)
    return total


@torch.no_grad()
def _encode_frozen(model, state_dict: dict, device) -> torch.Tensor:
    """``z = encode(state).detach()`` -- the encoder is frozen for probing."""
    model.eval()
    s = {k: v.to(device) for k, v in state_dict.items()}
    return model.encode(s).detach()


def fit_linear_probes(model, state_dict: dict, device, epochs: int = 200,
                      lr: float = 1e-2) -> LinearProbes:
    """Freeze the encoder, then fit one fresh linear probe per field on ``z``.

    Computes ``z = model.encode(state_dict).detach()`` once and trains the
    probes (Adam + cross-entropy) against the ground-truth labels in
    ``state_dict``. Register sign/digit fields are masked to valued registers.
    No gradient flows into the encoder -- only the probe weights are optimized.
    """
    state_dict = {k: v.to(device) for k, v in state_dict.items()}
    z = _encode_frozen(model, state_dict, device)
    probes = LinearProbes(model.cfg).to(device)
    opt = torch.optim.Adam(probes.parameters(), lr=lr)
    for _ in range(epochs):
        loss = _probe_loss(probes.logits(z), state_dict)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return probes


# ---------------------------------------------------------------------------
# Accuracy (the interpretability metric)
# ---------------------------------------------------------------------------


@torch.no_grad()
def accuracy_from_logits(logits: dict, state_dict: dict, cfg) -> dict[str, float]:
    """Per-field accuracy + a per-variable composite, all floats in [0, 1].

    Works for either a probe's logits or the model's own grounding-head logits,
    so the two are directly comparable. Register sign/digit accuracies and the
    composite are over valued registers only; digit fields require the *whole*
    digit block to match (mirrors the codec's exact-match rule).
    """
    pred = {k: v.argmax(-1) for k, v in logits.items()}
    valued = valued_mask(state_dict["reg_type"])  # (N, R)
    vn = valued.sum().clamp_min(1).float()

    type_ok = pred["reg_type"] == state_dict["reg_type"]            # (N, R)
    sign_ok = pred["reg_sign"] == state_dict["reg_sign"]            # (N, R)
    dig_ok = (pred["reg_digits"] == state_dict["reg_digits"]).all(-1)  # (N, R)

    heap_sign_ok = pred["heap_sign"] == state_dict["heap_sign"]
    heap_dig_ok = (pred["heap_digits"] == state_dict["heap_digits"]).all(-1)

    out = {
        "reg_type": type_ok.float().mean().item(),
        "reg_sign": (sign_ok & valued).sum().float().div(vn).item(),
        "reg_digits": (dig_ok & valued).sum().float().div(vn).item(),
        "heap_sign": heap_sign_ok.float().mean().item(),
        "heap_digits": heap_dig_ok.float().mean().item(),
        "pc": (pred["pc"] == state_dict["pc"]).float().mean().item(),
        "halted": (pred["halted"] == state_dict["halted"]).float().mean().item(),
        "error": (pred["error"] == state_dict["error"]).float().mean().item(),
        # per-variable composite: type AND sign AND all-digits, over valued regs
        "reg_composite": ((type_ok & sign_ok & dig_ok) & valued)
        .sum().float().div(vn).item(),
    }
    return out


@torch.no_grad()
def probe_accuracy(model, probes: LinearProbes, state_dict: dict,
                   device) -> dict[str, float]:
    """Per-field probe accuracy on the frozen latent (the >=95% target)."""
    z = _encode_frozen(model, state_dict, device)
    sd = {k: v.to(device) for k, v in state_dict.items()}
    return accuracy_from_logits(probes.logits(z), sd, model.cfg)


@torch.no_grad()
def heads_accuracy(model, state_dict: dict, device) -> dict[str, float]:
    """Same metric, but using the model's jointly-trained grounding heads.

    A baseline the linear probe is compared against -- the probe should match
    (or nearly match) the heads if the latent is linearly decodable.
    """
    z = _encode_frozen(model, state_dict, device)
    sd = {k: v.to(device) for k, v in state_dict.items()}
    return accuracy_from_logits(model.heads(z), sd, model.cfg)


# ---------------------------------------------------------------------------
# Causal intervention (Othello-GPT protocol)
# ---------------------------------------------------------------------------


@torch.no_grad()
def causal_intervention(model, probes: LinearProbes, state_dict: dict, device, *,
                        alpha: float = 6.0, max_examples: int = 128,
                        use_heads: bool = True) -> dict:
    """Edit one register's slot along the sign-probe direction; measure the flip.

    Protocol: take the sign probe's weight direction ``w[1] - w[0]`` (the axis
    that separates positive from negative). For each example pick one valued
    register, add ``+/- alpha * <slot-norm> * direction`` to that register's slot
    in ``z`` (sign chosen to push toward the *opposite* class), then decode the
    register's sign. If it flips, the latent axis is causally manipulable.

    Decoding uses the model's own grounding heads by default (``use_heads``), so
    a flip means the intervention moved the *model's* readout, not just the
    probe's -- the stronger Othello-GPT claim. Returns the targeted flip-rate
    plus the (structurally-guaranteed) other-register stability rate.
    """
    sd = {k: v.to(device) for k, v in state_dict.items()}
    z = _encode_frozen(model, sd, device)
    cfg = model.cfg
    R = cfg.num_regs
    n = min(z.shape[0], max_examples)
    z = z[:n]
    valued = valued_mask(sd["reg_type"][:n])           # (n, R)

    def decode_sign(zz: torch.Tensor) -> torch.Tensor:
        lg = model.heads(zz) if use_heads else probes.logits(zz)
        return lg["reg_sign"].argmax(-1)               # (n, R)

    # examples that have at least one valued register; target its first one
    has_valued = valued.any(1)
    idx = torch.arange(n, device=z.device)[has_valued]
    if idx.numel() == 0:
        return {"flip_rate": 0.0, "others_stable_rate": 1.0, "n": 0,
                "alpha": alpha, "field": "reg_sign",
                "decoder": "heads" if use_heads else "probe"}
    target_r = valued.float().argmax(1)[idx]           # first valued reg, (m,)

    w = probes.probes["reg_sign"].weight               # (2, d)
    direction = w[1] - w[0]
    direction = direction / direction.norm().clamp_min(1e-6)
    delta_mag = alpha * z[:, :R].norm(dim=-1).mean()   # scale to typical slot norm

    before = decode_sign(z)
    cur = before[idx, target_r]                        # (m,)
    push = torch.where(cur == 1, z.new_tensor(-1.0), z.new_tensor(1.0))

    z2 = z.clone()
    z2[idx, target_r] = (z2[idx, target_r]
                         + (push * delta_mag).unsqueeze(-1) * direction.unsqueeze(0))
    after = decode_sign(z2)

    flip_rate = (after[idx, target_r] != cur).float().mean().item()

    # stability of the *other* registers (per-slot heads => provably unchanged)
    m = idx.shape[0]
    cols = torch.arange(R, device=z.device).unsqueeze(0).expand(m, R)
    nontarget = cols != target_r.unsqueeze(1)
    if nontarget.any():
        same = (before[idx] == after[idx])
        stable = same[nontarget].float().mean().item()
    else:
        stable = 1.0

    return {"flip_rate": flip_rate, "others_stable_rate": stable, "n": int(m),
            "alpha": alpha, "field": "reg_sign",
            "decoder": "heads" if use_heads else "probe"}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _demo() -> None:
    import random

    from ..data.state_codec import CodecConfig
    from ..substrate.generators import GenSpec, make_example
    from ..train.train_m1 import TrainConfig, pick_device, train

    spec = GenSpec(num_vars=3, num_temps=6, max_depth=1, num_stmts=3,
                   max_const=4, max_input_val=4, max_loop_count=2)
    codec = CodecConfig(max_digits=4, base=10, max_pc=128)
    device = pick_device()

    # Briefly train so the encoder actually learns to encode state, then probe
    # the *frozen* encoder. (Drop the model_kw / steps for a stronger encoder.)
    print("[probes] training a small grounded WM briefly (steps=300)...")
    out = train(spec=spec, codec_cfg=codec, tc=TrainConfig(steps=300),
                n_train=600, n_eval=100, device=device,
                d_model=128, n_heads=4, enc_layers=2, dyn_layers=2)
    model, scodec = out["model"], out["scodec"]

    # Fresh held-out states to probe on.
    rng = random.Random(123)
    examples = []
    while len(examples) < 250:
        e = make_example(rng, spec)
        if len(e.trace) > 0:
            examples.append(e)
    state = collect_state_tensors(examples, scodec, max_states=4000, device=device)
    print(f"[probes] collected {next(iter(state.values())).shape[0]} states")

    probes = fit_linear_probes(model, state, device, epochs=300)
    p_acc = probe_accuracy(model, probes, state, device)
    h_acc = heads_accuracy(model, state, device)
    ci = causal_intervention(model, probes, state, device)

    print("\n[probes] frozen-encoder LINEAR PROBE accuracy (target >=0.95):")
    for k in p_acc:
        print(f"    {k:14s} probe {p_acc[k]:.4f}   heads {h_acc[k]:.4f}")
    print(f"\n[probes] CAUSAL INTERVENTION (decode={ci['decoder']}, "
          f"alpha={ci['alpha']}, n={ci['n']})")
    print(f"    targeted reg-sign flip-rate : {ci['flip_rate']:.4f}")
    print(f"    other-regs stability rate   : {ci['others_stable_rate']:.4f}")


if __name__ == "__main__":
    _demo()
