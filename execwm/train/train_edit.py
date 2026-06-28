"""Training loop + evaluation for the edit-conditioned dynamics model (M3 step-3).

Mirrors the structure of ``train_m1.train``: build codecs + model, collect data
via :func:`make_edit_example`, train with :func:`edit_loss`, and return a dict
``{model, scodec, ecodec, eval, ...}``.

Unlike M1 there is no latent rollout — each base step's edited state and
divergence are predicted independently (see ``model/edit_dynamics.py``). Eval
reports two things: divergence-detection accuracy (does the head find the right
*first* divergence step? plus per-step accuracy) and edited-state exact-match.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..data.edit_codec import EditCodec
from ..data.edit_dataset import make_edit_example
from ..data.state_codec import CodecConfig, StateCodec
from ..data.torch_data import _STATE_KEYS, flatten_time
from ..model.edit_dynamics import EditConditionedWM, edit_loss, true_divergence_mask
from ..model.world_model import ModelConfig, exact_match, per_var_accuracy
from ..substrate.edits import EditConfig
from ..substrate.generators import GenSpec

_EDIT_KEYS = ("kind", "index", "op", "dst", "slot", "reg", "imm_sign", "imm_digits")


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class TrainConfig:
    steps: int = 400
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 1e-2
    max_len: int = 24          # cap on number of base STATES per episode
    grad_clip: float = 1.0
    w_div: float = 1.0
    w_ground: float = 1.0


# ---------------------------------------------------------------------------
# Data: encode an EditExample into per-base-step tensors
# ---------------------------------------------------------------------------


@dataclass
class EncodedEditEpisode:
    base_states: dict[str, np.ndarray]    # each (Lb, *shape)
    edited_targets: dict[str, np.ndarray]  # each (Lb, *shape), aligned to base idx
    div_mask: np.ndarray                  # (Lb,) bool
    valid_edit: np.ndarray                # (Lb,) bool (edited state exists at idx)
    edit: dict[str, np.ndarray]           # edit fields
    length: int                           # Lb = number of base states kept


def encode_edit_episode(ex, scodec: StateCodec, ecodec: EditCodec,
                        max_len: int | None = None) -> EncodedEditEpisode:
    base = ex.base_trace.states
    edited = ex.edited_trace.states
    div = true_divergence_mask(ex.base_trace, ex.edited_trace)
    Lb = len(base)
    if max_len is not None and Lb > max_len:
        Lb = max_len
        base = base[:Lb]
        div = div[:Lb]
    n_edit = len(edited)
    valid_edit = np.array([t < n_edit for t in range(Lb)], dtype=bool)

    enc_base = [scodec.encode(s).as_dict() for s in base]
    # edited target aligned to base index; placeholder (base state) where invalid.
    enc_edit = [scodec.encode(edited[t] if t < n_edit else base[t]).as_dict()
                for t in range(Lb)]
    base_states = {k: np.stack([e[k] for e in enc_base]) for k in _STATE_KEYS}
    edited_targets = {k: np.stack([e[k] for e in enc_edit]) for k in _STATE_KEYS}
    edit = ecodec.encode(ex.edit).as_dict()
    return EncodedEditEpisode(base_states=base_states, edited_targets=edited_targets,
                              div_mask=div, valid_edit=valid_edit, edit=edit,
                              length=Lb)


class EditEpisodeDataset(Dataset):
    def __init__(self, examples: list, scodec: StateCodec, ecodec: EditCodec,
                 max_len: int = 24) -> None:
        self.eps = [encode_edit_episode(e, scodec, ecodec, max_len)
                    for e in examples if len(e.base_trace.states) > 0]

    def __len__(self) -> int:
        return len(self.eps)

    def __getitem__(self, i: int) -> EncodedEditEpisode:
        return self.eps[i]


def collate_edit_episodes(batch: list[EncodedEditEpisode]) -> dict:
    """Pad to the batch's max base length. Returns:
        s_<f>:   (B, Lmax, *shape)  base states
        e_<f>:   (B, Lmax, *shape)  edited-state targets (index-aligned)
        edit_<f>:(B, *shape)        edit fields
        valid:      (B, Lmax) bool  base step exists
        valid_edit: (B, Lmax) bool  edited state aligned at this base index
        div:        (B, Lmax) bool  true divergence mask
    """
    B = len(batch)
    L = max(e.length for e in batch)

    def pad_field(get, key, n_lead):
        sample = get(batch[0])[key]
        shape = sample.shape[1:]
        out = np.zeros((B, L, *shape), dtype=sample.dtype)
        for b, e in enumerate(batch):
            arr = get(e)[key]
            out[b, :arr.shape[0]] = arr
        return torch.from_numpy(out)

    out: dict[str, torch.Tensor] = {}
    for k in _STATE_KEYS:
        out[f"s_{k}"] = pad_field(lambda e: e.base_states, k, L)
        out[f"e_{k}"] = pad_field(lambda e: e.edited_targets, k, L)

    valid = torch.zeros(B, L, dtype=torch.bool)
    valid_edit = torch.zeros(B, L, dtype=torch.bool)
    div = torch.zeros(B, L, dtype=torch.bool)
    for b, e in enumerate(batch):
        valid[b, :e.length] = True
        valid_edit[b, :e.length] = torch.from_numpy(e.valid_edit)
        div[b, :e.length] = torch.from_numpy(e.div_mask)
    out["valid"] = valid
    out["valid_edit"] = valid_edit
    out["div"] = div

    for k in _EDIT_KEYS:
        arrs = [e.edit[k] for e in batch]
        out[f"edit_{k}"] = torch.from_numpy(np.stack(arrs)).long()
    return out


# ---------------------------------------------------------------------------
# Forward / loss for one batch
# ---------------------------------------------------------------------------


def _run_batch(model: EditConditionedWM, batch: dict, device):
    """Flatten a padded batch, run the model, return (out, flat targets/masks, B, L)."""
    valid = batch["valid"].to(device)                       # (B, L)
    B, L = valid.shape
    base = {k: batch[f"s_{k}"].to(device) for k in _STATE_KEYS}  # (B,L,*)
    base_flat, _, _ = flatten_time(base)                    # (B*L,*)

    edit = {k: batch[f"edit_{k}"].to(device) for k in _EDIT_KEYS}
    edit_emb = model.embed_edit(edit)                       # (B, d)
    edit_emb_flat = edit_emb[:, None, :].expand(B, L, edit_emb.shape[-1]).reshape(B * L, -1)

    out = model(base_flat, edit_emb_flat)

    edited_tgt = {k: batch[f"e_{k}"].to(device).reshape(-1, *batch[f"e_{k}"].shape[2:])
                  for k in _STATE_KEYS}
    div_target = batch["div"].to(device).reshape(B * L).float()
    base_valid = valid.reshape(B * L)
    edit_valid = batch["valid_edit"].to(device).reshape(B * L)
    return out, edited_tgt, div_target, base_valid, edit_valid, B, L


@torch.no_grad()
def evaluate(model: EditConditionedWM, loader: DataLoader, device) -> dict:
    model.eval()
    n_steps = step_hit = 0
    n_eps = first_hit = 0
    em_hit = em_n = 0
    pv_sum = pv_n = 0.0
    for batch in loader:
        out, edited_tgt, div_target, base_valid, edit_valid, B, L = \
            _run_batch(model, batch, device)
        pred = (out["p_div"] > 0.5)
        # per-step divergence accuracy over valid base steps
        bv = base_valid
        step_hit += int(((pred == (div_target > 0.5)) & bv).sum().item())
        n_steps += int(bv.sum().item())
        # per-episode first-divergence accuracy
        pred_bl = pred.reshape(B, L)
        true_bl = (div_target > 0.5).reshape(B, L)
        valid_bl = base_valid.reshape(B, L)
        for b in range(B):
            vb = valid_bl[b]
            tt = true_bl[b] & vb
            pp = pred_bl[b] & vb
            true_first = int(tt.float().argmax().item()) if tt.any() else -1
            pred_first = int(pp.float().argmax().item()) if pp.any() else -1
            first_hit += int(true_first == pred_first)
            n_eps += 1
        # edited-state exact match over aligned steps
        ev = edit_valid
        if ev.any():
            em = exact_match({k: v[ev] for k, v in out["logits"].items()},
                             {k: v[ev] for k, v in edited_tgt.items()})
            em_hit += int(em.float().sum().item())
            em_n += int(ev.sum().item())
            n = int(ev.sum().item())
            pv_sum += per_var_accuracy(
                {k: v[ev] for k, v in out["logits"].items()},
                {k: v[ev] for k, v in edited_tgt.items()}).item() * n
            pv_n += n
    model.train()
    return {
        "div_step_acc": step_hit / max(n_steps, 1),
        "div_first_acc": first_hit / max(n_eps, 1),
        "edited_exact_match": em_hit / max(em_n, 1),
        "edited_per_var_acc": pv_sum / max(pv_n, 1),
        "n_steps": n_steps, "n_eps": n_eps,
    }


# ---------------------------------------------------------------------------
# Build + train
# ---------------------------------------------------------------------------


def build(spec: GenSpec, codec_cfg: CodecConfig, edit_cfg: EditConfig, **model_kw):
    cfg = spec.config()
    scodec = StateCodec(cfg, codec_cfg)
    ecodec = EditCodec(cfg, edit_cfg, codec_cfg)
    mcfg = ModelConfig.from_codec(len(cfg.reg_names), scodec.num_cells,
                                  cfg.num_lists, codec_cfg, **model_kw)
    model = EditConditionedWM(mcfg)
    return model, scodec, ecodec


def _collect(spec, codec_cfg, edit_cfg, n, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        out.append(make_edit_example(rng, spec, codec_cfg, edit_cfg))
    return out


def train_edit(spec: GenSpec | None = None, codec_cfg: CodecConfig | None = None,
               tc: TrainConfig | None = None, *, n_train: int = 800,
               n_eval: int = 200, edit_cfg: EditConfig | None = None,
               seed: int = 0, device=None, log_every: int = 50, **model_kw) -> dict:
    spec = spec or GenSpec(num_vars=4, max_depth=2, num_stmts=5,
                           max_const=5, max_input_val=5, max_loop_count=3)
    codec_cfg = codec_cfg or CodecConfig(max_digits=6, base=10, max_pc=256)
    # edit index must fit the EditEncoder's (max_pc + 1) index table.
    edit_cfg = edit_cfg or EditConfig(max_program_len=codec_cfg.max_pc)
    tc = tc or TrainConfig()
    device = device or pick_device()

    model, scodec, ecodec = build(spec, codec_cfg, edit_cfg, **model_kw)
    model.to(device)

    train_ex = _collect(spec, codec_cfg, edit_cfg, n_train, seed)
    eval_ex = _collect(spec, codec_cfg, edit_cfg, n_eval, seed + 99)
    train_ds = EditEpisodeDataset(train_ex, scodec, ecodec, max_len=tc.max_len)
    eval_ds = EditEpisodeDataset(eval_ex, scodec, ecodec, max_len=tc.max_len)
    train_loader = DataLoader(train_ds, batch_size=tc.batch_size, shuffle=True,
                              collate_fn=collate_edit_episodes, drop_last=True)
    eval_loader = DataLoader(eval_ds, batch_size=tc.batch_size, shuffle=False,
                             collate_fn=collate_edit_episodes)

    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[edit] device={device} params={n_params/1e6:.2f}M "
          f"train_eps={len(train_ds)} eval_eps={len(eval_ds)}")

    step = 0
    data_iter = iter(train_loader)
    while step < tc.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        out, edited_tgt, div_target, base_valid, edit_valid, B, L = \
            _run_batch(model, batch, device)
        loss, metrics = edit_loss(out, div_target, edited_tgt,
                                  base_valid=base_valid, edit_valid=edit_valid,
                                  w_div=tc.w_div, w_ground=tc.w_ground)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        step += 1
        if step % log_every == 0 or step == 1:
            print(f"[edit] step {step:4d}  loss {metrics['loss']:.3f}  "
                  f"div {metrics['L_div']:.3f}  ground {metrics['L_ground']:.3f}")

    ev = evaluate(model, eval_loader, device)
    print(f"[edit] EVAL div_first_acc {ev['div_first_acc']:.3f}  "
          f"div_step_acc {ev['div_step_acc']:.3f}  "
          f"edited_exact_match {ev['edited_exact_match']:.3f}")
    return {"model": model, "scodec": scodec, "ecodec": ecodec, "eval": ev,
            "device": device}


if __name__ == "__main__":
    train_edit()
