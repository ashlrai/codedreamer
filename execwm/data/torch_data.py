"""Torch episode dataset: encode each execution trace once into per-step tensors,
then serve padded batches with a validity mask so the trainer can do both
teacher-forced single-step prediction and from-start latent rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from ..substrate.generators import Example
from .action_codec import ActionCodec
from .state_codec import StateCodec

# state fields whose per-element arrays are 1-D (vs the 2-D digit fields)
_STATE_KEYS = ("reg_type", "reg_sign", "reg_digits", "heap_sign", "heap_digits",
               "pc", "halted", "error")
_ACTION_KEYS = ("op", "dst", "a_kind", "a_reg", "a_sign", "a_digits",
                "b_kind", "b_reg", "b_sign", "b_digits", "list_id", "target")


@dataclass
class EncodedEpisode:
    states: dict[str, np.ndarray]   # each (T+1, *shape)
    actions: dict[str, np.ndarray]  # each (T, *shape)
    length: int                     # T (number of steps)


def encode_episode(ex: Example, scodec: StateCodec, acodec: ActionCodec,
                   max_len: int | None = None) -> EncodedEpisode:
    states = ex.trace.states
    actions = ex.trace.actions
    T = len(actions)
    if max_len is not None and T > max_len:
        T = max_len
        states = states[:T + 1]
        actions = actions[:T]
    enc_s = [scodec.encode(s).as_dict() for s in states]
    enc_a = [acodec.encode(a).as_dict() for a in actions]
    state_arrays = {k: np.stack([e[k] for e in enc_s]) for k in _STATE_KEYS}
    action_arrays = {k: np.stack([e[k] for e in enc_a]) for k in _ACTION_KEYS}
    return EncodedEpisode(states=state_arrays, actions=action_arrays, length=T)


class EpisodeDataset(Dataset):
    def __init__(self, examples: list[Example], scodec: StateCodec,
                 acodec: ActionCodec, max_len: int = 48) -> None:
        self.eps = [encode_episode(e, scodec, acodec, max_len) for e in examples
                    if len(e.trace) > 0]

    def __len__(self) -> int:
        return len(self.eps)

    def __getitem__(self, i: int) -> EncodedEpisode:
        return self.eps[i]


def collate_episodes(batch: list[EncodedEpisode]) -> dict:
    """Pad to the batch's max length. Returns tensors:
        s_<field>: (B, Lmax+1, *shape)   states
        a_<field>: (B, Lmax,   *shape)   actions
        valid:     (B, Lmax)             step t has an action
    """
    B = len(batch)
    L = max(e.length for e in batch)

    def pad_states(key: str) -> torch.Tensor:
        sample = batch[0].states[key]
        shape = sample.shape[1:]
        out = np.zeros((B, L + 1, *shape), dtype=sample.dtype)
        for b, e in enumerate(batch):
            arr = e.states[key]
            out[b, :arr.shape[0]] = arr
        return torch.from_numpy(out)

    def pad_actions(key: str) -> torch.Tensor:
        sample = batch[0].actions[key]
        shape = sample.shape[1:]
        out = np.zeros((B, L, *shape), dtype=sample.dtype)
        for b, e in enumerate(batch):
            arr = e.actions[key]
            out[b, :arr.shape[0]] = arr
        return torch.from_numpy(out)

    valid = torch.zeros(B, L, dtype=torch.bool)
    for b, e in enumerate(batch):
        valid[b, :e.length] = True

    out: dict[str, torch.Tensor] = {"valid": valid}
    for k in _STATE_KEYS:
        out[f"s_{k}"] = pad_states(k)
    for k in _ACTION_KEYS:
        out[f"a_{k}"] = pad_actions(k)
    return out


def slice_state(batch: dict, t_index, device) -> dict[str, torch.Tensor]:
    """Extract state-field dict at time index (int or LongTensor) -> model input."""
    return {k: batch[f"s_{k}"][:, t_index].to(device) for k in _STATE_KEYS}


def slice_action(batch: dict, t_index, device) -> dict[str, torch.Tensor]:
    return {k: batch[f"a_{k}"][:, t_index].to(device) for k in _ACTION_KEYS}


def flatten_time(d: dict[str, torch.Tensor]) -> tuple[dict, int, int]:
    """Reshape a (B, L, *) field dict to (B*L, *) for a single encoder pass."""
    any_v = next(iter(d.values()))
    B, L = any_v.shape[:2]
    flat = {k: v.reshape(B * L, *v.shape[2:]) for k, v in d.items()}
    return flat, B, L
