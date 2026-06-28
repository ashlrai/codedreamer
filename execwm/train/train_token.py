"""Train + evaluate the TOKEN-space baseline (the control for the latent thesis).

This mirrors ``train_m1.py``'s structure — same generators, same codecs, same
``EpisodeDataset`` / ``collate_episodes`` plumbing, same exact-match metric — but
swaps the grounded latent world model for a vanilla autoregressive Transformer
that predicts the next state as a flat token sequence (see
``execwm/model/token_baseline.py``). Keeping every knob comparable (params, data,
grading) is the point: any gap in single-step exact match is then attributable to
the *architecture*, which is exactly the claim the project is testing.

Training is plain teacher-forced next-token cross-entropy over the next-state
(+EOS) continuation; eval is greedy autoregressive decode graded with the shared
``exact_match_labels`` rule over decoded label dicts.
"""

from __future__ import annotations

import time as _time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.action_codec import ActionCodec
from ..data.dataset import collect_examples, flatten_transitions
from ..data.state_codec import CodecConfig, StateCodec
from ..data.torch_data import (_ACTION_KEYS, _STATE_KEYS, EpisodeDataset,
                               collate_episodes)
from ..model.delta import exact_match_labels
from ..model.token_baseline import (SerializerDims, TokenBaseline,
                                     TokenModelConfig, TokenSerializer,
                                     build_token_baseline,
                                     per_var_accuracy_labels,
                                     predict_next_labels,
                                     teacher_forced_token_accuracy)
from ..substrate.generators import GenSpec
from .train_m1 import TrainConfig, pick_device


# ---------------------------------------------------------------------------
# Batch -> flat (input_ids, labels) for all valid transitions in a padded batch
# ---------------------------------------------------------------------------


def _batch_sequences(serializer: TokenSerializer, batch: dict, device
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten every valid (s_t, a_t) -> s_{t+1} step of a collated episode batch
    into stacked decoder-only training sequences."""
    valid = batch["valid"].to(device)                          # (B, L)
    B, L = valid.shape
    vf = valid.reshape(B * L)

    def gather(prefix: str, keys, t0: int, t1: int) -> dict:
        out = {}
        for k in keys:
            v = batch[f"{prefix}{k}"][:, t0:t1].to(device)
            out[k] = v.reshape(B * L, *v.shape[2:])[vf]
        return out

    cur = gather("s_", _STATE_KEYS, 0, L)
    nxt = gather("s_", _STATE_KEYS, 1, L + 1)
    act = gather("a_", _ACTION_KEYS, 0, L)
    return serializer.build_training_sequence(cur, act, nxt)


# ---------------------------------------------------------------------------
# Evaluation: greedy decode over an arbitrary example list, graded with the
# shared exact-match rule. The OOD bench calls this on in-dist/OOD/CF sets.
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_token_baseline(model: TokenBaseline, serializer: TokenSerializer,
                            scodec: StateCodec, acodec: ActionCodec,
                            examples: list, device, batch_size: int = 128) -> dict:
    """Greedy-decode every (s_t, a_t) -> s_{t+1} transition of ``examples`` and
    grade single-step exact match + per-variable accuracy.

    Returns ``{step_exact_match, per_var_acc, n}`` with metrics in [0, 1]."""
    model.eval()
    examples = [e for e in examples if len(e.trace) > 0]
    if not examples:
        return {"step_exact_match": 0.0, "per_var_acc": 0.0, "n": 0}

    flat = flatten_transitions(examples, scodec, acodec)
    cur = {k: torch.from_numpy(flat[f"s_{k}"]) for k in _STATE_KEYS}
    act = {k: torch.from_numpy(flat[f"a_{k}"]) for k in _ACTION_KEYS}
    nxt = {k: torch.from_numpy(flat[f"ns_{k}"]) for k in _STATE_KEYS}
    N = flat["ex_id"].shape[0]

    em_sum = pv_sum = 0.0
    n = 0
    for i in range(0, N, batch_size):
        sl = slice(i, min(i + batch_size, N))
        cur_b = {k: v[sl] for k, v in cur.items()}
        act_b = {k: v[sl] for k, v in act.items()}
        tgt_b = {k: v[sl].to(device) for k, v in nxt.items()}
        pred = predict_next_labels(model, serializer, cur_b, act_b, device)
        bsz = tgt_b["pc"].shape[0]
        em_sum += exact_match_labels(pred, tgt_b).float().sum().item()
        pv_sum += per_var_accuracy_labels(pred, tgt_b).item() * bsz
        n += bsz
        # autoregressive greedy_decode (no KV cache) fragments MPS unified memory;
        # release the cache each batch so it can't accumulate to an OOM.
        if getattr(device, "type", None) == "mps":
            torch.mps.empty_cache()
    model.train()
    return {"step_exact_match": em_sum / max(n, 1),
            "per_var_acc": pv_sum / max(n, 1), "n": n}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_token_baseline(path: str, model: TokenBaseline,
                        serializer: TokenSerializer, meta: dict | None = None) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "model_config": vars(model.cfg),
        "serializer_dims": vars(serializer.dims),
        "meta": meta or {},
    }, path)


def load_token_baseline(path: str, device=None) -> dict:
    device = device or pick_device()
    ckpt = torch.load(path, map_location=device, weights_only=False)
    serializer = TokenSerializer.from_dims(**ckpt["serializer_dims"])
    model = TokenBaseline(TokenModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    return {"model": model, "serializer": serializer,
            "meta": ckpt.get("meta", {}), "device": device}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def build(spec: GenSpec, codec_cfg: CodecConfig, **model_kw):
    cfg = spec.config()
    scodec = StateCodec(cfg, codec_cfg)
    acodec = ActionCodec(cfg, codec_cfg)
    serializer = TokenSerializer(scodec, acodec)
    model = build_token_baseline(serializer, **model_kw)
    return model, serializer, scodec, acodec


def train_token_baseline(spec: GenSpec | None = None,
                         codec_cfg: CodecConfig | None = None,
                         tc: TrainConfig | None = None, n_train: int = 4000,
                         n_eval: int = 600, seed: int = 0, device=None,
                         log_every: int = 100, **model_kw) -> dict:
    spec = spec or GenSpec(num_vars=4, max_depth=2, num_stmts=5,
                           max_const=5, max_input_val=5, max_loop_count=3)
    codec_cfg = codec_cfg or CodecConfig(max_digits=6, base=10, max_pc=256)
    tc = tc or TrainConfig()
    device = device or pick_device()

    model, serializer, scodec, acodec = build(spec, codec_cfg, **model_kw)
    model.to(device)

    pred = lambda ex: True  # in-distribution training data
    t0 = _time.perf_counter()
    train_ex, _ = collect_examples(spec, n_train, pred, seed, scodec, acodec)
    eval_ex, _ = collect_examples(spec, n_eval, pred, seed + 99, scodec, acodec)
    print(f"[token] collected {len(train_ex)}+{len(eval_ex)} examples "
          f"in {_time.perf_counter()-t0:.1f}s; encoding episodes...", flush=True)
    t0 = _time.perf_counter()
    train_ds = EpisodeDataset(train_ex, scodec, acodec, max_len=tc.max_len)
    print(f"[token] encoded episodes in {_time.perf_counter()-t0:.1f}s", flush=True)
    train_loader = DataLoader(train_ds, batch_size=tc.batch_size, shuffle=True,
                              collate_fn=collate_episodes, drop_last=True)

    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr,
                            weight_decay=tc.weight_decay)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[token] device={device} params={n_params/1e6:.2f}M "
          f"vocab={serializer.vocab_size} T_state={serializer.T_state} "
          f"T_full={serializer.T_full} train_eps={len(train_ds)}")

    step = 0
    data_iter = iter(train_loader)
    while step < tc.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        input_ids, labels = _batch_sequences(serializer, batch, device)
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               labels.reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        step += 1
        if step % log_every == 0 or step == 1:
            acc = teacher_forced_token_accuracy(logits, labels)
            print(f"[token] step {step:4d}  loss {loss.item():.3f}  "
                  f"tf_tok_acc {acc:.3f}")
        # MPS allocator caches aggressively; without periodic release the 375-token
        # training activations accumulate to an OOM over a long run.
        if getattr(device, "type", None) == "mps" and step % 25 == 0:
            torch.mps.empty_cache()

    ev = evaluate_token_baseline(model, serializer, scodec, acodec, eval_ex, device,
                                 batch_size=8)
    print(f"[token] EVAL single-step exact-match {ev['step_exact_match']:.4f}  "
          f"per-var {ev['per_var_acc']:.4f}  (n={ev['n']})")
    return {"model": model, "serializer": serializer, "scodec": scodec,
            "acodec": acodec, "eval": ev, "device": device}


if __name__ == "__main__":
    train_token_baseline()
