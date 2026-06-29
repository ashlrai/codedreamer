"""How much OOD comparison is readout-recoverable from the frozen latent?

FINDINGS_FRONTIER target #2: "a stronger comparison readout leaves recoverable
order signal on the table." The model's own (linear) comparison readout scores
OOD ``cmp_result`` ~= 0.63. Question: from the SAME frozen predicted-next latent,
how much OOD comparison accuracy can a *stronger* readout recover?

This is a read-only, eval-time probe on the existing checkpoint
(``artifacts/neurosym_model.pt``, CPU only). It does NOT retrain or modify the
world model.

Protocol
--------
1. Load the frozen checkpoint (CPU). Build an in-distribution example set (the
   saved spec) and a magnitude-OOD set (``replace(spec, max_const=400,
   max_input_val=400)``), ~300 episodes each.
2. For every *comparison-op* transition (op in {LT,LE,EQ,NE,GT,GE} with a dst
   register), run the model's own pipeline: ``z = encode(state)``,
   ``zn = dynamics(z, action(a))`` (the **predicted next latent**), and take the
   *written register's* latent slot ``zn[:, dst]`` as the probe feature. The
   label is the ground-truth BOOL comparison outcome (the dst register's value
   in the true next state, 0/1 -- see ``execwm/eval/neurosym.py``).
3. Train two readouts ON IN-DIST ONLY and score on held-out in-dist AND OOD:
     (a) LINEAR probe   -- a single nn.Linear (baseline-equivalent; the model's
         own digit head is itself linear).
     (b) MLP probe      -- a 2-layer MLP (higher capacity).
   Also record the model's OWN readout accuracy on the exact same transitions
   for a clean apples-to-apples comparison (both the strict ``cmp_result`` =
   type+digits exact, and the bool-value-only accuracy).

The interesting question: does extra readout capacity recover OOD comparison
(=> the gap is readout, recoverable), or is the ceiling set by the latent (=>
capacity won't help OOD; the gap is representational, needs a better latent/prior)?

Run:  PYTHONPATH=. python scripts/analysis_cmp_readout.py
"""
from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from execwm.data.action_codec import ALL_OPS
from execwm.data.dataset import collect_examples
from execwm.data.torch_data import (EpisodeDataset, _ACTION_KEYS, _STATE_KEYS,
                                     collate_episodes, flatten_time)
from execwm.eval.checkpoint import load_checkpoint
from execwm.substrate.vm import Op

_CMP = {Op.LT, Op.LE, Op.EQ, Op.NE, Op.GT, Op.GE}
_CMP_IDX = torch.tensor([i for i, op in enumerate(ALL_OPS) if op in _CMP])


# ---------------------------------------------------------------------------
# Feature + label extraction over comparison-op transitions
# ---------------------------------------------------------------------------


@torch.no_grad()
def collect_cmp_transitions(model, examples, scodec, acodec, device, *,
                            max_len: int = 24, batch_size: int = 64):
    """Run the model's own encode+dynamics; for every comparison-op transition
    return (features, labels, model_correct_strict, model_correct_value).

    features        : (M, d) the predicted-next latent of the written register slot.
    labels          : (M,)   ground-truth BOOL comparison outcome in {0,1}.
    model_strict    : (M,)   model's own readout correct (reg_type + exact digits)
                             -- this reproduces the ``cmp_result`` metric (~0.63 OOD).
    model_value     : (M,)   model's own decoded bool value correct (value-only).
    """
    model.eval()
    ds = EpisodeDataset(examples, scodec, acodec, max_len=max_len)
    base = model.cfg.base
    D = model.cfg.max_digits
    num_regs = model.cfg.num_regs
    cmp_idx = _CMP_IDX.to(device)
    # MSB-first place values for decoding digits -> magnitude.
    place = (base ** torch.arange(D - 1, -1, -1, device=device)).long()  # (D,)

    feats, labels, m_strict, m_value = [], [], [], []
    if len(ds) == 0:
        return (np.zeros((0, model.cfg.d_model), np.float32),
                np.zeros((0,), np.int64), np.zeros((0,), bool), np.zeros((0,), bool))

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_episodes)

    def _in(values, idx):  # membership without torch.isin
        return (values.unsqueeze(-1) == idx).any(-1)

    for batch in loader:
        valid = batch["valid"].to(device)
        B, L = valid.shape
        sel = valid.reshape(B * L)
        if not sel.any():
            continue
        cur = {k: batch[f"s_{k}"][:, :L].to(device) for k in _STATE_KEYS}
        cur_flat, _, _ = flatten_time(cur)
        z = model.encode(cur_flat)
        act = {k: batch[f"a_{k}"][:, :L].to(device) for k in _ACTION_KEYS}
        act_flat, _, _ = flatten_time(act)
        zn = model.dynamics(z, model.action(act_flat))   # predicted next latent
        logits = model.heads(zn)
        tgt = {k: batch[f"s_{k}"][:, 1:L + 1].to(device).reshape(
                   -1, *batch[f"s_{k}"].shape[2:]) for k in _STATE_KEYS}

        zn = zn[sel]
        logits = {k: v[sel] for k, v in logits.items()}
        tgt = {k: v[sel] for k, v in tgt.items()}
        op = act_flat["op"][sel]
        dst = act_flat["dst"][sel]
        N = int(sel.sum().item())

        is_cmp = _in(op, cmp_idx) & (dst < num_regs)
        if not is_cmp.any():
            continue
        rows = torch.arange(N, device=device)[is_cmp]
        d_idx = dst[is_cmp]

        # feature = predicted-next latent of the written (dst) register slot
        feat = zn[rows, d_idx]                                   # (M, d)

        # label = true bool outcome = magnitude of the dst register's true digits
        true_dig = tgt["reg_digits"][rows, d_idx]                # (M, D)
        true_mag = (true_dig * place).sum(-1)                    # (M,)
        label = (true_mag > 0).long()                            # bool result {0,1}

        # model's own readout on the same slot
        pred_dig = logits["reg_digits"][rows, d_idx].argmax(-1)  # (M, D)
        pred_type = logits["reg_type"][rows, d_idx].argmax(-1)
        true_type = tgt["reg_type"][rows, d_idx]
        w_dig = (pred_dig == true_dig).all(-1)
        w_type = (pred_type == true_type)
        strict = (w_dig & w_type)                                # == cmp_result
        pred_mag = (pred_dig * place).sum(-1)
        value_ok = ((pred_mag > 0).long() == label)

        feats.append(feat.cpu().numpy().astype(np.float32))
        labels.append(label.cpu().numpy().astype(np.int64))
        m_strict.append(strict.cpu().numpy().astype(bool))
        m_value.append(value_ok.cpu().numpy().astype(bool))

    if not feats:
        return (np.zeros((0, model.cfg.d_model), np.float32),
                np.zeros((0,), np.int64), np.zeros((0,), bool), np.zeros((0,), bool))
    return (np.concatenate(feats), np.concatenate(labels),
            np.concatenate(m_strict), np.concatenate(m_value))


# ---------------------------------------------------------------------------
# Readout probes (torch; sklearn is numpy-ABI-incompatible in this env)
# ---------------------------------------------------------------------------


def _train_probe(net, X, y, *, epochs=300, lr=1e-2, wd=1e-4, seed=0, batch=4096):
    torch.manual_seed(seed)
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y).float()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    n = len(yt)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                net(Xt[idx]).squeeze(-1), yt[idx])
            loss.backward()
            opt.step()
    return net


@torch.no_grad()
def _probe_acc(net, X, y):
    if len(y) == 0:
        return float("nan"), 0
    logit = net(torch.from_numpy(X)).squeeze(-1)
    pred = (logit > 0).long().numpy()
    return float((pred == y).mean()), len(y)


def _linear_probe(d):
    return torch.nn.Linear(d, 1)


def _mlp_probe(d, h=256):
    return torch.nn.Sequential(
        torch.nn.Linear(d, h), torch.nn.GELU(),
        torch.nn.Linear(h, h), torch.nn.GELU(),
        torch.nn.Linear(h, 1))


def _majority_acc(y):
    if len(y) == 0:
        return float("nan")
    p = y.mean()
    return float(max(p, 1 - p))


def _split(X, y, frac=0.8, seed=0):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y))
    cut = int(len(y) * frac)
    tr, te = perm[:cut], perm[cut:]
    return X[tr], y[tr], X[te], y[te]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/neurosym_model.pt")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cpu")  # CPU ONLY (GPU/MPS is busy with another job)
    print(f"[cmp] loading {args.ckpt} on {device} ...", flush=True)
    ck = load_checkpoint(args.ckpt, device=device)
    model, scodec, acodec = ck["model"], ck["scodec"], ck["acodec"]
    spec = ck["spec"]
    print(f"[cmp] in-dist spec: max_const={spec.max_const}, "
          f"max_input_val={spec.max_input_val}; cmp_ops={[o.name for o in spec.cmp_ops]}",
          flush=True)

    ood_spec = dataclasses.replace(spec, max_const=400, max_input_val=400)

    print(f"[cmp] collecting {args.episodes} in-dist + {args.episodes} OOD episodes ...",
          flush=True)
    indist_ex, _ = collect_examples(spec, args.episodes, lambda ex: True,
                                    args.seed + 11, scodec, acodec)
    ood_ex, ood_att = collect_examples(ood_spec, args.episodes, lambda ex: True,
                                       args.seed + 23, scodec, acodec)
    print(f"[cmp] OOD collected from {ood_att} attempts", flush=True)

    Xi, yi, m_strict_i, m_value_i = collect_cmp_transitions(
        model, indist_ex, scodec, acodec, device)
    Xo, yo, m_strict_o, m_value_o = collect_cmp_transitions(
        model, ood_ex, scodec, acodec, device)
    d = Xi.shape[1]
    print(f"[cmp] cmp transitions: in-dist={len(yi)}  OOD={len(yo)}  feat-dim={d}",
          flush=True)
    print(f"[cmp] base-rate(result=True): in-dist={yi.mean():.3f}  OOD={yo.mean():.3f}",
          flush=True)

    # split in-dist into train / held-out test; probes see ONLY in-dist train.
    Xtr, ytr, Xte, yte = _split(Xi, yi, frac=0.8, seed=args.seed)

    # (a) linear probe
    lin = _train_probe(_linear_probe(d), Xtr, ytr, epochs=300, seed=args.seed)
    lin_id, _ = _probe_acc(lin, Xte, yte)
    lin_ood, _ = _probe_acc(lin, Xo, yo)

    # (b) MLP probe (higher capacity)
    mlp = _train_probe(_mlp_probe(d), Xtr, ytr, epochs=400, seed=args.seed)
    mlp_id, _ = _probe_acc(mlp, Xte, yte)
    mlp_ood, _ = _probe_acc(mlp, Xo, yo)

    # model's own readout on the SAME transitions (in-dist subset matched to held-out
    # test indices is unnecessary -- the model is frozen, so report over all in-dist).
    model_strict_id = float(m_strict_i.mean()) if len(m_strict_i) else float("nan")
    model_strict_ood = float(m_strict_o.mean()) if len(m_strict_o) else float("nan")
    model_value_id = float(m_value_i.mean()) if len(m_value_i) else float("nan")
    model_value_ood = float(m_value_o.mean()) if len(m_value_o) else float("nan")

    print("\n# Comparison-readout recovery from the frozen predicted-next latent\n")
    print("| readout | in-dist acc | OOD acc | OOD drop |")
    print("|---|---|---|---|")
    print(f"| model own readout: cmp_result (type+digits) | {model_strict_id:.3f} | "
          f"{model_strict_ood:.3f} | {model_strict_id - model_strict_ood:+.3f} |")
    print(f"| model own readout: bool value only          | {model_value_id:.3f} | "
          f"{model_value_ood:.3f} | {model_value_id - model_value_ood:+.3f} |")
    print(f"| probe (a) LINEAR                             | {lin_id:.3f} | "
          f"{lin_ood:.3f} | {lin_id - lin_ood:+.3f} |")
    print(f"| probe (b) MLP (2-hidden-layer)              | {mlp_id:.3f} | "
          f"{mlp_ood:.3f} | {mlp_id - mlp_ood:+.3f} |")
    print(f"| majority baseline                           | {_majority_acc(yte):.3f} | "
          f"{_majority_acc(yo):.3f} | -- |")
    print("\nDONE", flush=True)

    return {
        "model_strict_id": model_strict_id, "model_strict_ood": model_strict_ood,
        "model_value_id": model_value_id, "model_value_ood": model_value_ood,
        "lin_id": lin_id, "lin_ood": lin_ood,
        "mlp_id": mlp_id, "mlp_ood": mlp_ood,
        "maj_id": _majority_acc(yte), "maj_ood": _majority_acc(yo),
        "n_id": int(len(yi)), "n_ood": int(len(yo)),
    }


if __name__ == "__main__":
    main()
