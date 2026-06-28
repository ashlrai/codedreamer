"""Where does the OOD comparison failure live? — linear ORDER/SIGN probes.

Othello-GPT-style read-only probing of the *existing* trained checkpoint
(``artifacts/neurosym_model.pt``). Known finding (FINDINGS_NEUROSYM.md):
comparison-outcome accuracy degrades with magnitude (in-dist ~0.79 -> OOD ~0.63).

Hypothesis under test: the per-register latent z_i *linearly* encodes a
register's SIGN and the *pairwise ORDER* of two registers in-distribution, but
that linear structure degrades at OOD magnitude.

Protocol
--------
1. Load the frozen checkpoint (CPU only). Build an in-distribution example set
   (the saved spec as-is) and a magnitude-OOD set
   (``replace(spec, max_const=400, max_input_val=400)``), ~300 episodes each.
2. For every state, ``z = model.encode(state)`` -> per-slot latent (N, S, d);
   register slots are ``z[:, :R]``. Ground-truth register values come from the
   codec labels (sign/digits) of the *same* encoded state, so z_i and its label
   are aligned by construction.
3. Two probing tasks over valued INT registers:
     SIGN  — predict (value_i < 0) from a single register's latent z_i.
     ORDER — predict (value_i < value_j) from the concat [z_i, z_j] for pairs.
   Each probe (sklearn LogisticRegression, frozen encoder) is fit ON IN-DIST
   latents only, then scored on held-out in-dist AND on OOD.

Run:  PYTHONPATH=. python scripts/analysis_order_probe.py
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from execwm.data.dataset import collect_examples
from execwm.data.state_codec import decode_int
from execwm.eval.checkpoint import load_checkpoint
from execwm.eval.probes import collect_state_tensors
from execwm.substrate.vm import VType

try:
    from sklearn.linear_model import LogisticRegression
    _HAVE_SKLEARN = True
except Exception:  # noqa: BLE001
    _HAVE_SKLEARN = False

import dataclasses


# ---------------------------------------------------------------------------
# Latent + ground-truth extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def encode_states(model, state_dict, device):
    """z = encode(state) on the frozen encoder -> (N, S, d) numpy float32."""
    model.eval()
    s = {k: v.to(device) for k, v in state_dict.items()}
    return model.encode(s).detach().cpu().numpy().astype(np.float32)


def reg_values(state_dict, codec_cfg) -> np.ndarray:
    """Decode signed integer values per register -> (N, R) int64 array.

    Uses the same (sign, MSB-first digits) labels the encoder consumed, so the
    decoded value aligns with latent slot i exactly.
    """
    sign = state_dict["reg_sign"].cpu().numpy()        # (N, R)
    digits = state_dict["reg_digits"].cpu().numpy()    # (N, R, D)
    N, R = sign.shape
    vals = np.zeros((N, R), dtype=np.int64)
    for n in range(N):
        for i in range(R):
            vals[n, i] = decode_int(int(sign[n, i]), digits[n, i], codec_cfg)
    return vals


def int_mask(state_dict) -> np.ndarray:
    """(N, R) bool mask of registers whose type is INT (signed numeric payload)."""
    rt = state_dict["reg_type"].cpu().numpy()
    return rt == VType.INT.value


# ---------------------------------------------------------------------------
# Build probe datasets
# ---------------------------------------------------------------------------


def build_sign_dataset(z, vals, mask):
    """SIGN task: X = z_i for each INT register; y = (value_i < 0)."""
    N, R = mask.shape
    rows, ys = [], []
    for n in range(N):
        for i in range(R):
            if mask[n, i]:
                rows.append(z[n, i])
                ys.append(1 if vals[n, i] < 0 else 0)
    if not rows:
        return np.zeros((0, z.shape[-1]), np.float32), np.zeros((0,), np.int64)
    return np.stack(rows).astype(np.float32), np.asarray(ys, np.int64)


def build_order_dataset(z, vals, mask, *, max_pairs_per_state=8, seed=0):
    """ORDER task: X = [z_i, z_j] for INT register pairs; y = (value_i < value_j).

    Ties (value_i == value_j) are dropped so the label is a clean strict order.
    Pairs per state are capped to keep the set light.
    """
    rng = random.Random(seed)
    N, R = mask.shape
    rows, ys = [], []
    for n in range(N):
        idx = [i for i in range(R) if mask[n, i]]
        pairs = [(i, j) for a, i in enumerate(idx) for j in idx[a + 1:]]
        if len(pairs) > max_pairs_per_state:
            pairs = rng.sample(pairs, max_pairs_per_state)
        for i, j in pairs:
            vi, vj = int(vals[n, i]), int(vals[n, j])
            if vi == vj:
                continue
            rows.append(np.concatenate([z[n, i], z[n, j]]))
            ys.append(1 if vi < vj else 0)
    if not rows:
        return np.zeros((0, 2 * z.shape[-1]), np.float32), np.zeros((0,), np.int64)
    return np.stack(rows).astype(np.float32), np.asarray(ys, np.int64)


# ---------------------------------------------------------------------------
# Probe fit / score
# ---------------------------------------------------------------------------


def _fit_logreg(X, y):
    if _HAVE_SKLEARN:
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X, y)
        return ("sklearn", clf)
    # torch fallback: single nn.Linear + logistic loss
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y).float()
    lin = torch.nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2)
    for _ in range(400):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            lin(Xt).squeeze(-1), yt)
        loss.backward()
        opt.step()
    return ("torch", lin)


def _predict(model, X):
    kind, clf = model
    if kind == "sklearn":
        return clf.predict(X)
    with torch.no_grad():
        return (torch.from_numpy(X).matmul(clf.weight.squeeze(0))
                + clf.bias > 0).long().numpy()


def _acc(model, X, y):
    if len(y) == 0:
        return float("nan"), 0
    return float((_predict(model, X) == y).mean()), len(y)


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="artifacts/neurosym_model.pt")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--max-states", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cpu")  # CPU ONLY (GPU is busy with another job)
    print(f"[probe] loading {args.ckpt} on {device} ...", flush=True)
    ck = load_checkpoint(args.ckpt, device=device)
    model, scodec, acodec = ck["model"], ck["scodec"], ck["acodec"]
    spec, codec_cfg = ck["spec"], ck["codec_cfg"]
    print(f"[probe] in-dist spec: max_const={spec.max_const}, "
          f"max_input_val={spec.max_input_val}; "
          f"codec base={codec_cfg.base}, digits={codec_cfg.max_digits}", flush=True)

    ood_spec = dataclasses.replace(spec, max_const=400, max_input_val=400)

    print(f"[probe] collecting {args.episodes} in-dist + {args.episodes} OOD "
          "episodes ...", flush=True)
    indist_ex, _ = collect_examples(spec, args.episodes, lambda ex: True,
                                    args.seed + 11, scodec, acodec)
    ood_ex, ood_att = collect_examples(ood_spec, args.episodes, lambda ex: True,
                                       args.seed + 23, scodec, acodec)
    print(f"[probe] OOD collected from {ood_att} attempts", flush=True)

    # encode -> latents + ground-truth labels
    sd_id = collect_state_tensors(indist_ex, scodec, args.max_states, device)
    sd_ood = collect_state_tensors(ood_ex, scodec, args.max_states, device)
    z_id, z_ood = encode_states(model, sd_id, device), encode_states(model, sd_ood, device)
    v_id, v_ood = reg_values(sd_id, codec_cfg), reg_values(sd_ood, codec_cfg)
    m_id, m_ood = int_mask(sd_id), int_mask(sd_ood)
    R = model.cfg.num_regs
    z_id, z_ood = z_id[:, :R], z_ood[:, :R]  # register slots only
    print(f"[probe] states: in-dist={z_id.shape[0]}, OOD={z_ood.shape[0]}; "
          f"R={R}", flush=True)
    print(f"[probe] |value| in-dist max={np.abs(v_id[m_id]).max() if m_id.any() else 0}, "
          f"OOD max={np.abs(v_ood[m_ood]).max() if m_ood.any() else 0}", flush=True)

    # ---- SIGN probe ----
    Xs_id, ys_id = build_sign_dataset(z_id, v_id, m_id)
    Xs_ood, ys_ood = build_sign_dataset(z_ood, v_ood, m_ood)
    Xs_tr, ys_tr, Xs_te, ys_te = _split(Xs_id, ys_id, seed=args.seed)
    sign_model = _fit_logreg(Xs_tr, ys_tr)
    sign_id_acc, sign_id_n = _acc(sign_model, Xs_te, ys_te)
    sign_ood_acc, sign_ood_n = _acc(sign_model, Xs_ood, ys_ood)

    # ---- ORDER probe ----
    Xo_id, yo_id = build_order_dataset(z_id, v_id, m_id, seed=args.seed)
    Xo_ood, yo_ood = build_order_dataset(z_ood, v_ood, m_ood, seed=args.seed + 1)
    Xo_tr, yo_tr, Xo_te, yo_te = _split(Xo_id, yo_id, seed=args.seed)
    order_model = _fit_logreg(Xo_tr, yo_tr)
    order_id_acc, order_id_n = _acc(order_model, Xo_te, yo_te)
    order_ood_acc, order_ood_n = _acc(order_model, Xo_ood, yo_ood)

    backend = "sklearn.LogisticRegression" if _HAVE_SKLEARN else "torch nn.Linear"
    print(f"\n[probe] backend = {backend}")
    print(f"[probe] SIGN  train={len(ys_tr)}  in-dist-test={sign_id_n}  ood={sign_ood_n}"
          f"  | base-rate(neg) in-dist={ys_id.mean():.3f} ood={ys_ood.mean():.3f}")
    print(f"[probe] ORDER train={len(yo_tr)}  in-dist-test={order_id_n}  ood={order_ood_n}"
          f"  | base-rate(<) in-dist={yo_id.mean():.3f} ood={yo_ood.mean():.3f}")

    print("\n# ORDER / SIGN linear-probe accuracy vs magnitude\n")
    print("| probe | in-dist acc | OOD acc | in-dist majority | OOD majority |")
    print("|---|---|---|---|---|")
    print(f"| SIGN  (value_i < 0)            | {sign_id_acc:.3f} | {sign_ood_acc:.3f} "
          f"| {_majority_acc(ys_te):.3f} | {_majority_acc(ys_ood):.3f} |")
    print(f"| ORDER (value_i < value_j)      | {order_id_acc:.3f} | {order_ood_acc:.3f} "
          f"| {_majority_acc(yo_te):.3f} | {_majority_acc(yo_ood):.3f} |")
    print("\nDONE", flush=True)

    return {
        "sign_id": sign_id_acc, "sign_ood": sign_ood_acc,
        "order_id": order_id_acc, "order_ood": order_ood_acc,
    }


if __name__ == "__main__":
    main()
