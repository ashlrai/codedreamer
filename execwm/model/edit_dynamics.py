"""Edit-conditioned dynamics with a divergence head (M3 step-3).

Why this exists
---------------
The earlier "WM-as-scorer" experiment failed because re-simulating an *edited*
program from scratch in latent space rolls the single-step dynamics over the
whole program, and the per-step error compounds over long horizons (see
``PLAN_M3.md`` §2 and the rollout-horizon curve in M1). The fix here is to stop
re-simulating: an edit usually perturbs execution at exactly one place and leaves
the rest of the trace identical. So instead of unrolling, we predict, *per base
step*, two things directly from the base state and the edit:

  (a) ``p_div[t]`` — the probability that the edited program's state at step ``t``
      differs from the base program's state at ``t`` (the "where does the edit
      bite" signal that lets a planner re-simulate only the divergence), and
  (b) the *changed* state at step ``t`` — the existing M1 grounding heads applied
      to a FiLM-conditioned (per-slot ``gamma``/``beta`` from the edit embedding)
      version of the base latent.

There is **no latent rollout** here — each edited state index is predicted
independently from the corresponding base state index plus the edit embedding.
That is the whole point: it sidesteps the compounding-error failure mode.

Honest compromise (documented, not faked)
------------------------------------------
The conditioning is *index-aligned*: the predicted edited state at base index
``t`` is decoded from the base latent at index ``t``. When an edit changes
control flow the two traces shift relative to one another after the divergence,
so for indices past the divergence the index alignment no longer corresponds to
"the same point of execution". We handle this honestly:

* The divergence label uses a *length-aware* rule — if the edited trace is
  shorter than the base trace at index ``t`` (or vice-versa), that index counts
  as diverged. So the divergence head still learns the correct *first* divergence
  point even for control-flow edits.
* The grounding CE for the changed state is trained/evaluated only on indices
  where the edited trace actually has a state (``edit_valid``). Past a
  control-flow divergence the index-aligned target may no longer be the
  semantically-corresponding state; this is a known limitation of the
  index-aligned framing and is the reason the *divergence point* (which the head
  predicts robustly) is the primary planning signal, with the changed-state
  decode as a secondary, best-effort head.

The per-slot grounding heads' OUTPUT interface is UNCHANGED (we reuse
``GroundedLatentWM``'s heads verbatim), so every M1/M2 metric and linear probe
applies to the predicted edited states without modification.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.action_codec import ALL_OPS
from ..substrate.edits import EditKind
from .world_model import (GroundedLatentWM, GroundingHeads, LatentDynamics,
                          ModelConfig, StateEncoder, ValueEmbedding,
                          grounding_loss)


# ---------------------------------------------------------------------------
# Edit embedding
# ---------------------------------------------------------------------------


class EditEncoder(nn.Module):
    """Embed an :class:`~execwm.substrate.edits.Edit` (in its :class:`EditCodec`
    integer-field form) into one vector, mirroring ``ActionEncoder``'s
    structured-field embedding style: a sum of per-field embeddings followed by a
    residual MLP. Only the field(s) relevant to the edit kind carry signal; the
    rest sit at sentinel ids (``none_op``/``none_reg``/``slot=none``), which simply
    map to fixed learned vectors.

    Field id ranges (see ``EditCodec``):
      kind  in {0..len(EditKind)-1}
      index in {0..max_program_len-1}  (assumed <= cfg.max_pc; see note below)
      op    in {0..len(ALL_OPS)}       (len(ALL_OPS) == none_op sentinel)
      dst   in {0..num_regs}           (num_regs == none_reg sentinel)
      slot  in {0,1,2}                 (none/a/b)
      reg   in {0..num_regs}           (num_regs == none_reg sentinel)
      imm   (sign, digits) via the shared ValueEmbedding

    Note: ``index`` is embedded with an ``cfg.max_pc + 1`` table, which assumes the
    edit codec's ``max_program_len <= cfg.max_pc + 1`` (true for the project
    defaults, where both are 256). The build helper in ``train_edit`` enforces a
    matching ``max_pc``.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.kind = nn.Embedding(len(EditKind), d)
        self.index = nn.Embedding(cfg.max_pc + 1, d)
        self.op = nn.Embedding(len(ALL_OPS) + 1, d)     # +1 none_op sentinel
        self.dst = nn.Embedding(cfg.num_regs + 1, d)    # +1 none_reg sentinel
        self.slot = nn.Embedding(3, d)                  # none / a / b
        self.reg = nn.Embedding(cfg.num_regs + 1, d)    # +1 none_reg sentinel
        self.value = ValueEmbedding(cfg.base, cfg.max_digits, d)
        self.mlp = nn.Sequential(nn.Linear(d, d * cfg.ffn_mult), nn.GELU(),
                                 nn.Linear(d * cfg.ffn_mult, d))

    def forward(self, e: dict[str, torch.Tensor]) -> torch.Tensor:
        h = (self.kind(e["kind"]) + self.index(e["index"]) + self.op(e["op"])
             + self.dst(e["dst"]) + self.slot(e["slot"]) + self.reg(e["reg"])
             + self.value(e["imm_sign"], e["imm_digits"]))
        return h + self.mlp(h)


# ---------------------------------------------------------------------------
# Conditioning blocks
# ---------------------------------------------------------------------------


class FiLMConditioner(nn.Module):
    """Per-slot FiLM (gamma, beta) generated from the edit embedding.

    A learned per-slot embedding is added to the (broadcast) edit embedding and an
    MLP produces ``(gamma, beta)`` for every slot. The final layer is zero-init so
    that at initialization ``gamma == beta == 0`` and the conditioned latent equals
    the base latent (``(1 + gamma) * z + beta == z``) — a stable identity start,
    matching the residual philosophy of ``LatentDynamics``.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.slot_emb = nn.Embedding(cfg.num_slots, d)
        self.to_film = nn.Sequential(
            nn.Linear(d, d * cfg.ffn_mult), nn.GELU(),
            nn.Linear(d * cfg.ffn_mult, 2 * d))
        nn.init.zeros_(self.to_film[-1].weight)
        nn.init.zeros_(self.to_film[-1].bias)
        self.register_buffer("slot_idx", torch.arange(cfg.num_slots),
                             persistent=False)

    def forward(self, edit_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # edit_emb: (N, d) -> gamma, beta: (N, S, d)
        cond = edit_emb[:, None, :] + self.slot_emb(self.slot_idx)[None]
        gamma, beta = self.to_film(cond).chunk(2, dim=-1)
        return gamma, beta


class DivergenceHead(nn.Module):
    """Per-step divergence logit: P(edited state at this step differs from base).

    Pools the base per-slot latent over slots and concatenates the edit embedding,
    then a shallow MLP -> one logit. ``sigmoid(logit)`` lives in [0, 1]."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        d = cfg.d_model
        self.mlp = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, z_base: torch.Tensor, edit_emb: torch.Tensor) -> torch.Tensor:
        # z_base: (N, S, d)  edit_emb: (N, d) -> (N,) logit
        pooled = z_base.mean(dim=1)
        h = torch.cat([pooled, edit_emb], dim=-1)
        return self.mlp(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Full edit-conditioned world model
# ---------------------------------------------------------------------------


class EditConditionedWM(nn.Module):
    """Wraps a :class:`GroundedLatentWM` and adds edit-conditioned prediction.

    Reuses the wrapped model's ``encoder`` and ``heads`` verbatim (so the M1/M2
    grounding interface and linear probes apply unchanged), and adds:
      * ``edit_encoder``  — embed the edit action,
      * ``film``          — per-slot FiLM conditioning from the edit embedding,
      * ``edit_dynamics`` — a slot-mixing transformer (same block as
        ``LatentDynamics``, with the edit embedding injected as the "action") to
        let conditioned slots interact,
      * ``divergence``    — the per-step divergence head.

    The wrapped model's own ``dynamics``/``target_encoder`` are left in place but
    unused by this module (kept so the wrap is a faithful ``GroundedLatentWM``).
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.wm = GroundedLatentWM(cfg)
        self.edit_encoder = EditEncoder(cfg)
        self.film = FiLMConditioner(cfg)
        self.edit_dynamics = LatentDynamics(cfg)
        self.divergence = DivergenceHead(cfg)

    # -- reused M1 interface --------------------------------------------------

    def encode(self, s: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.wm.encode(s)

    def heads(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.wm.heads(z)

    def embed_edit(self, e: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.edit_encoder(e)

    # -- edit-conditioned prediction -----------------------------------------

    def predict_edited_latent(self, z_base: torch.Tensor,
                              edit_emb: torch.Tensor) -> torch.Tensor:
        """FiLM-modulate the base per-slot latent on the edit, then mix slots.

        z_base: (N, S, d)   edit_emb: (N, d) -> (N, S, d)."""
        gamma, beta = self.film(edit_emb)
        z = (1.0 + gamma) * z_base + beta
        return self.edit_dynamics(z, edit_emb)

    def forward(self, base_state: dict[str, torch.Tensor],
                edit_emb: torch.Tensor) -> dict[str, Any]:
        """Predict the edited state and divergence at a batch of base steps.

        base_state: a flattened state-field dict, each value (N, *shape).
        edit_emb:   (N, d) edit embedding aligned to each base step (an episode's
                    edit embedding broadcast across its steps).
        Returns ``logits`` (grounding head dict), ``div_logit`` (N,), ``p_div``
        (N,) in [0, 1], and the latents.
        """
        z_base = self.encode(base_state)                 # (N, S, d)
        z_edit = self.predict_edited_latent(z_base, edit_emb)
        logits = self.heads(z_edit)
        div_logit = self.divergence(z_base, edit_emb)    # (N,)
        return {
            "logits": logits,
            "div_logit": div_logit,
            "p_div": torch.sigmoid(div_logit),
            "z_base": z_base,
            "z_edit": z_edit,
        }


# ---------------------------------------------------------------------------
# Divergence label + loss
# ---------------------------------------------------------------------------


def true_divergence_mask(base_trace: Any, edited_trace: Any,
                         scodec: Any | None = None) -> np.ndarray:
    """Per-base-step boolean mask: does the edited state differ from the base?

    Returns an array of length ``len(base_trace.states)``. Index ``t`` is True iff
    the edited program's state at step ``t`` differs from the base program's state
    at ``t``. Index 0 (the shared ``init_state``) is therefore always False.

    Length-aware: if the edited trace has no state at index ``t`` (a control-flow
    edit made it shorter), that index counts as diverged (True). This is what lets
    the divergence head locate the first divergence even for control-flow edits.

    Equality semantics match the dataset's :func:`traces_equivalent` (dataclass
    ``MachineState`` equality, which agrees with the codec's exact-match here since
    UNDEF registers are ``None`` in both). Pass ``scodec`` to instead use the
    codec's UNDEF-masked ``exact_match`` (the operational grading rule)."""
    base = base_trace.states
    edited = edited_trace.states
    n_base, n_edit = len(base), len(edited)
    mask = np.zeros(n_base, dtype=bool)
    for t in range(n_base):
        if t >= n_edit:
            mask[t] = True
        elif scodec is not None:
            mask[t] = not scodec.exact_match(scodec.encode(base[t]),
                                             scodec.encode(edited[t]))
        else:
            mask[t] = base[t] != edited[t]
    return mask


def edit_loss(out: dict, div_target: torch.Tensor, edited_tgt: dict, *,
              base_valid: torch.Tensor | None = None,
              edit_valid: torch.Tensor | None = None,
              w_div: float = 1.0, w_ground: float = 1.0) -> tuple[torch.Tensor, dict]:
    """Combined edit objective: divergence BCE + grounding CE on edited states.

    out:        forward() output (flattened over N = sum of base-step counts).
    div_target: (N,) float in {0,1}, the true per-step divergence mask.
    edited_tgt: state-field label dict, each value (N, *shape) — the edited state
                aligned to each base step.
    base_valid: (N,) bool over which steps to score divergence (default: all).
    edit_valid: (N,) bool over which steps the edited state exists & is aligned;
                grounding CE is computed only there (default: all).
    """
    div_logit = out["div_logit"]
    if base_valid is None:
        base_valid = torch.ones_like(div_logit, dtype=torch.bool)
    if edit_valid is None:
        edit_valid = torch.ones_like(div_logit, dtype=torch.bool)

    bsel = base_valid
    L_div = F.binary_cross_entropy_with_logits(
        div_logit[bsel], div_target[bsel].float())

    gsel = edit_valid
    if gsel.any():
        L_ground = grounding_loss({k: v[gsel] for k, v in out["logits"].items()},
                                  {k: v[gsel] for k, v in edited_tgt.items()})
    else:
        L_ground = div_logit.new_zeros(())

    total = w_div * L_div + w_ground * L_ground
    metrics = {"loss": float(total.detach()), "L_div": float(L_div.detach()),
               "L_ground": float(L_ground.detach())}
    return total, metrics
