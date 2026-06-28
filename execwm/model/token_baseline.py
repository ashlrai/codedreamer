"""Token-space baseline — the apples-to-apples control for the latent thesis.

Meta's *Code World Model* and friends model program execution by predicting the
next machine state as **text/tokens** with a vanilla autoregressive Transformer.
This module builds exactly that control: a decoder-only causal Transformer that,
given ``[current-state tokens ⊕ action tokens]`` as a prompt, generates the
``next-state tokens`` left-to-right, teacher-forced at train time and greedily
decoded at eval. It deliberately does **not** use the slotted grounded latent of
``world_model.py`` — that contrast is the whole experiment.

The trick that keeps the comparison clean is the serialization. The state/action
codecs (``state_codec.py`` / ``action_codec.py``) already turn a state into a
bundle of fixed-shape integer *label* arrays; those labels are themselves a
lossless, reversible serialization. We simply **flatten** them, in one fixed
canonical order, into a single integer token sequence. Because every field has a
fixed shape, *position determines field* — no separator tokens are needed inside
a record, and the inverse (``tokens_to_state_labels``) is a pure slice-and-reshape.
Grading then reuses the identical exact-match rule the latent model is judged by
(``exact_match_labels``), so the only thing that differs between the two models is
the architecture, not the data, the targets, or the metric.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.action_codec import ALL_OPS
from ..substrate.vm import VType
from .world_model import valued_mask

# ---------------------------------------------------------------------------
# Vocabulary: a handful of reserved special tokens, then a single contiguous
# block of "integer value" tokens. Every field value (a digit, a sign, a type,
# a register index, a pc class, ...) is a non-negative int, so value v maps to
# token id (v + VALUE_OFFSET). Position fixes the field's meaning, so one shared
# value block is both sufficient and 100% reversible.
# ---------------------------------------------------------------------------

PAD, BOS, SEP, EOS = 0, 1, 2, 3
VALUE_OFFSET = 4
_NUM_SPECIAL = VALUE_OFFSET


_STATE_KEYS = ("reg_type", "reg_sign", "reg_digits", "heap_sign", "heap_digits",
               "pc", "halted", "error")


@dataclass
class SerializerDims:
    """Everything needed to (re)build a :class:`TokenSerializer` without codecs."""
    num_regs: int
    num_cells: int
    max_digits: int
    base: int
    max_pc: int
    num_lists: int
    n_ops: int
    n_types: int


class TokenSerializer:
    """Flatten codec label dicts into a flat integer token sequence and back.

    Construct from the live codecs (``TokenSerializer(scodec, acodec)``) or from
    saved dims (:meth:`from_dims`) at load time. All methods are batch-first:
    they accept/return label dicts whose fields carry a leading ``N`` dimension.
    """

    def __init__(self, scodec=None, acodec=None, *, dims: SerializerDims | None = None):
        if dims is None:
            if scodec is None or acodec is None:
                raise ValueError("provide (scodec, acodec) or dims=")
            dims = SerializerDims(
                num_regs=scodec.num_regs,
                num_cells=scodec.num_cells,
                max_digits=scodec.codec.max_digits,
                base=scodec.codec.base,
                max_pc=scodec.codec.max_pc,
                num_lists=scodec.config.num_lists,
                n_ops=len(ALL_OPS),
                n_types=len(VType),
            )
        self.dims = dims
        d = dims
        self.R, self.C, self.D = d.num_regs, d.num_cells, d.max_digits
        self.base, self.max_pc = d.base, d.max_pc
        # largest field value that can ever appear (sentinels included):
        #   pc/target -> max_pc, dst/reg -> num_regs, list_id -> num_lists,
        #   op -> n_ops-1, digit -> base-1, type -> n_types-1
        self.max_value = max(d.max_pc, d.num_regs, d.num_lists,
                             d.n_ops - 1, d.base - 1, d.n_types - 1, 1)
        self.vocab_size = _NUM_SPECIAL + self.max_value + 1

        # canonical state-token layout (field -> width); position => field.
        self.T_state = (2 * self.R + self.R * self.D
                        + self.C + self.C * self.D + 3)
        # action layout: op,dst,a_kind,a_reg,a_sign,a_digits(D),
        #                b_kind,b_reg,b_sign,b_digits(D),list_id,target
        self.T_action = 10 + 2 * self.D
        # decoder-only sequence: BOS s a SEP  ns EOS
        self.T_prompt = 2 + self.T_state + self.T_action
        self.T_full = self.T_prompt + self.T_state + 1

    @classmethod
    def from_dims(cls, **kw) -> "TokenSerializer":
        return cls(dims=SerializerDims(**kw))

    # -- state <-> tokens ----------------------------------------------------

    def state_to_tokens(self, s: dict) -> torch.Tensor:
        """(batched label dict) -> LongTensor (N, T_state) of token ids."""
        N = s["pc"].shape[0]
        chunks = [
            s["reg_type"].reshape(N, self.R),
            s["reg_sign"].reshape(N, self.R),
            s["reg_digits"].reshape(N, self.R * self.D),
            s["heap_sign"].reshape(N, self.C),
            s["heap_digits"].reshape(N, self.C * self.D),
            s["pc"].reshape(N, 1),
            s["halted"].reshape(N, 1),
            s["error"].reshape(N, 1),
        ]
        flat = torch.cat([c.long() for c in chunks], dim=1)
        return flat + VALUE_OFFSET

    def tokens_to_state_labels(self, tokens: torch.Tensor) -> dict:
        """Inverse of :meth:`state_to_tokens`. (N, T_state) -> label dict.

        Values are clamped into each field's legal range so that even an
        untrained model's junk tokens parse into a gradeable (if wrong) state.
        """
        v = (tokens.long() - VALUE_OFFSET).clamp_min(0)
        N = v.shape[0]
        R, C, D = self.R, self.C, self.D
        i = 0
        reg_type = v[:, i:i + R]; i += R
        reg_sign = v[:, i:i + R]; i += R
        reg_digits = v[:, i:i + R * D].reshape(N, R, D); i += R * D
        heap_sign = v[:, i:i + C]; i += C
        heap_digits = v[:, i:i + C * D].reshape(N, C, D); i += C * D
        pc = v[:, i]; i += 1
        halted = v[:, i]; i += 1
        error = v[:, i]; i += 1
        return {
            "reg_type": reg_type.clamp_(0, self.dims.n_types - 1),
            "reg_sign": reg_sign.clamp_(0, 1),
            "reg_digits": reg_digits.clamp_(0, self.base - 1),
            "heap_sign": heap_sign.clamp_(0, 1),
            "heap_digits": heap_digits.clamp_(0, self.base - 1),
            "pc": pc.clamp_(0, self.max_pc),
            "halted": halted.clamp_(0, 1),
            "error": error.clamp_(0, 1),
        }

    # -- action -> tokens (decode not needed for the baseline) ---------------

    def action_to_tokens(self, a: dict) -> torch.Tensor:
        """(batched action label dict) -> LongTensor (N, T_action)."""
        N = a["op"].shape[0]
        col = lambda k: a[k].reshape(N, 1)
        chunks = [
            col("op"), col("dst"), col("a_kind"), col("a_reg"), col("a_sign"),
            a["a_digits"].reshape(N, self.D),
            col("b_kind"), col("b_reg"), col("b_sign"),
            a["b_digits"].reshape(N, self.D),
            col("list_id"), col("target"),
        ]
        flat = torch.cat([c.long() for c in chunks], dim=1)
        return flat + VALUE_OFFSET

    # -- full decoder-only sequences -----------------------------------------

    def build_prompt(self, state: dict, action: dict) -> torch.Tensor:
        """[BOS] + state tokens + action tokens + [SEP]  -> (N, T_prompt)."""
        s_tok = self.state_to_tokens(state)
        a_tok = self.action_to_tokens(action)
        N = s_tok.shape[0]
        dev = s_tok.device
        bos = torch.full((N, 1), BOS, dtype=torch.long, device=dev)
        sep = torch.full((N, 1), SEP, dtype=torch.long, device=dev)
        return torch.cat([bos, s_tok, a_tok, sep], dim=1)

    def build_training_sequence(self, state: dict, action: dict,
                                next_state: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (input_ids, labels) for teacher forcing.

        full = [BOS s a SEP  ns EOS]; input = full[:-1], labels = full[1:] with
        every position outside the next-state (+EOS) region set to -100 so the
        loss is computed only on the generated continuation.
        """
        prompt = self.build_prompt(state, action)              # (N, P)
        ns_tok = self.state_to_tokens(next_state)              # (N, T_state)
        N, dev = prompt.shape[0], prompt.device
        eos = torch.full((N, 1), EOS, dtype=torch.long, device=dev)
        full = torch.cat([prompt, ns_tok, eos], dim=1)         # (N, T_full)
        input_ids = full[:, :-1]
        labels = full[:, 1:].clone()
        P = self.T_prompt
        labels[:, :P - 1] = -100                               # supervise ns + EOS only
        return input_ids, labels


# ---------------------------------------------------------------------------
# The model: a plain decoder-only causal Transformer over the token stream.
# ---------------------------------------------------------------------------


@dataclass
class TokenModelConfig:
    vocab_size: int
    max_seq_len: int
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    ffn_mult: int = 4
    dropout: float = 0.0


class TokenBaseline(nn.Module):
    """Decoder-only causal Transformer: next-token prediction over the flat
    state/action token stream. No grounded latent, no slots — the control."""

    def __init__(self, cfg: TokenModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.tok_emb = nn.Embedding(cfg.vocab_size, d)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, d)
        layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, dim_feedforward=d * cfg.ffn_mult,
            dropout=cfg.dropout, batch_first=True, activation="gelu",
            norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, cfg.n_layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size)
        self.register_buffer("pos_idx", torch.arange(cfg.max_seq_len),
                             persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        x = self.tok_emb(input_ids) + self.pos_emb(self.pos_idx[:T])[None]
        mask = torch.triu(torch.full((T, T), float("-inf"), device=input_ids.device),
                          diagonal=1)
        h = self.transformer(x, mask=mask, is_causal=True)
        return self.head(self.norm(h))                         # (B, T, vocab)

    @torch.no_grad()
    def greedy_decode(self, prompt: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Autoregressively generate ``n_steps`` tokens after ``prompt``.

        Generation is restricted to the value-token block (special tokens are
        never emitted), guaranteeing the output parses back into a state.
        """
        self.eval()
        seq = prompt
        for _ in range(n_steps):
            logits = self.forward(seq)[:, -1]                  # (B, vocab)
            nxt = logits[:, VALUE_OFFSET:].argmax(-1) + VALUE_OFFSET
            seq = torch.cat([seq, nxt[:, None]], dim=1)
        return seq[:, prompt.shape[1]:]                        # (B, n_steps)


def build_token_baseline(serializer: TokenSerializer, **model_kw) -> TokenBaseline:
    cfg = TokenModelConfig(vocab_size=serializer.vocab_size,
                           max_seq_len=serializer.T_full, **model_kw)
    return TokenBaseline(cfg)


# ---------------------------------------------------------------------------
# Inference + grading helpers (label-dict space, so the bench can reuse the
# exact-match rule the latent model is judged by).
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_next_labels(model: TokenBaseline, serializer: TokenSerializer,
                        state: dict, action: dict, device) -> dict:
    """Greedy-decode the next-state token sequence for a batch and parse it back
    into a state label dict (argmax'd ints), ready for ``exact_match_labels``."""
    model.eval()
    state = {k: v.to(device) for k, v in state.items()}
    action = {k: v.to(device) for k, v in action.items()}
    prompt = serializer.build_prompt(state, action)
    gen = model.greedy_decode(prompt, serializer.T_state)
    return serializer.tokens_to_state_labels(gen)


@torch.no_grad()
def teacher_forced_token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of supervised positions whose argmax token is correct."""
    pred = logits.argmax(-1)
    sel = labels != -100
    if sel.sum() == 0:
        return 0.0
    return (pred[sel] == labels[sel]).float().mean().item()


@torch.no_grad()
def per_var_accuracy_labels(pred: dict, tgt: dict) -> torch.Tensor:
    """Mean per-register correctness (type+sign+digits) over valued registers.

    Mirrors ``world_model.per_var_accuracy`` but operates on argmax'd label
    dicts (what greedy decoding yields) rather than logits."""
    mask = valued_mask(tgt["reg_type"])
    correct = ((pred["reg_type"] == tgt["reg_type"])
               & (pred["reg_sign"] == tgt["reg_sign"])
               & (pred["reg_digits"] == tgt["reg_digits"]).all(-1)) & mask
    return correct.sum().float() / mask.sum().clamp_min(1).float()
