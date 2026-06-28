"""Counterfactual intervention metric — the M2 causal-reasoning centerpiece.

The thesis of the world model is that it learns the *transition function*, not a
lookup table over training trajectories. We test that directly with interventions
on real transitions, using the VM ``step`` as a perfect oracle for the truth:

* ``do(reg = v')`` — take a real ``(s_t, a_t)``, overwrite one *defined* register
  with a new value, and let the VM compute the true counterfactual next state
  ``step(s_t', a_t)``. The model must predict it from ``encode(s_t') + a_t``.
* ``do(replace action)`` — keep ``s_t`` but swap ``a_t`` for a different *valid*
  instruction ``a'`` (reading only defined registers / immediates), true next is
  ``step(s_t, a')``.

Both kinds of ``(state, action)`` pairs are *off the trajectory distribution* the
model trained on: no generated program would necessarily have produced that exact
state-with-that-action. Predicting them correctly therefore requires modeling the
actual transition, which is the whole point. We compare against an ``identity``
baseline ("predict no change") as a trivial lower bound to beat.

Everything here is graded with the same exact-match / per-variable metrics used in
training (``execwm.model.world_model``), so the numbers are directly comparable to
the in-distribution single-step exact-match.
"""

from __future__ import annotations

import random

import numpy as np
import torch

from ..data.action_codec import ActionCodec
from ..data.dataset import collect_examples
from ..data.state_codec import CodecConfig, EncodeError, StateCodec
from ..model.world_model import exact_match, per_var_accuracy
from ..substrate.generators import GenSpec
from ..substrate.vm import (ARITH_OPS, CMP_OPS, Instr, MachineState, Op, VType,
                            step)

# Opcodes that are "safe" replacements for action interventions: they read only
# the operands we hand them (which we keep referencing defined regs / immediates)
# and never trap when fed a nonzero divisor. We deliberately exclude DIV/MOD
# (div-by-zero trap), jumps (need a target / change control flow shape), and
# LOAD/STORE/HALT to keep the swapped instruction a clean local computation.
_SAFE_ACTION_OPS: tuple[Op, ...] = (
    Op.ADD, Op.SUB, Op.MUL,
    Op.LT, Op.LE, Op.EQ, Op.NE, Op.GT, Op.GE,
    Op.MOV, Op.CONST,
)

_VALUED = (VType.INT, VType.BOOL)


# ---------------------------------------------------------------------------
# State comparison (mirrors the codec's exact-match rule, on MachineState)
# ---------------------------------------------------------------------------


def _state_equal(a: MachineState, b: MachineState) -> bool:
    """True iff two states are exact-match equal, ignoring the numeric payload of
    registers that are UNDEF (their value is junk). This mirrors
    :meth:`StateCodec.exact_match` so the helpers' notion of "the counterfactual
    actually differs" matches how the model is graded."""
    if a.pc != b.pc or a.halted != b.halted or a.error != b.error:
        return False
    if a.types != b.types:
        return False
    if a.heap != b.heap:
        return False
    for name, vtype in a.types.items():
        if vtype in _VALUED and a.regs[name] != b.regs[name]:
            return False
    return True


def _read_regs(instr: Instr) -> set[str]:
    """Register names the instruction *reads* (so intervening on one changes the
    computed result, not just a persisted slot)."""
    reads: set[str] = set()

    def add(x) -> None:
        if isinstance(x, str):
            reads.add(x)

    op = instr.op
    if op is Op.MOV:
        add(instr.a)
    elif op in ARITH_OPS or op in CMP_OPS:
        add(instr.a)
        add(instr.b)
    elif op in (Op.JZ, Op.JNZ):
        add(instr.a)
    elif op is Op.LOAD:
        add(instr.a)
    elif op is Op.STORE:
        add(instr.a)
        add(instr.b)
    # CONST, JMP, HALT read no registers
    return reads


# ---------------------------------------------------------------------------
# Sampling real base transitions
# ---------------------------------------------------------------------------


def sample_base_transitions(spec: GenSpec, n: int, seed: int,
                            codec_cfg: CodecConfig | None = None,
                            ) -> list[tuple[MachineState, Instr]]:
    """Sample ``n`` real ``(state_t, instr_t)`` transitions from freshly generated,
    *terminating, encodable* examples — random valid steps across many programs."""
    codec_cfg = codec_cfg or CodecConfig(max_digits=6, base=10, max_pc=256)
    cfg = spec.config()
    scodec = StateCodec(cfg, codec_cfg)
    acodec = ActionCodec(cfg, codec_cfg)
    rng = random.Random(seed)

    # Collect enough examples that random step-picking can yield n pairs.
    # Each terminating example contributes len(trace) >= 1 steps.
    n_examples = max(8, n // 3 + 4)
    examples, _ = collect_examples(spec, n_examples, lambda ex: True,
                                   seed, scodec, acodec)

    pairs: list[tuple[MachineState, Instr]] = []
    # Pick random (example, step) until we have n (with replacement across the
    # example pool, without re-picking the same step).
    seen: set[tuple[int, int]] = set()
    guard = 0
    max_guard = n * 200 + 1000
    while len(pairs) < n and guard < max_guard:
        guard += 1
        ex = rng.choice(examples)
        ex_id = id(ex)
        t = rng.randrange(len(ex.trace.actions))
        key = (ex_id, t)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((ex.trace.states[t], ex.trace.actions[t]))
    return pairs


# ---------------------------------------------------------------------------
# Interventions
# ---------------------------------------------------------------------------


def intervene_register(state: MachineState, instr: Instr, rng: random.Random,
                       value_range: tuple[int, int] = (-10, 10),
                       ) -> tuple[MachineState, Instr] | None:
    """``do(reg = v')``: reassign one currently-defined register to a new value so
    the VM's next state actually differs.

    Returns ``(modified_state, instr)`` whose ``step`` result is provably distinct
    from the un-intervened transition, or ``None`` if no such intervention exists
    (e.g. no defined registers). Registers the instruction *reads* are preferred
    (they change the computed result); otherwise any non-overwritten defined
    register works (its changed value persists into the next state). The
    instruction is returned unchanged.
    """
    defined = [n for n, t in state.types.items()
               if t in _VALUED and state.regs[n] is not None]
    if not defined:
        return None

    reads = _read_regs(instr) & set(defined)
    read_list = list(reads)
    other = [n for n in defined if n not in reads]
    rng.shuffle(read_list)
    rng.shuffle(other)
    candidates = read_list + other  # prefer causally-read registers

    orig_next = step(state, instr)
    lo, hi = value_range
    for name in candidates:
        old = state.regs[name]
        values = [v for v in range(lo, hi + 1) if v != old]
        rng.shuffle(values)
        for v in values:
            mod = state.copy()
            mod.regs[name] = v
            mod.types[name] = VType.INT
            if not _state_equal(step(mod, instr), orig_next):
                return mod, instr
    return None


def intervene_action(state: MachineState, instr: Instr, rng: random.Random,
                     imm_range: tuple[int, int] = (-9, 9),
                     max_tries: int = 32,
                     ) -> tuple[MachineState, Instr] | None:
    """``do(replace action)``: keep ``state``, swap ``instr`` for a different valid
    instruction whose execution reads only currently-defined registers (or
    immediates) — so it never traps on an undefined read.

    Returns ``(state, instr')`` whose VM next state differs from the original
    transition, or ``None`` if no differing valid swap is found.
    """
    reg_names = list(state.regs.keys())
    defined = [n for n, t in state.types.items()
               if t in _VALUED and state.regs[n] is not None]
    if not reg_names:
        return None

    orig_next = step(state, instr)
    lo, hi = imm_range

    def operand():
        if defined and rng.random() < 0.6:
            return rng.choice(defined)
        return rng.randint(lo, hi)

    for _ in range(max_tries):
        op = rng.choice(_SAFE_ACTION_OPS)
        dst = rng.choice(reg_names)
        if op is Op.CONST:
            cand = Instr(op, dst=dst, a=rng.randint(lo, hi))
        elif op is Op.MOV:
            cand = Instr(op, dst=dst, a=operand())
        else:  # arithmetic / comparison
            cand = Instr(op, dst=dst, a=operand(), b=operand())
        if cand == instr:
            continue
        # cand reads only defined regs / immediates -> step never raises.
        if not _state_equal(step(state, cand), orig_next):
            return state, cand
    return None


def make_register_pairs(base_pairs: list[tuple[MachineState, Instr]],
                        rng: random.Random,
                        value_range: tuple[int, int] = (-10, 10),
                        ) -> list[tuple[MachineState, Instr]]:
    """Apply ``intervene_register`` to each base transition, dropping the (rare)
    pairs with no valid intervention."""
    out = []
    for state, instr in base_pairs:
        res = intervene_register(state, instr, rng, value_range)
        if res is not None:
            out.append(res)
    return out


def make_action_pairs(base_pairs: list[tuple[MachineState, Instr]],
                      rng: random.Random,
                      ) -> list[tuple[MachineState, Instr]]:
    """Apply ``intervene_action`` to each base transition, dropping any with no
    valid swap."""
    out = []
    for state, instr in base_pairs:
        res = intervene_action(state, instr, rng)
        if res is not None:
            out.append(res)
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _stack(dicts: list[dict[str, np.ndarray]], device) -> dict[str, torch.Tensor]:
    """Stack a list of per-example codec dicts into batched int64 tensors."""
    out: dict[str, torch.Tensor] = {}
    for k in dicts[0]:
        arr = np.stack([d[k] for d in dicts])
        out[k] = torch.from_numpy(arr).to(device).long()
    return out


@torch.no_grad()
def evaluate_counterfactual(model, scodec: StateCodec, acodec: ActionCodec,
                            pairs: list[tuple[MachineState, Instr]], device,
                            ) -> dict:
    """Grade the model on intervened ``(state, action)`` pairs against the VM truth.

    For each pair: the true next state is ``step(state, action)``; the model
    predicts next from ``encode(state) + action``. Pairs whose state or true-next
    fall outside the codec's representable range are skipped. Returns
    ``{n, exact_match, per_var, n_skipped}`` with metrics in ``[0, 1]``.
    """
    enc_s: list[dict] = []
    enc_a: list[dict] = []
    enc_t: list[dict] = []
    n_skipped = 0
    for state, instr in pairs:
        try:
            true_next = step(state, instr)
            s = scodec.encode(state).as_dict()
            a = acodec.encode(instr).as_dict()
            t = scodec.encode(true_next).as_dict()
        except EncodeError:
            n_skipped += 1
            continue
        enc_s.append(s)
        enc_a.append(a)
        enc_t.append(t)

    n = len(enc_s)
    if n == 0:
        return {"n": 0, "exact_match": 0.0, "per_var": 0.0, "n_skipped": n_skipped}

    was_training = model.training
    model.eval()
    s_dict = _stack(enc_s, device)
    a_dict = _stack(enc_a, device)
    t_dict = _stack(enc_t, device)
    z = model.encode(s_dict)
    z_next = model.predict_next(z, a_dict)
    logits = model.heads(z_next)
    em = exact_match(logits, t_dict).float().mean().item()
    pv = per_var_accuracy(logits, t_dict).item()
    if was_training:
        model.train()
    return {"n": n, "exact_match": em, "per_var": pv, "n_skipped": n_skipped}


def identity_baseline(pairs: list[tuple[MachineState, Instr]]) -> float:
    """Trivial "predict no change" lower bound: the fraction of pairs where the
    intervened next state equals the intervened current state. Because most
    instructions advance the program counter, this is near zero — any real causal
    model must beat it."""
    if not pairs:
        return 0.0
    hits = 0
    for state, instr in pairs:
        if _state_equal(step(state, instr), state):
            hits += 1
    return hits / len(pairs)


# ---------------------------------------------------------------------------
# Demo: train a small model and report the headline causal metric
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from ..train.train_m1 import TrainConfig, train

    spec = GenSpec(num_vars=4, max_depth=2, num_stmts=5,
                   max_const=5, max_input_val=5, max_loop_count=3)
    codec_cfg = CodecConfig(max_digits=6, base=10, max_pc=256)

    out = train(spec=spec, codec_cfg=codec_cfg, tc=TrainConfig(steps=400),
                n_train=1500, n_eval=300, seed=0)
    model, scodec, acodec, device = (out["model"], out["scodec"],
                                     out["acodec"], out["device"])

    rng = random.Random(123)
    base = sample_base_transitions(spec, 400, seed=7, codec_cfg=codec_cfg)
    reg_pairs = make_register_pairs(base, rng, value_range=(-9, 9))
    act_pairs = make_action_pairs(base, rng)

    reg_res = evaluate_counterfactual(model, scodec, acodec, reg_pairs, device)
    act_res = evaluate_counterfactual(model, scodec, acodec, act_pairs, device)

    print("\n=== COUNTERFACTUAL (M2) ===")
    print(f"in-distribution single-step exact-match: "
          f"{out['eval']['step_exact_match']:.4f}")
    print(f"do(reg=v')   exact-match {reg_res['exact_match']:.4f}  "
          f"per-var {reg_res['per_var']:.4f}  "
          f"(n={reg_res['n']}, skipped={reg_res['n_skipped']})  "
          f"vs identity baseline {identity_baseline(reg_pairs):.4f}")
    print(f"do(action)   exact-match {act_res['exact_match']:.4f}  "
          f"per-var {act_res['per_var']:.4f}  "
          f"(n={act_res['n']}, skipped={act_res['n_skipped']})  "
          f"vs identity baseline {identity_baseline(act_pairs):.4f}")
