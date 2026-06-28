"""Divergence-aware planning scorer built on the edit-conditioned world model.

What this is
------------
A planning scorer that ranks edit candidates **without running the VM and without
rolling a program out in latent space**. It is the intended realisation of the M3
payoff after two earlier dead-ends:

* ``OracleScorer`` (see :mod:`execwm.plan.planner`) is exact but runs the VM on
  every candidate — no executions are saved.
* ``WorldModelScorer`` (see :mod:`execwm.plan.wm_scorer`) ran zero VM calls but
  *re-simulated the whole edited program in latent space*: it fetched each
  instruction by the model's own decoded ``pc`` and stepped the single-step
  dynamics forward. Per-step latent error compounds, so exact-match collapsed past
  horizon ~8 and the scores were too noisy to plan with.

:class:`EditConditionedWMScorer` fixes both at once using
:class:`~execwm.model.edit_dynamics.EditConditionedWM`:

1. **Zero VM calls (``.executions == 0``).** The scorer never calls
   ``run_traced``. Its ``executions`` attribute is created at ``0`` and is never
   incremented, so :func:`divergence_beam_plan` (and ``beam_plan``) fold in no
   hidden interpreter cost from scoring.

2. **No long-horizon latent rollout.** Instead of unrolling, the model predicts
   the *edited* state at each base step **directly** from the corresponding base
   latent plus a single edit embedding (FiLM conditioning + one slot-mixing
   block). Every edited-state index is produced in one batched forward — there is
   no step-to-step recurrence, so there is no compounding-error horizon to die on.
   The per-step divergence head (``p_div``) localises *where* the edit bites.

How a candidate is scored
-------------------------
A beam candidate is ``child = apply_edit(parent_program, edit)``. To score it:

* Obtain the parent program's per-step states (a list of
  :class:`~execwm.substrate.vm.MachineState`). For the **root** program these are
  its *true* executed states — :func:`divergence_beam_plan` runs the VM **once** at
  the root to seed them (the single, amortised base trace the design assumes). For
  any deeper parent we **reuse the parent's predicted edited states** (cached from
  when the parent was itself scored) — so no further VM runs happen.
* Encode those base states into per-slot latents (one batched ``encode``), embed
  the edit once, and call ``EditConditionedWM.forward`` to get the index-aligned
  predicted edited states' grounding logits and ``p_div``.
* Decode the goal-relevant predicted state (the final index) back into a
  ``MachineState`` via the grounding heads and score it with
  :func:`~execwm.plan.goal_tasks.goal_distance`.
* Cache the full predicted edited-state list under the child's signature so the
  child's own children can reuse it (the latent prediction chains forward through
  the beam without ever touching the VM).

Index-alignment caveat (honest, not faked)
-------------------------------------------
The conditioning is *index-aligned*: the predicted edited state at base index
``t`` is decoded from the base latent at index ``t``. For value/register edits
(``CHANGE_DST`` / ``CHANGE_OPERAND`` / ``CHANGE_IMM``) the control flow is
unchanged, so index ``t`` is the same execution point in both traces and the
final-index decode is the genuine edited final state. For a control-flow edit
(``CHANGE_OP`` on a conditional jump) the two traces shift relative to one another
after the divergence, so past that point the index-aligned decode no longer
corresponds to "the same point of execution". The robust signal there is the
**divergence point** the head predicts (``p_div``) plus the *changed-state* decode
near it; the final-index goal score is a best-effort estimate, which is why this
is a planning *heuristic* (ranking candidates) and every committed candidate is
still **verified on the real VM** before the search claims success. Whether the
ranking is good enough to save executions depends on the separately-trained
model's accuracy — this module only provides the (VM-free, rollout-free) plumbing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..data.state_codec import EncodeError, EncodedState, StateCodec
from ..substrate.edits import Edit, EditError, apply_edit, enumerate_valid_edits
from ..substrate.vm import (Config, Instr, MachineState, VMError, run_traced)
from .goal_tasks import Goal, GoalTask, goal_distance, satisfies
from .search_baseline import SearchResult


# ---------------------------------------------------------------------------
# Grounding-head label -> MachineState reconstruction
# ---------------------------------------------------------------------------


def state_from_labels(scodec: StateCodec,
                      labels: dict[str, Any]) -> MachineState:
    """Reconstruct a :class:`MachineState` from per-field class *labels*.

    ``labels`` holds the integer label arrays/scalars the grounding heads emit
    (the codec's exact notion of a state):
      ``reg_type`` (R,), ``reg_sign`` (R,), ``reg_digits`` (R, D),
      ``heap_sign`` (C,), ``heap_digits`` (C, D), and scalar ``pc`` / ``halted`` /
      ``error``. This is the inverse of :meth:`StateCodec.encode` 's label form, so
      feeding the true labels of a state round-trips it (an UNDEF register's
      payload comes back as ``None`` by the codec rule).
    """
    enc = EncodedState(
        reg_type=np.asarray(labels["reg_type"], dtype=np.int64),
        reg_sign=np.asarray(labels["reg_sign"], dtype=np.int64),
        reg_digits=np.asarray(labels["reg_digits"], dtype=np.int64),
        heap_sign=np.asarray(labels["heap_sign"], dtype=np.int64),
        heap_digits=np.asarray(labels["heap_digits"], dtype=np.int64),
        pc=np.asarray(int(labels["pc"]), dtype=np.int64),
        halted=np.asarray(int(labels["halted"]), dtype=np.int64),
        error=np.asarray(int(labels["error"]), dtype=np.int64),
    )
    return scodec.decode(enc)


def decode_logits_to_state(scodec: StateCodec, logits: dict[str, torch.Tensor],
                           idx: int) -> MachineState:
    """Decode batch row ``idx`` of a grounding-head ``logits`` dict to a state.

    Takes the argmax label per field at the given index and delegates to
    :func:`state_from_labels`. ``logits`` is the output of
    ``EditConditionedWM.heads`` (shapes: ``reg_*`` ``(N, R, ...)``, ``heap_*``
    ``(N, C, ...)``, ``pc``/``halted``/``error`` ``(N, ...)``).
    """
    def lab(key: str) -> np.ndarray:
        return logits[key].argmax(-1)[idx].detach().cpu().numpy().astype(np.int64)

    labels = {
        "reg_type": lab("reg_type"),
        "reg_sign": lab("reg_sign"),
        "reg_digits": lab("reg_digits"),
        "heap_sign": lab("heap_sign"),
        "heap_digits": lab("heap_digits"),
        "pc": int(logits["pc"].argmax(-1)[idx]),
        "halted": int(logits["halted"].argmax(-1)[idx]),
        "error": int(logits["error"].argmax(-1)[idx]),
    }
    return state_from_labels(scodec, labels)


# ---------------------------------------------------------------------------
# The edit-conditioned WM scorer
# ---------------------------------------------------------------------------


class EditConditionedWMScorer:
    """Score edit candidates with an :class:`EditConditionedWM` — zero VM calls.

    Construct with ``EditConditionedWMScorer(edit_model, scodec, ecodec, device)``.

    Two entry points, both keeping ``executions == 0`` for the object's whole
    lifetime (it never runs ``run_traced``):

    * ``scorer(program, init_state, goal)`` — matches the ``beam_plan`` scorer
      protocol. The bare protocol passes only the program, not the edit that
      produced it, so this path predicts the edited state from ``init_state`` under
      a *null* edit — a finite, honest fallback (used by direct callers / tests).
    * ``score_edit(parent_sig, edit, child_program, init_state, goal)`` — the rich
      path :func:`divergence_beam_plan` uses, which supplies the per-candidate edit
      and the parent's (true-at-root / predicted-deeper) base states, and caches
      the child's predicted states for its own children.
    """

    def __init__(self, edit_model: Any, scodec: StateCodec, ecodec: Any,
                 device: torch.device | str = "cpu",
                 max_base_steps: int = 64) -> None:
        self.model = edit_model
        self.scodec = scodec
        self.ecodec = ecodec
        self.device = torch.device(device)
        self.max_base_steps = max_base_steps
        self.executions = 0  # invariant: never incremented (no VM calls)
        # program-signature -> list[MachineState] (true at root, predicted deeper)
        self._cache: dict[tuple[Instr, ...], list[MachineState]] = {}

    # -- base-state cache -----------------------------------------------------

    def reset(self) -> None:
        """Clear the predicted/true base-state cache (call once per task)."""
        self._cache = {}

    def seed_base_states(self, program_sig: tuple[Instr, ...],
                         states: list[MachineState]) -> None:
        """Seed the *true* executed states of a (root) program into the cache."""
        self._cache[program_sig] = list(states)

    # -- tensor encoding ------------------------------------------------------

    def _encode_states(self, states: list[MachineState]) -> dict[str, torch.Tensor]:
        """Batch-encode a list of states into the model's (N, *) tensor dict."""
        encs = [self.scodec.encode(s).as_dict() for s in states]
        out: dict[str, torch.Tensor] = {}
        for k in encs[0]:
            arr = np.stack([np.asarray(e[k]) for e in encs])  # (N, *shape)
            out[k] = torch.as_tensor(arr, dtype=torch.long, device=self.device)
        return out

    # -- core prediction (no VM, no rollout) ----------------------------------

    @torch.no_grad()
    def _predict(self, base_states: list[MachineState],
                 edit: Edit | None) -> list[MachineState] | None:
        """Predict the index-aligned edited states directly from base states.

        Returns the decoded predicted edited states (one per base step), or
        ``None`` if a state/edit could not be codec-encoded. No VM, no rollout:
        a single batched ``forward`` over all base steps at once.
        """
        self.model.eval()
        states = (base_states[:self.max_base_steps]
                  if self.max_base_steps else base_states)
        if not states:
            return None
        try:
            base_t = self._encode_states(states)
        except EncodeError:
            return None
        n = len(states)
        d = self.model.cfg.d_model
        if edit is None:
            edit_emb = torch.zeros(n, d, device=self.device)
        else:
            try:
                ed = self.ecodec.encode(edit).as_dict()
            except (EncodeError, KeyError):
                return None
            edit_t = {k: torch.as_tensor(np.asarray(v), dtype=torch.long,
                                         device=self.device).unsqueeze(0)
                      for k, v in ed.items()}
            edit_emb = self.model.embed_edit(edit_t).expand(n, -1)
        out = self.model.forward(base_t, edit_emb)
        logits = out["logits"]
        return [decode_logits_to_state(self.scodec, logits, i) for i in range(n)]

    @staticmethod
    def _score(predicted: list[MachineState] | None, goal: Goal) -> float:
        """Goal distance of the predicted *final* state (the goal-relevant index)."""
        if not predicted:
            return float("inf")
        return goal_distance(goal, predicted[-1])

    # -- protocol entry (bare program; no edit context) -----------------------

    def __call__(self, program: list[Instr], init_state: MachineState,
                 goal: Goal) -> float:
        """``beam_plan`` scorer protocol. Predicts from ``init_state`` under a null
        edit (finite, honest fallback when no edit context is supplied)."""
        predicted = self._predict([init_state], None)
        return self._score(predicted, goal)

    # -- rich entry used by divergence_beam_plan ------------------------------

    def score_edit(self, parent_sig: tuple[Instr, ...], edit: Edit,
                   child_program: list[Instr], init_state: MachineState,
                   goal: Goal) -> float:
        """Score ``child = apply_edit(parent, edit)`` and cache its predicted states.

        Uses the parent's cached base states (true at the root, predicted deeper);
        falls back to ``[init_state]`` if the parent was never cached. Never runs
        the VM. The child's predicted edited states are cached so its children can
        reuse them (the prediction chains forward through the beam).
        """
        base_states = self._cache.get(parent_sig) or [init_state]
        predicted = self._predict(base_states, edit)
        # Cache something for the child either way so deeper candidates proceed.
        self._cache[tuple(child_program)] = predicted or [init_state]
        return self._score(predicted, goal)


# ---------------------------------------------------------------------------
# Edit-aware beam planner that carries the per-candidate edit to the scorer
# ---------------------------------------------------------------------------


def divergence_beam_plan(task: GoalTask, config: Config | None = None, *,
                         scorer: EditConditionedWMScorer, beam_width: int,
                         max_depth: int, max_executions: int,
                         verify_k: int | None = None,
                         rng=None) -> SearchResult:
    """Beam search over edits, scored by an :class:`EditConditionedWMScorer`.

    This mirrors :func:`~execwm.plan.planner.beam_plan` but resolves a protocol
    mismatch: ``beam_plan`` only passes the *candidate program* to its scorer, not
    the *edit* that produced it nor the *parent*. The edit-conditioned scorer needs
    both. So this wrapper tracks ``(program, plan, signature)`` per candidate and
    calls ``scorer.score_edit(parent_sig, edit, child, init, goal)``.

    Honest cost accounting (everything VM is counted):
      * **one** ``run_traced`` at the root to seed the scorer's true base states
        (the single base trace the design assumes; deeper candidates reuse
        *predicted* states, so the scorer adds zero VM),
      * ``run_traced`` verification runs on the best ``verify_k`` of each beam.
    ``scorer.executions`` stays ``0`` throughout (it never runs the VM); the
    reported total is ``root-run + verifications``. ``max_executions`` caps it.
    """
    config = config or task.config
    init = task.init_state
    goal = task.goal
    edit_cfg = task.edit_config
    max_steps = task.max_steps
    verify_k = beam_width if verify_k is None else verify_k

    scorer.reset()
    scorer_base = getattr(scorer, "executions", 0)
    base_runs = 0
    verifications = 0

    def total() -> int:
        return base_runs + verifications + (
            getattr(scorer, "executions", 0) - scorer_base)

    root = list(task.base_bytecode)
    root_sig = tuple(root)

    # Seed the root's TRUE executed states with ONE VM run (counted). This is the
    # only base trace the scorer needs from the interpreter; every deeper parent
    # reuses predicted states, so the scorer itself never runs the VM.
    if total() >= max_executions:
        return SearchResult(False, total(), None, 0)
    base_runs += 1
    try:
        root_trace = run_traced(root, init, max_steps=max_steps)
        scorer.seed_base_states(root_sig, list(root_trace.states))
        if satisfies(goal, root_trace):  # defensive: base already solves it
            return SearchResult(True, total(), [], 0)
    except VMError:
        scorer.seed_base_states(root_sig, [init])

    frontier: list[tuple[list[Instr], tuple, tuple[Instr, ...]]] = [
        (root, (), root_sig)]
    seen: set[tuple[Instr, ...]] = {root_sig}

    for depth in range(1, max_depth + 1):
        scored: list[tuple[float, list[Instr], tuple, tuple[Instr, ...]]] = []
        for prog, plan, parent_sig in frontier:
            for edit in enumerate_valid_edits(prog, config, rng, edit_cfg):
                try:
                    child = apply_edit(prog, edit)
                except EditError:
                    continue
                sig = tuple(child)
                if sig in seen:
                    continue
                seen.add(sig)
                if total() >= max_executions:
                    return SearchResult(False, total(), None, depth)
                s = scorer.score_edit(parent_sig, edit, child, init, goal)
                scored.append((s, child, plan + (edit,), sig))
        if not scored:
            break
        scored.sort(key=lambda x: x[0])
        beam = scored[:beam_width]

        for _, child, plan, _sig in beam[:verify_k]:
            if total() + 1 > max_executions:
                return SearchResult(False, total(), None, depth)
            verifications += 1
            try:
                trace = run_traced(child, init, max_steps=max_steps)
            except VMError:
                continue
            if satisfies(goal, trace):
                return SearchResult(True, total(), list(plan), depth)

        frontier = [(child, plan, sig) for _, child, plan, sig in beam]

    return SearchResult(False, total(), None, max_depth)
