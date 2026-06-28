"""A *learned* world-model scorer for the M3 planning harness (the M3 payoff test).

Unlike :func:`~execwm.plan.planner.cheap_scorer` (a single linear pass that ignores
control flow) and :class:`~execwm.plan.planner.OracleScorer` (which actually runs
the VM), :class:`WorldModelScorer` scores a candidate program by *simulating it in
the latent space of a trained* :class:`~execwm.model.world_model.GroundedLatentWM`,
running **zero** VM executions.

The simulation is self-contained: encode the initial state, then repeatedly decode
the program counter from the current latent, fetch ``program[pc]`` from the real
bytecode, encode that instruction as an action, push the latent one step forward
with ``predict_next``, and decode again -- until the decoded state is halted/errored,
``pc`` falls outside the program, or a step budget is hit. The decoded final state
is scored with :func:`~execwm.plan.goal_tasks.goal_distance`. This reuses the exact
encode -> dynamics -> decode primitives the trainer's ``rollout_horizon`` exercises;
the only difference is that the *action at each step is chosen by the model's own
decoded pc* (a closed-loop rollout) rather than a teacher-forced ground-truth action.

Because the loop never calls ``run_traced``, the ``executions`` attribute stays ``0``
and :func:`~execwm.plan.planner.beam_plan` folds in no hidden VM cost. The scorer is
only as accurate as the model's single-step decode compounded over the rollout, so
its scores are an *estimate* (see the M3 honesty caveat: latent error compounds).

Lossy decode note: a decoded :class:`~execwm.substrate.vm.MachineState` reconstructs
register values / types / pc / halted / error from the grounding heads' argmax, which
is exactly the codec's notion of state, but the model can decode an *internally
inconsistent* state (e.g. a register typed INT with a wrong magnitude). We do not
repair these; the goal checker reads them as-is, which is the point of the test.
"""

from __future__ import annotations

import numpy as np
import torch

from ..data.action_codec import ActionCodec
from ..data.state_codec import EncodeError, EncodedState, StateCodec
from ..model.world_model import GroundedLatentWM
from ..substrate.vm import Instr, MachineState
from .goal_tasks import Goal, goal_distance


def _state_to_tensors(scodec: StateCodec, state: MachineState,
                      device: torch.device) -> dict[str, torch.Tensor]:
    """Encode a :class:`MachineState` into the batched (B=1) tensor dict the
    model's ``encode`` consumes. Scalars become shape ``(1,)``, vectors ``(1, ...)``."""
    enc = scodec.encode(state).as_dict()
    return {k: torch.as_tensor(v, dtype=torch.long, device=device).unsqueeze(0)
            for k, v in enc.items()}


def _action_to_tensors(acodec: ActionCodec, instr: Instr,
                       device: torch.device) -> dict[str, torch.Tensor]:
    """Encode an :class:`Instr` into the batched (B=1) action tensor dict."""
    enc = acodec.encode(instr).as_dict()
    return {k: torch.as_tensor(v, dtype=torch.long, device=device).unsqueeze(0)
            for k, v in enc.items()}


def decode_latent_to_state(model: GroundedLatentWM, z: torch.Tensor,
                           scodec: StateCodec) -> MachineState:
    """Decode a per-slot latent ``(1, S, d)`` into a concrete :class:`MachineState`.

    Runs the grounding heads, takes the argmax label per field (the codec's exact
    notion of a state), assembles an :class:`EncodedState`, and decodes it. The
    register payload of an UNDEF-typed register is junk by the codec rule, so its
    value comes back as ``None`` -- matching how the VM and goal checker treat it.
    """
    logits = model.heads(z)

    def lab(key: str) -> np.ndarray:
        return logits[key].argmax(-1)[0].detach().cpu().numpy().astype(np.int64)

    enc = EncodedState(
        reg_type=lab("reg_type"),
        reg_sign=lab("reg_sign"),
        reg_digits=lab("reg_digits"),
        heap_sign=lab("heap_sign"),
        heap_digits=lab("heap_digits"),
        pc=np.asarray(int(logits["pc"].argmax(-1)[0]), dtype=np.int64),
        halted=np.asarray(int(logits["halted"].argmax(-1)[0]), dtype=np.int64),
        error=np.asarray(int(logits["error"].argmax(-1)[0]), dtype=np.int64),
    )
    return scodec.decode(enc)


class WorldModelScorer:
    """Score a program by closed-loop latent simulation -- no VM executions.

    Construct with ``WorldModelScorer(model, scodec, acodec, device, max_steps=...)``.
    Calling ``scorer(program, init_state, goal)`` returns ``goal_distance(goal,
    decoded_final_state)`` as a float. ``executions`` stays ``0`` for the object's
    whole lifetime (it never runs ``run_traced``), so ``beam_plan`` reports the WM
    planner's cost as exactly its real-VM *verification* runs.
    """

    def __init__(self, model: GroundedLatentWM, scodec: StateCodec,
                 acodec: ActionCodec, device: torch.device | str = "cpu",
                 max_steps: int = 64) -> None:
        self.model = model
        self.scodec = scodec
        self.acodec = acodec
        self.device = torch.device(device)
        self.max_steps = max_steps
        self.executions = 0  # invariant: never incremented (no VM calls)

    @torch.no_grad()
    def simulate(self, program: list[Instr],
                 init_state: MachineState) -> MachineState:
        """Roll the latent forward from ``init_state`` under ``program``, fetching
        each instruction by the model's own decoded ``pc``, and return the decoded
        final state. Stops on decoded halt/error, an out-of-program ``pc``, an
        un-encodable instruction, or the step budget."""
        self.model.eval()
        z = self.model.encode(_state_to_tensors(self.scodec, init_state, self.device))
        state = decode_latent_to_state(self.model, z, self.scodec)

        for _ in range(self.max_steps):
            if state.halted or state.error:
                break
            if not (0 <= state.pc < len(program)):
                break  # fell off the end -> termination (as in run_traced)
            try:
                action = _action_to_tensors(self.acodec, program[state.pc], self.device)
            except (EncodeError, KeyError):
                break  # instruction not codec-encodable; treat as terminating
            z = self.model.predict_next(z, action)
            state = decode_latent_to_state(self.model, z, self.scodec)

        return state

    def __call__(self, program: list[Instr], init_state: MachineState,
                 goal: Goal) -> float:
        final_state = self.simulate(program, init_state)
        return goal_distance(goal, final_state)
