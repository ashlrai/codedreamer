"""Random program/input generators and the out-of-distribution split machinery.

The generators synthesize ASTs over a fixed variable set while tracking which
variables are *definitely defined*, so compiled programs never read an undefined
register. Loop counts are bounded and loop variables are reserved, so every
generated program terminates within the VM step budget.

The point of generating at the AST level is precise control over the five OOD
axes the project is built to test: trace length, numeric magnitude, nesting
depth, compositional novelty (held-out operator pairings), and program size.
``realized_metrics`` reports the actual axis values of an executed example so the
dataset builder can bucket train/test and *prove* the splits are disjoint.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace

from .dsl import (Assign, BinOp, Const, Expr, For, If, ListLoad, ListStore,
                  Program, Stmt, Var, compile_program, make_config)
from .vm import ARITH_OPS, CMP_OPS, Config, MachineState, Op, Trace, run_traced


@dataclass
class GenSpec:
    """Knobs controlling the distribution of generated programs and inputs."""

    num_vars: int = 4
    num_inputs: int = 2
    num_temps: int = 14
    max_depth: int = 2
    max_expr_depth: int = 3    # expression nesting, decoupled from statement depth
    num_stmts: int = 5
    max_const: int = 10        # magnitude bound for literal constants
    max_input_val: int = 10    # magnitude bound for input register values
    max_loop_count: int = 4
    arith_ops: tuple[Op, ...] = ARITH_OPS
    cmp_ops: tuple[Op, ...] = CMP_OPS
    use_heap: bool = True
    num_lists: int = 1
    list_len: int = 4
    max_steps: int = 256
    # weights for statement kinds at block level
    w_assign: float = 1.0
    w_if: float = 0.6
    w_for: float = 0.5
    w_liststore: float = 0.4
    w_listload: float = 0.4
    # compositional-novelty holdout: (outer_context, op) pairs forbidden in gen.
    # context is "loop" (inside a For body) or "if" (inside an If body).
    forbidden_pairs: frozenset[tuple[str, Op]] = field(default_factory=frozenset)

    def config(self) -> Config:
        return make_config(self.num_vars, num_temps=self.num_temps,
                           num_lists=self.num_lists, list_len=self.list_len,
                           max_steps=self.max_steps)


# ---------------------------------------------------------------------------
# Expression / statement generation
# ---------------------------------------------------------------------------


def _gen_expr(rng: random.Random, spec: GenSpec, defined: list[str],
             depth: int, context: str, *, boolean: bool) -> Expr:
    """Generate an expression. ``boolean`` requests a comparison at the top
    (for conditions); ``context`` ('top'|'if'|'loop') gates forbidden ops."""
    if boolean:
        op = rng.choice([o for o in spec.cmp_ops
                         if (context, o) not in spec.forbidden_pairs] or list(spec.cmp_ops))
        left = _gen_expr(rng, spec, defined, depth - 1, context, boolean=False)
        right = _gen_expr(rng, spec, defined, depth - 1, context, boolean=False)
        return BinOp(op, left, right)

    if depth <= 0 or rng.random() < 0.45:
        if defined and rng.random() < 0.6:
            return Var(rng.choice(defined))
        return Const(rng.randint(0, spec.max_const))

    allowed = [o for o in spec.arith_ops if (context, o) not in spec.forbidden_pairs]
    if not allowed:
        allowed = list(spec.arith_ops)
    op = rng.choice(allowed)
    left = _gen_expr(rng, spec, defined, depth - 1, context, boolean=False)
    right = _gen_expr(rng, spec, defined, depth - 1, context, boolean=False)
    return BinOp(op, left, right)


def _gen_block(rng: random.Random, spec: GenSpec, defined: set[str],
              depth: int, n_stmts: int, context: str,
              reserved: set[str]) -> list[Stmt]:
    """Generate a straight-line-ish block. ``defined`` is mutated to reflect
    unconditional definitions; ``reserved`` holds loop vars that must not be
    reassigned. Conditional definitions (inside if/for) do not escape the block."""
    body: list[Stmt] = []
    assignable = [n for n in spec.config().reg_names
                  if n.startswith("v") and n not in reserved]
    for _ in range(n_stmts):
        # Build the menu of statement kinds available *here*. Kinds that need a
        # writable register (assign/for/listload) are only offered when one is
        # free; otherwise we fall back to if/liststore, or stop the block.
        kinds: list[str] = []
        weights: list[float] = []
        if assignable:
            kinds.append("assign"); weights.append(spec.w_assign)
            if depth > 0:
                kinds.append("for"); weights.append(spec.w_for)
            if spec.use_heap:
                kinds.append("listload"); weights.append(spec.w_listload)
        if depth > 0:
            kinds.append("if"); weights.append(spec.w_if)
        if spec.use_heap:
            kinds.append("liststore"); weights.append(spec.w_liststore)
        if not kinds:
            break
        kind = rng.choices(kinds, weights=weights, k=1)[0]
        defs = sorted(defined)

        if kind == "assign":
            target = rng.choice(assignable)
            expr = _gen_expr(rng, spec, defs, spec.max_expr_depth, context, boolean=False)
            body.append(Assign(target, expr))
            defined.add(target)

        elif kind == "if":
            cond = _gen_expr(rng, spec, defs, spec.max_expr_depth, "if", boolean=bool(defs))
            inner = set(defined)
            then = _gen_block(rng, spec, inner, depth - 1,
                              max(1, n_stmts // 2), "if", reserved)
            orelse = (_gen_block(rng, spec, set(defined), depth - 1,
                                 max(1, n_stmts // 2), "if", reserved)
                      if rng.random() < 0.5 else [])
            body.append(If(cond, then, orelse))
            # conditional defs do not escape

        elif kind == "for":
            loop_var = rng.choice(assignable)
            count = Const(rng.randint(1, spec.max_loop_count))
            inner_reserved = reserved | {loop_var}
            inner_defined = set(defined) | {loop_var}
            inner_body = _gen_block(rng, spec, inner_defined, depth - 1,
                                    max(1, n_stmts // 2), "loop", inner_reserved)
            body.append(For(loop_var, count, inner_body))
            defined.add(loop_var)  # loop var is unconditionally assigned 0

        elif kind == "liststore":
            idx = Const(rng.randint(0, spec.list_len - 1))
            value = _gen_expr(rng, spec, defs, spec.max_expr_depth, context, boolean=False)
            body.append(ListStore(rng.randrange(spec.num_lists), idx, value))

        elif kind == "listload":
            target = rng.choice(assignable)
            idx = Const(rng.randint(0, spec.list_len - 1))
            body.append(ListLoad(target, rng.randrange(spec.num_lists), idx))
            defined.add(target)

    return body


def generate_program(rng: random.Random, spec: GenSpec) -> Program:
    """Generate a compilable, terminating AST program over ``spec``'s config."""
    config = spec.config()
    input_vars = {f"v{i}" for i in range(spec.num_inputs)}
    defined = set(input_vars)
    body = _gen_block(rng, spec, defined, spec.max_depth, spec.num_stmts,
                      "top", reserved=set())
    return Program(body=body, config=config)


def sample_inputs(rng: random.Random, spec: GenSpec) -> tuple[dict[str, int], list[list[int]]]:
    """Sample initial register values for the input vars and heap contents."""
    regs = {f"v{i}": rng.randint(-spec.max_input_val, spec.max_input_val)
            for i in range(spec.num_inputs)}
    heap = [[rng.randint(-spec.max_input_val, spec.max_input_val)
             for _ in range(spec.list_len)] for _ in range(spec.num_lists)]
    return regs, heap


# ---------------------------------------------------------------------------
# Example construction + realized metrics
# ---------------------------------------------------------------------------


@dataclass
class Example:
    program: Program
    bytecode: list
    init_state: MachineState
    trace: Trace


def make_example(rng: random.Random, spec: GenSpec) -> Example:
    """Generate a program + inputs and execute it to a full trace."""
    program = generate_program(rng, spec)
    bytecode = compile_program(program)
    regs, heap = sample_inputs(rng, spec)
    init = program.config.initial_state(regs=regs, heap=heap)
    trace = run_traced(bytecode, init, max_steps=spec.max_steps)
    return Example(program=program, bytecode=bytecode, init_state=init, trace=trace)


def realized_metrics(ex: Example) -> dict[str, int]:
    """The actual OOD-axis values of an executed example, for bucketing/asserts."""
    max_mag = 0
    for st in ex.trace.states:
        for v in st.regs.values():
            if v is not None:
                max_mag = max(max_mag, abs(v))
        for cells in st.heap:
            for v in cells:
                max_mag = max(max_mag, abs(v))
    return {
        "trace_len": len(ex.trace),
        "max_magnitude": max_mag,
        "nesting_depth": _ast_depth(ex.program.body),
        "program_size": _ast_size(ex.program.body),
        "terminated": int(ex.trace.terminated),
    }


def _ast_depth(body: list[Stmt]) -> int:
    best = 0
    for s in body:
        if isinstance(s, If):
            best = max(best, 1 + _ast_depth(s.then), 1 + _ast_depth(s.orelse))
        elif isinstance(s, For):
            best = max(best, 1 + _ast_depth(s.body))
    return best


def _ast_size(body: list[Stmt]) -> int:
    total = 0
    for s in body:
        total += 1
        if isinstance(s, If):
            total += _ast_size(s.then) + _ast_size(s.orelse)
        elif isinstance(s, For):
            total += _ast_size(s.body)
    return total


# ---------------------------------------------------------------------------
# OOD axis specifications
# ---------------------------------------------------------------------------

# Each axis: (train predicate threshold, test predicate threshold). The split
# builder generates from the matching spec and *filters by realized metric* so
# the train/test buckets are provably disjoint on that axis.

@dataclass(frozen=True)
class OODAxis:
    name: str
    metric: str          # key in realized_metrics
    train_max: int       # train: metric <= train_max
    test_min: int        # test:  metric >= test_min  (test_min > train_max)
    train_spec: GenSpec
    test_spec: GenSpec


def default_axes(base: GenSpec | None = None) -> list[OODAxis]:
    """The five canonical OOD axes, each as a train/test spec pair + thresholds."""
    base = base or GenSpec()
    held_out_pairs = frozenset({("loop", Op.MOD), ("loop", Op.MUL)})
    return [
        OODAxis("trace_length", "trace_len", train_max=32, test_min=64,
                train_spec=replace(base, num_stmts=4, max_loop_count=3, max_depth=2),
                test_spec=replace(base, num_stmts=8, max_loop_count=8, max_depth=3)),
        OODAxis("magnitude", "max_magnitude", train_max=30, test_min=300,
                train_spec=replace(base, max_const=5, max_input_val=5),
                test_spec=replace(base, max_const=400, max_input_val=400)),
        OODAxis("nesting_depth", "nesting_depth", train_max=2, test_min=4,
                train_spec=replace(base, max_depth=2, num_vars=8),
                test_spec=replace(base, max_depth=5, num_stmts=8, num_vars=8)),
        OODAxis("program_size", "program_size", train_max=8, test_min=16,
                train_spec=replace(base, num_stmts=4, max_depth=2, num_vars=8),
                test_spec=replace(base, num_stmts=16, max_depth=3, num_vars=8)),
        OODAxis("compositional", "program_size", train_max=10_000, test_min=0,
                # disjointness here is structural (held-out op/context pairs),
                # not metric-thresholded; see build_split's structural check.
                train_spec=replace(base, forbidden_pairs=held_out_pairs),
                test_spec=replace(base, forbidden_pairs=frozenset())),
    ]
