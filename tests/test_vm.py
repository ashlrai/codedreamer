"""Correctness tests for the VM, compiler, and generators.

The centerpiece is a *differential* test: an independent reference interpreter
evaluates the AST directly, and we assert the compiled-bytecode VM reaches the
same final machine state on many random programs. Agreement between two
independently-written implementations is strong evidence the ground-truth oracle
is correct.
"""

import random

import pytest

from execwm.substrate.dsl import (Assign, BinOp, Const, For, If, ListLoad,
                                   ListStore, Program, Var, compile_program,
                                   make_config)
from execwm.substrate.generators import (GenSpec, default_axes, make_example,
                                          realized_metrics)
from execwm.substrate.vm import Op, VType, run_traced


# --------------------------------------------------------------------------
# A reference interpreter for the AST (independent of the bytecode VM).
# --------------------------------------------------------------------------

class _Trap(Exception):
    pass


def _eval_expr(expr, env):
    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, Var):
        return env[expr.name]
    if isinstance(expr, BinOp):
        a, b = _eval_expr(expr.left, env), _eval_expr(expr.right, env)
        op = expr.op
        if op is Op.ADD: return a + b
        if op is Op.SUB: return a - b
        if op is Op.MUL: return a * b
        if op is Op.DIV:
            if b == 0: raise _Trap
            return a // b
        if op is Op.MOD:
            if b == 0: raise _Trap
            return a % b
        if op is Op.LT: return int(a < b)
        if op is Op.LE: return int(a <= b)
        if op is Op.EQ: return int(a == b)
        if op is Op.NE: return int(a != b)
        if op is Op.GT: return int(a > b)
        if op is Op.GE: return int(a >= b)
    raise AssertionError(f"bad expr {expr}")


def _exec_block(body, env, heap):
    for stmt in body:
        if isinstance(stmt, Assign):
            env[stmt.target] = _eval_expr(stmt.expr, env)
        elif isinstance(stmt, If):
            if _eval_expr(stmt.cond, env) != 0:
                _exec_block(stmt.then, env, heap)
            else:
                _exec_block(stmt.orelse, env, heap)
        elif isinstance(stmt, For):
            cnt = _eval_expr(stmt.count, env)
            env[stmt.var] = 0
            while env[stmt.var] < cnt:
                _exec_block(stmt.body, env, heap)
                env[stmt.var] += 1
        elif isinstance(stmt, ListStore):
            heap[stmt.list_id][_eval_expr(stmt.index, env)] = _eval_expr(stmt.value, env)
        elif isinstance(stmt, ListLoad):
            env[stmt.target] = heap[stmt.list_id][_eval_expr(stmt.index, env)]
        else:
            raise AssertionError(f"bad stmt {stmt}")


def _ref_run(program: Program, init):
    """Run the reference interpreter; returns (env, heap, trapped)."""
    env = {name: init.regs[name] for name in program.config.reg_names}
    heap = [list(cells) for cells in init.heap]
    try:
        _exec_block(program.body, env, heap)
        return env, heap, False
    except _Trap:
        return env, heap, True


# --------------------------------------------------------------------------
# Hand-written sanity programs
# --------------------------------------------------------------------------

def test_arithmetic_and_branch():
    cfg = make_config(num_vars=3, num_temps=4)
    # v0 = input; if v0 < 5: v1 = v0 * 2 else v1 = v0 + 100
    prog = Program(body=[
        If(BinOp(Op.LT, Var("v0"), Const(5)),
           [Assign("v1", BinOp(Op.MUL, Var("v0"), Const(2)))],
           [Assign("v1", BinOp(Op.ADD, Var("v0"), Const(100)))]),
    ], config=cfg)
    code = compile_program(prog)

    for v0, expected in [(3, 6), (5, 105), (10, 110)]:
        init = cfg.initial_state(regs={"v0": v0})
        tr = run_traced(code, init, max_steps=64)
        assert tr.terminated and not tr.final_state.error
        assert tr.final_state.regs["v1"] == expected
        assert tr.final_state.types["v1"] is VType.INT


def test_for_loop_sum():
    cfg = make_config(num_vars=3, num_temps=4)
    # v1 = 0; for v2 in range(v0): v1 = v1 + v2   ->  sum 0..v0-1
    prog = Program(body=[
        Assign("v1", Const(0)),
        For("v2", Var("v0"), [Assign("v1", BinOp(Op.ADD, Var("v1"), Var("v2")))]),
    ], config=cfg)
    code = compile_program(prog)
    init = cfg.initial_state(regs={"v0": 5})
    tr = run_traced(code, init, max_steps=256)
    assert tr.terminated and tr.final_state.regs["v1"] == 0 + 1 + 2 + 3 + 4
    assert tr.final_state.regs["v2"] == 5  # loop var ends at count


def test_div_zero_traps():
    cfg = make_config(num_vars=2, num_temps=2)
    prog = Program(body=[Assign("v1", BinOp(Op.DIV, Var("v0"), Const(0)))], config=cfg)
    code = compile_program(prog)
    tr = run_traced(code, cfg.initial_state(regs={"v0": 9}), max_steps=16)
    assert tr.final_state.error and tr.final_state.halted


def test_heap_store_load():
    cfg = make_config(num_vars=2, num_temps=4, num_lists=1, list_len=4)
    prog = Program(body=[
        ListStore(0, Const(2), BinOp(Op.ADD, Var("v0"), Const(1))),
        ListLoad("v1", 0, Const(2)),
    ], config=cfg)
    code = compile_program(prog)
    tr = run_traced(code, cfg.initial_state(regs={"v0": 7}), max_steps=16)
    assert tr.final_state.regs["v1"] == 8
    assert tr.final_state.heap[0][2] == 8


# --------------------------------------------------------------------------
# Differential test: bytecode VM vs reference interpreter
# --------------------------------------------------------------------------

def test_differential_vm_vs_reference():
    rng = random.Random(0)
    spec = GenSpec(num_vars=4, num_inputs=2, max_depth=2, num_stmts=5,
                   max_const=6, max_input_val=6, max_loop_count=3, max_steps=512)
    checked = 0
    for _ in range(800):
        ex = make_example(rng, spec)
        if not ex.trace.terminated:
            continue  # ran into budget; skip (rare with these specs)
        env, heap, trapped = _ref_run(ex.program, ex.init_state)
        final = ex.trace.final_state
        if final.error:
            assert trapped, "VM trapped but reference did not"
            continue
        assert not trapped, "reference trapped but VM did not"
        # user variables must match exactly (temps are VM-internal)
        for name in ex.program.config.reg_names:
            if name.startswith("v"):
                assert final.regs[name] == env[name], (
                    f"reg {name}: VM={final.regs[name]} ref={env[name]}")
        assert final.heap == heap
        checked += 1
    assert checked > 300, f"too few terminating programs checked ({checked})"


def test_generated_programs_never_read_undefined():
    """Generation must guarantee compilable programs that never trap on an
    undefined register read (a VMError would be raised if they did)."""
    rng = random.Random(7)
    spec = GenSpec(max_depth=3, num_stmts=6)
    for _ in range(500):
        ex = make_example(rng, spec)  # raises VMError if an undefined read occurs
        assert ex.trace is not None


def test_ood_axes_are_separable():
    """Each axis' train spec should mostly land below the threshold and the test
    spec mostly above, so a disjoint split is actually achievable."""
    rng = random.Random(3)
    for axis in default_axes():
        if axis.name == "compositional":
            continue  # structural holdout, not metric-thresholded
        train_vals, test_vals = [], []
        for _ in range(120):
            tr = realized_metrics(make_example(rng, axis.train_spec))[axis.metric]
            te = realized_metrics(make_example(rng, axis.test_spec))[axis.metric]
            train_vals.append(tr)
            test_vals.append(te)
        n_train_ok = sum(v <= axis.train_max for v in train_vals)
        n_test_ok = sum(v >= axis.test_min for v in test_vals)
        # we don't need every sample to qualify (the builder filters), but a
        # healthy fraction must, or the axis thresholds are mis-set.
        assert n_train_ok > 40, f"{axis.name}: too few train below {axis.train_max}"
        assert n_test_ok > 20, f"{axis.name}: too few test above {axis.test_min}"
