"""A tiny imperative DSL and its compiler to the bytecode in :mod:`vm`.

Programs are built as small ASTs (assignments, if/else, while, for, list
read/write). The compiler lowers them to flat bytecode with resolved jump
targets. Working at the AST level lets the generators control *structural* knobs
directly — nesting depth, control-flow shape — which are exactly the
out-of-distribution axes the project cares about.

Registers are split into user variables (``v0..``) and compiler temporaries
(``t0..``); both live in a fixed :class:`~execwm.substrate.vm.Config` so every
program over a given config shares one constant-shape machine state.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Union

from .vm import Config, Instr, Op

# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Const:
    value: int


@dataclass(frozen=True)
class Var:
    name: str


@dataclass(frozen=True)
class BinOp:
    op: Op  # one of ARITH_OPS or CMP_OPS
    left: "Expr"
    right: "Expr"


Expr = Union[Const, Var, BinOp]


@dataclass(frozen=True)
class Assign:
    target: str
    expr: Expr


@dataclass(frozen=True)
class If:
    cond: Expr
    then: list["Stmt"]
    orelse: list["Stmt"]


@dataclass(frozen=True)
class While:
    cond: Expr
    body: list["Stmt"]


@dataclass(frozen=True)
class For:
    """``for var in range(count): body`` — count is evaluated once up front."""

    var: str
    count: Expr
    body: list["Stmt"]


@dataclass(frozen=True)
class ListStore:
    list_id: int
    index: Expr
    value: Expr


@dataclass(frozen=True)
class ListLoad:
    target: str
    list_id: int
    index: Expr


Stmt = Union[Assign, If, While, For, ListStore, ListLoad]


@dataclass(frozen=True)
class Program:
    body: list[Stmt]
    config: Config


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


class _Emitter:
    """Accumulates instructions and supports jump-target backpatching."""

    def __init__(self) -> None:
        self.code: list[Instr] = []

    def emit(self, instr: Instr) -> int:
        self.code.append(instr)
        return len(self.code) - 1

    def patch_target(self, index: int, target: int) -> None:
        self.code[index] = dataclasses.replace(self.code[index], target=target)

    def here(self) -> int:
        return len(self.code)


class _TempPool:
    """Hands out temporary registers in a stack discipline (acquire/release)."""

    def __init__(self, names: list[str]) -> None:
        self._free = list(names)
        self._all = list(names)
        self.high_water = 0

    def acquire(self) -> str:
        if not self._free:
            raise CompileError("ran out of temp registers; raise num_temps")
        name = self._free.pop()
        self.high_water = max(self.high_water, len(self._all) - len(self._free))
        return name

    def release(self, name: str) -> None:
        self._free.append(name)


class CompileError(Exception):
    pass


def make_config(num_vars: int, num_temps: int = 8, num_lists: int = 1,
               list_len: int = 4, max_steps: int = 256) -> Config:
    """Build a VM config with ``num_vars`` user vars (``v*``) and temps (``t*``)."""
    reg_names = tuple([f"v{i}" for i in range(num_vars)]
                      + [f"t{i}" for i in range(num_temps)])
    return Config(reg_names=reg_names, num_lists=num_lists,
                  list_len=list_len, max_steps=max_steps)


def _operand(em: _Emitter, temps: _TempPool, expr: Expr) -> tuple[object, str | None]:
    """Reduce ``expr`` to a VM operand (immediate int or register name),
    emitting code if needed. Returns ``(operand, temp_to_release_or_None)``."""
    if isinstance(expr, Const):
        return expr.value, None
    if isinstance(expr, Var):
        return expr.name, None
    # compound expression -> materialize into a fresh temp
    dst = temps.acquire()
    _compile_expr_into(em, temps, expr, dst)
    return dst, dst


def _compile_expr_into(em: _Emitter, temps: _TempPool, expr: Expr, dst: str) -> None:
    """Emit code computing ``expr`` into register ``dst``."""
    if isinstance(expr, Const):
        em.emit(Instr(Op.CONST, dst=dst, a=expr.value))
        return
    if isinstance(expr, Var):
        em.emit(Instr(Op.MOV, dst=dst, a=expr.name))
        return
    if isinstance(expr, BinOp):
        a_op, a_tmp = _operand(em, temps, expr.left)
        b_op, b_tmp = _operand(em, temps, expr.right)
        em.emit(Instr(expr.op, dst=dst, a=a_op, b=b_op))
        if b_tmp:
            temps.release(b_tmp)
        if a_tmp:
            temps.release(a_tmp)
        return
    raise CompileError(f"unknown expr {expr!r}")


def _compile_stmt(em: _Emitter, temps: _TempPool, stmt: Stmt) -> None:
    if isinstance(stmt, Assign):
        _compile_expr_into(em, temps, stmt.expr, stmt.target)

    elif isinstance(stmt, If):
        tcond = temps.acquire()
        _compile_expr_into(em, temps, stmt.cond, tcond)
        jz = em.emit(Instr(Op.JZ, a=tcond, target=-1))
        temps.release(tcond)
        for s in stmt.then:
            _compile_stmt(em, temps, s)
        if stmt.orelse:
            jmp_end = em.emit(Instr(Op.JMP, target=-1))
            em.patch_target(jz, em.here())  # else starts here
            for s in stmt.orelse:
                _compile_stmt(em, temps, s)
            em.patch_target(jmp_end, em.here())
        else:
            em.patch_target(jz, em.here())

    elif isinstance(stmt, While):
        loop_start = em.here()
        tcond = temps.acquire()
        _compile_expr_into(em, temps, stmt.cond, tcond)
        jz = em.emit(Instr(Op.JZ, a=tcond, target=-1))
        temps.release(tcond)
        for s in stmt.body:
            _compile_stmt(em, temps, s)
        em.emit(Instr(Op.JMP, target=loop_start))
        em.patch_target(jz, em.here())

    elif isinstance(stmt, For):
        # for var in range(count): body   ==>   var=0; while var<count: body; var+=1
        tcount = temps.acquire()
        _compile_expr_into(em, temps, stmt.count, tcount)
        em.emit(Instr(Op.CONST, dst=stmt.var, a=0))
        loop_start = em.here()
        tcond = temps.acquire()
        em.emit(Instr(Op.LT, dst=tcond, a=stmt.var, b=tcount))
        jz = em.emit(Instr(Op.JZ, a=tcond, target=-1))
        temps.release(tcond)
        for s in stmt.body:
            _compile_stmt(em, temps, s)
        em.emit(Instr(Op.ADD, dst=stmt.var, a=stmt.var, b=1))
        em.emit(Instr(Op.JMP, target=loop_start))
        em.patch_target(jz, em.here())
        temps.release(tcount)

    elif isinstance(stmt, ListStore):
        idx_op, idx_tmp = _operand(em, temps, stmt.index)
        val_op, val_tmp = _operand(em, temps, stmt.value)
        em.emit(Instr(Op.STORE, list_id=stmt.list_id, a=idx_op, b=val_op))
        if val_tmp:
            temps.release(val_tmp)
        if idx_tmp:
            temps.release(idx_tmp)

    elif isinstance(stmt, ListLoad):
        idx_op, idx_tmp = _operand(em, temps, stmt.index)
        em.emit(Instr(Op.LOAD, dst=stmt.target, list_id=stmt.list_id, a=idx_op))
        if idx_tmp:
            temps.release(idx_tmp)

    else:
        raise CompileError(f"unknown stmt {stmt!r}")


def compile_program(program: Program) -> list[Instr]:
    """Lower an AST :class:`Program` to flat bytecode ending in HALT."""
    em = _Emitter()
    temp_names = [n for n in program.config.reg_names if n.startswith("t")]
    temps = _TempPool(temp_names)
    for stmt in program.body:
        _compile_stmt(em, temps, stmt)
    em.emit(Instr(Op.HALT))
    return em.code
