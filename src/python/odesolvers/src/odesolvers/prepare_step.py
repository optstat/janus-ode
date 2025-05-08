# prepare_step.py
import torch
from typing import Optional
from .linsolve import decom_rc_gmres
from .linsolve import build_lu_solvers
from .stepcontext import StepContext          # make sure this import exists
from .radau_tables import *
TABLES = {1: coertv1, 3: coertv3, 5: coertv5, 7: coertv7}


def prepare_step(ctx: StepContext, *, stages: int, mass: torch.Tensor,
                 gmres: bool = True) -> None:
    """
    • With stages ∈ {1,3,5,7}  → load Radau-IIA tables, populate ctx.T, …
    • With stages = 0          → BDF-type single-γ step; ctx.ValP must already
                                 be set (done inside `bdf2_step`).
    In both cases this routine (re)builds `ctx.solve_fns` when needed.
    """

    # ---------------------------------------------------------------------
    # 1.  Load / refresh Runge-Kutta tables (skip when stages == 0)
    # ---------------------------------------------------------------------
    if stages == 0:
        # BDF-2 or other one-stage γ methods
        ctx.T = ctx.TI = ctx.C = ctx.Dd = None     # not used
    else:
        need_new_table = (ctx.T is None) or (ctx.T.shape[0] != stages)
        if need_new_table:
            ctx.T, ctx.TI, ctx.C, ctx.ValP, ctx.Dd = TABLES[stages]()
            ctx.need_fact = True                   # trigger refactorisation

    # ---------------------------------------------------------------------
    # 2.  Recompute Jacobian if flagged
    # ---------------------------------------------------------------------
    if ctx.need_jac:
        ctx.J = (ctx.jac_f(ctx.t, ctx.y) if ctx.jac_f is not None
                 else torch.autograd.functional.jacobian(
                        lambda y_: ctx.ode_f(ctx.t, y_), ctx.y))
        ctx.stats["fcall"] += 1
        ctx.need_jac  = False
        ctx.need_fact = True                      # Jacobian changed → refactor

    # ---------------------------------------------------------------------
    # 3.  Re-factorise M − γJ (or stage matrices) if needed
    # ---------------------------------------------------------------------
    if ctx.need_fact:
        if ctx.ValP is None:
            raise ValueError("ctx.ValP must be set before factorisation")

        if gmres:
            ctx.solve_fns = decom_rc_gmres(ctx.h, ctx.ValP,
                                           mass, ctx.J, real=True)
        else:
            ctx.solve_fns = build_lu_solvers(ctx.h, ctx.ValP,
                                             mass, ctx.J, real_layout=True)

        ctx.stats["fact"] += 1
        ctx.need_fact = False