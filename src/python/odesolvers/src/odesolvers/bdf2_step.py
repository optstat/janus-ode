# bdf2_step.py
"""
One BDF-2 step re-using the same simplified_newton() used for Radau.
γ = 2/(3h)  → only ONE linear system per step.
"""
import torch
from .simplified_newton import simplified_newton
from .prepare_step import prepare_step          # <-- correct import path


def bdf2_step(ctx, mass, *, gmres: bool = True):
    """
    Parameters
    ----------
    ctx   : StepContext   (expects .y, .y_nm1, .t, .h, .ode_f, ...)
    mass  : torch.Tensor  mass-matrix (n×n)
    gmres : bool          choose GMRES or dense-LU factorisation

    Returns
    -------
    newt  : NewtonResult  (converged, z, rejected flag, need_new_lu …)
    """

    # ------------------------------------------------------------------ #
    # 1.  γ and BDF right-hand side
    # ------------------------------------------------------------------ #
    gamma = 2.0 / (3.0 * ctx.h)                       # BDF-2 coefficient
    rhs   = (4.0 / 3.0) * ctx.y - (1.0 / 3.0) * ctx.y_nm1

    # ------------------------------------------------------------------ #
    # 2.  (Re)factorise M - γJ if needed
    # ------------------------------------------------------------------ #
    #  • ctx.need_fact     : flagged by driver when Jacobian/step changes
    #  • also refactorise if γ itself changed (step-size change)
    if ctx.ValP is None or ctx.ValP[0] != gamma:
        ctx.need_fact = True

    if ctx.need_fact:
        # ValP must contain ONE scalar γ for prepare_step (stages == 0)
        ctx.ValP = torch.tensor([gamma], dtype=ctx.y.dtype)
        prepare_step(ctx, stages=0, mass=mass, gmres=gmres)
        ctx.need_fact = False

    # ------------------------------------------------------------------ #
    # 3.  Simplified Newton solve   (single stage, s = 1)
    # ------------------------------------------------------------------ #
    # --- simplified Newton (single-stage) ---------------------------------
    one = torch.tensor([[1.0]], dtype=ctx.y.dtype)   # 1×1 identity
    C1  = torch.tensor([0.0],  dtype=ctx.y.dtype)    # abscissa 0

    newt = simplified_newton(
        rhs, ctx.t + ctx.h, ctx.h, mass, ctx.J,
        OdeFcn=ctx.ode_f, ode_args=(),
        T=one, TI=one, C=C1,                # <-- pass dummies, not None
        ValP=ctx.ValP,
        solve_fns=ctx.solve_fns,
        RelTol1=ctx.rtol * torch.ones_like(ctx.y),
        Nit=7, stats=None
    )
    return newt