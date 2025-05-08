import math, torch, inspect
from dataclasses import dataclass
from typing import Callable, Sequence, List
from .solvrad import solve_radau           # stage linear solves    


@dataclass
class NewtonResult:
    converged:   bool          # True  if Newton satisfied conv. test
    rejected:    bool          # True  if step must be retried with new h
    need_new_lu: bool          # True  if you must rebuild LU / GMRES
    h_new:       float         # possibly reduced step size
    w:           torch.Tensor  # stage state updates  (n, s)
    z:           torch.Tensor  # stage corrections    (n, s)
    stats:       "NewtonStats" # updated counters


@dataclass
class NewtonStats:
    StepNbr:    int = 0
    StepRejNbr: int = 0
    NewtRejNbr: int = 0
    SolveNbr:   int = 0
    FcnNbr:     int = 0



# --------------------------------------------------------------------------- #
def simplified_newton(
    y:         torch.Tensor,
    t:         float,
    h:         float,
    Mass:      torch.Tensor | None,
    Jac:       torch.Tensor,
    OdeFcn:    Callable[[float, torch.Tensor, ...], torch.Tensor],
    ode_args:  Sequence = (),
    *,
    T:         torch.Tensor,      # (s,s)
    TI:        torch.Tensor,      # (s,s)
    C:         torch.Tensor,      # (s,)
    ValP:      torch.Tensor,      # (s,)
    solve_fns: List[Callable],    # stage solvers (LU or GMRES)
    RelTol1:   torch.Tensor,      # (n,) component acc.
    Nit:       int = 7,
    FacConv0:  float = 0.9,
    ExpmNs:    float = 2.0,
    stats:     NewtonStats | None = None,
) -> NewtonResult:
    """
    One simplified-Newton iteration for a single Radau step.

    Does *not* compute local error; caller calls estrad() afterwards.
    """

    if stats is None:
        stats = NewtonStats()

    n, s = y.numel(), ValP.numel()
    sqrt_stg_ny = math.sqrt(s * n)

    # Hairer’s FNewt constant
    FNewt = max(10.0 * torch.finfo(y.dtype).eps / RelTol1.min().item(),
                min(0.03, RelTol1.min().pow(1 / ExpmNs - 1).item()))
    if s == 1:
        FNewt = max(10.0 * torch.finfo(y.dtype).eps / RelTol1.min().item(),
                    0.03)

    FacConv = max(FacConv0, torch.finfo(y.dtype).eps) ** 0.8
    Theta   = 1.0

    # work arrays
    z = torch.zeros(n, s, dtype=y.dtype)
    w = torch.zeros_like(z)
    f = torch.empty_like(z)

    # evaluate f(t,y) once (used for Estrad later)
    sig = len(inspect.signature(OdeFcn).parameters)
    f0  = OdeFcn(t, y, *ode_args) if sig > 2 else OdeFcn(t, y)
    stats.FcnNbr += 1

    need_new_lu = False
    old_nrm     = 1.0
    thq_old     = 1.0

    newton_iter = 0
    while True:
        newton_iter += 1

        # --- RHS evaluations at each stage ---------------------------------
        for q in range(s):
            f[:, q] = OdeFcn(t + C[q].item() * h,
                             y + z[:, q],
                             *ode_args) if sig > 2 else \
                       OdeFcn(t + C[q].item() * h,
                              y + z[:, q])
        stats.FcnNbr += s
        # ------------------------------------------------------------------
        # special case: one-stage scheme (BDF-2, implicit Euler, etc.)
        # ------------------------------------------------------------------
        if TI.shape == (1, 1):                 # NbrStg == 1
            z = f.clone()               # z = f(t + C[0] * h, y + z[:, 0])
        else:
            # --- build RHS for linear solves: Z = TI @ f -----------------------
            z = (f.transpose(0, 1) @ TI.T).transpose(0, 1)
        # --- solve stage linear systems ------------------------------------
        z = solve_radau(z, w, ValP, h, solve_fns,
                        Mass=Mass, real_layout=True)
        stats.SolveNbr += 1

        # --- Newton convergence measure ------------------------------------
        new_nrm = (z / RelTol1[:, None]).norm(dim=0).sum() / sqrt_stg_ny

        if newton_iter > 1:
            thq = new_nrm / old_nrm
            Theta = thq if newton_iter == 2 else math.sqrt(thq * thq_old)
            thq_old = thq

            # divergence?
            if Theta >= 0.99:
                stats.StepRejNbr += 1
                stats.NewtRejNbr += 1
                return NewtonResult(False, True, True, h * 0.5,
                                    w, z, stats)

            # slow convergence?
            FacConv = Theta / (1.0 - Theta)
            dyth = (FacConv * new_nrm *
                    Theta ** (Nit - 1 - newton_iter) / FNewt)
            if dyth >= 1.0:
                qnew  = max(1e-4, min(20.0, dyth))
                hhfac = 0.8 * qnew ** (-1.0 / (4 + Nit - 1 - newton_iter))
                stats.StepRejNbr += 1
                stats.NewtRejNbr += 1
                return NewtonResult(False, True, True, h * hhfac,
                                    w, z, stats)

        # update for next iteration
        old_nrm = max(new_nrm, torch.finfo(y.dtype).eps)

        w += z                                   # update stage values
        # ---- build z for next iteration ---------------------------------
        if T.shape == (1, 1):                    # s = 1  → BDF-type
            z = w.clone()                        # (n,1)  scalar case
        else:
            z = (w.transpose(0, 1) @ T.T).transpose(0, 1)

        # convergence check
        if FacConv * new_nrm <= FNewt:
            return NewtonResult(True, False, False, h,
                                w, z, stats)

        if newton_iter >= Nit:                   # exceeded max iterations
            stats.StepRejNbr += 1
            stats.NewtRejNbr += 1
            return NewtonResult(False, True, True, h * 0.5,
                                w, z, stats)