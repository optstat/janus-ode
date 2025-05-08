# estrad.py
import math, inspect, torch
from types import SimpleNamespace
from typing import Callable, Any, Sequence


def estrad(
    z:       torch.Tensor,          # (n, s)  stage values after Newton step
    Dd:      torch.Tensor,          # (s,)    Radau error coefficients
    h:       float,
    solve_1: Callable[[torch.Tensor], torch.Tensor],   # first-stage solver
    Mass:    torch.Tensor | None,   # (n, n) or None
    Scal:    torch.Tensor,          # (n,)   error scale vector
    f0:      torch.Tensor,          # (n,)   f(t, y)  from previous step
    *,
    First:   bool,
    Reject:  bool,
    Stat:    SimpleNamespace,
    OdeFcn:  Callable[[float, torch.Tensor, Any], torch.Tensor],
    t:       float,
    y:       torch.Tensor,
    ode_args: Sequence[Any] = (),
) -> tuple[float, SimpleNamespace]:
    """
    Radau local-error estimator (Hairer II Sec. IV.8).
    Returns (err, updated Stat).

    Parameters
    ----------
    solve_1 : callable that solves  (γ₁/h * M − J) x = rhs
              e.g.  solve_fns[0] from decom_rc_gmres or build_lu_solvers.
    Stat    : SimpleNamespace with at least `.FcnNbr` (function call counter)
    """

    sqrt_n = math.sqrt(y.numel())

    # 1 · temp = z · (Dd/h)
    temp = z @ (Dd / h)
    if Mass is not None:
        temp = Mass @ temp                            # (n,)

    # 2 · error vector via stage-1 linear solve
    err_vec = solve_1(f0 + temp)

    # 3 · scaled Euclidean norm
    err = torch.linalg.norm(err_vec / Scal) / sqrt_n
    err = max(err.item(), 1e-10)                      # avoid exact zero

    # 4 · optional second evaluation if err > 1 and (First or Reject)
    if err >= 1.0 and (First or Reject):
        # call ODE RHS with perturbed state
        sig = len(inspect.signature(OdeFcn).parameters)
        if sig > 2:
            f_new = OdeFcn(t, y + err, *ode_args)
        else:
            f_new = OdeFcn(t, y + err)

        Stat.FcnNbr += 1

        err_vec = solve_1(f_new + temp)
        err     = (
            torch.linalg.norm(err_vec / Scal) / sqrt_n
        ).item()
        err     = max(err, 1e-10)

    return err, Stat