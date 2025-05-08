# solvrad.py
import torch
from typing import List, Callable


def solve_radau(
    z: torch.Tensor,                      # (n, s) – mutated in place
    w: torch.Tensor,                      # (n, s)
    ValP: torch.Tensor,                   # (s,)   γ, Re α₁, Im α₁, …
    h: float,
    solve_fns: List[Callable],            # one closure per Radau stage block
    Mass: torch.Tensor | None = None,
    real_layout: bool = True,             # True for split-real layout
):
    """
    Overwrites z with the solution of (γₖ M – J) zₖ = rhsₖ  for each stage.
    """
    inv_h = 1.0 / h
    n, s  = z.shape

    Mw = Mass @ w if Mass is not None else w          # (n, s)

    if real_layout:
        # --- stage 0: purely real -----------------------------------------
        gamma0 = ValP[0].real * inv_h                 # real scalar
        rhs0   = z[:, 0] - gamma0 * Mw[:, 0]
        z[:, 0] = solve_fns[0](rhs0).real                  # stays real
        k = 1  # index into solve_fns
        for col in range(1, s, 2):                   # (Re, Im) pairs
            re =  ValP[col]     * inv_h
            im =  ValP[col + 1] * inv_h

            rhs_re = z[:, col]     - re * Mw[:, col]     + im * Mw[:, col + 1]
            rhs_im = z[:, col + 1] - im * Mw[:, col]     - re * Mw[:, col + 1]
            rhs_c  = rhs_re + 1j * rhs_im

            sol_c  = solve_fns[k](rhs_c)             # complex solve
            k += 1

            z[:, col]     = sol_c.real
            z[:, col + 1] = sol_c.imag

    else:  # ------ fully complex layout ------------------------------------
        for k in range(s):
            γ = ValP[k] * inv_h
            rhs = z[:, k] - γ * Mw[:, k]
            z[:, k] = solve_fns[k](rhs)

    return z