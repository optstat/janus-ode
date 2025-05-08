#!/usr/bin/env python3
"""
radau_stage_driver.py
Validate stage solves for 3-stage Radau-IIA (order 5) with either
  • PETSc GMRES (+ILU)  --method gmres
  • torch.linalg.solve (dense LU) --method lu
Checks residual of stage 0.
"""

import argparse
import torch
import radau_tables as rt
from .linsolve import decom_rc_gmres, build_lu_solvers
from .solvrad import solve_radau
from .estrad import estrad
from types  import SimpleNamespace


def your_rhs_function(t, y, *args):
    """
    Replace this with your actual ODE RHS function.
    For testing, it can be a simple linear function.
    """
    # Example: dy/dt = -y
    return -y  # or any other function of t, y, and args

# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["gmres", "lu"], default="gmres")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)

    # synthetic stiff system --------------------------------------------------
    n = 6
    Mass = torch.eye(n)
    Jac  = torch.randn(n, n) * 0.3 - torch.eye(n)
    h    = 1e-2

    # Radau-IIA (s = 3) coeffs -----------------------------------------------
    _, _, _, ValP3, _ = rt.coertv3(dtype=torch.float64)  # γ, Re α, Im α

    # build stage solvers -----------------------------------------------------
    if args.method == "gmres":
        solve_fns = decom_rc_gmres(h, ValP3, Mass, Jac,
                                   real=True, tol=1e-12, maxits=400, pc="ilu")
    else:
        solve_fns = build_lu_solvers(h, ValP3, Mass, Jac, real_layout=True)

    # work arrays -------------------------------------------------------------
    s = len(ValP3)               # = 3  (γ, Re, Im columns)
    z = torch.randn(n, s)
    w = torch.randn(n, s)

    # stage solve -------------------------------------------------------------
    z = solve_radau(z, w, ValP3, h, solve_fns,
                    Mass=Mass, real_layout=True)

    # residual for stage 0 ----------------------------------------------------
    gamma = (ValP3[0] / h).to(torch.complex128)
    M_c   = Mass.to(torch.complex128)
    J_c   = Jac.to(torch.complex128)
    x0_c  = z[:, 0].to(torch.complex128)            # promote to C128
    b0    = torch.zeros_like(x0_c)                  # same dtype

    res   = gamma * (M_c @ x0_c) - J_c @ x0_c - b0

    print(f"[{args.method}]  ‖(γ₁ M − J)x − b‖ = {torch.linalg.norm(res):.2e}")

    # after you have z, solve_fns, Mass, etc.
    Dd3 = rt.coertv3(dtype=torch.float64)[4]          # error coeffs
    Scal = torch.ones(n)                              # example scaling
    Stat = SimpleNamespace(FcnNbr=0)
    t = 0.0                                          # initial time
    y = torch.randn(n)                              # initial state

    err, Stat = estrad(
        z, Dd3, h,
        solve_1 = solve_fns[0],       # LU or GMRES, same interface
        Mass    = Mass,
        Scal    = Scal,
        f0      = torch.zeros(n),     # replace with f(t,y) from predictor
        First   = True,
        Reject  = False,
        Stat    = Stat,
        OdeFcn  = your_rhs_function,  # use actual RHS here
        t       = t,
        y       = y,
        ode_args = (),                # extra args if needed
    )

    print("local error estimate =", err)
    print("function calls so far =", Stat.FcnNbr)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()