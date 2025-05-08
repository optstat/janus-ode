#!/usr/bin/env python3
"""
tests/test_vdp_bdf2.py
Stiff Van-der-Pol oscillator (μ = 1000) integrated by the BDF-2 backend
of CommonIntegrator.

Pass criterion:
    ‖y_BDF2(tf) − y_scipy(tf)‖_∞  < 2e-3
"""

import math, time, torch
from pathlib import Path
from odesolvers.common_integrator import CommonIntegrator

# ----------------------------------------------------------------------
#  reference solution via SciPy  (only for the test – optional import)
try:
    from scipy.integrate import solve_ivp
except ImportError:
    solve_ivp = None                    # will skip the reference check
# ----------------------------------------------------------------------


def vdp_rhs(t: float, y: torch.Tensor, mu: float = 1_000.0) -> torch.Tensor:
    """Stiff van der Pol RHS written in torch."""
    return torch.tensor([y[1],
                         mu * ((1.0 - y[0] ** 2) * y[1] - y[0])],
                        dtype=y.dtype)


def integrate_bdf2(mu=1000.0, t0=0.0, tf=0.01, y0=None, h0=1e-4):
    """Run CommonIntegrator in BDF-2 mode and return final state Tensor."""
    if y0 is None:
        y0 = torch.tensor([2.0, 0.0], dtype=torch.float64)

    solver = CommonIntegrator(lambda t, y: vdp_rhs(t, y, mu),
                              t0, y0, tf,
                              mode="bdf2",
                              gmres=False,          # dense LU
                              h_init=h0)

    t_out, y_out = solver.run()
    return y_out[-1], solver.ctx.stats["step"]


def reference_scipy(mu=1000.0, t0=0.0, tf=0.01, y0=None):
    import numpy as np
    if y0 is None:
        y0 = np.array([2.0, 0.0], dtype=float)

    def rhs(t, y):               # NumPy version for SciPy
        return np.array([y[1], mu * ((1 - y[0] ** 2) * y[1] - y[0])])

    sol = solve_ivp(rhs, (t0, tf), y0, method="Radau",
                    rtol=1e-10, atol=1e-12)
    return torch.as_tensor(sol.y[:, -1], dtype=torch.float64)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)

    y_bdf2, nstep = integrate_bdf2()

    print(f"BDF-2 accepted steps : {nstep}")
    print(f"y_BDF2(tf)          : {y_bdf2.tolist()}")

    if solve_ivp is None:
        print("\nSciPy not installed → skipped reference comparison.")
        exit(0)

    y_ref = reference_scipy()
    err   = torch.linalg.norm(y_bdf2 - y_ref, ord=float("inf")).item()

    print(f"reference Radau y   : {y_ref.tolist()}")
    print(f"‖error‖_inf         : {err:.3e}")

    # simple pass/fail
    tol = 2e-3
    assert err < tol, f"BDF-2 error {err:.2e} exceeds tol {tol:.1e}"
    print("\n✅  BDF-2 test passed.")