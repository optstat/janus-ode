"""
Forced Van‑der‑Pol (minimum‑time transfer)
========================================

This module sets up the **Pontryagin Minimum Principle** shooting equations for a
forced Van‑der‑Pol oscillator.  It is intended as the *first real* test‑bed for
the SSSC + SER‑B continuation engine you already have.

Problem statement
-----------------
Minimise terminal time **T** subject to

    ẋ₁ = x₂
    ẋ₂ = μ(1 − x₁²)x₂ − x₁ + u,  |u| ≤ u_max

with fixed endpoints **x(0)=x0**, **x(T)=xf**.
Time is made a state through *ṫ = 1* ⇒ we keep *p₃=−1* so that
H(T)=0 is automatically enforced by the additional residual below.

The control is homotopically smoothed

    u(t) = −u_max · tanh(λ p₂).

For λ → ∞ this recovers true bang–bang.

Unknown shooting vector
-----------------------
    z = [p₁(0), p₂(0), T]^T ∈ ℝ³.

Residual returned by ``F(z)`` is

    r = [x(T) − xf, H(T)] ∈ ℝ³.

A finite‑difference Jacobian ``JF`` is provided so the same SSSC driver can be
re‑used unchanged.
"""

from __future__ import annotations
import numpy as np
from numpy.linalg import norm
from scipy.integrate import solve_ivp

###############################################################################
# Control law & ODE right‑hand side                                           #
###############################################################################

def control(p2: float, lam: float = 10.0, u_max: float = 1.0) -> float:
    """Smoothed bang–bang control u = −u_max·tanh(λ p₂)."""
    return -u_max * np.tanh(lam * p2)

def rhs(t: float, y: np.ndarray, *, mu: float, lam: float, u_max: float) -> np.ndarray:
    """Time derivative for state + costate (x₁,x₂,p₁,p₂)."""
    x1, x2, p1, p2 = y
    u = control(p2, lam, u_max)

    dx1 = x2
    dx2 = mu * (1 - x1**2) * x2 - x1 + u

    dp1 = p2 * (2 * mu * x1 * x2 + 1)
    dp2 = -p1 - mu * (1 - x1**2) * p2

    return np.array([dx1, dx2, dp1, dp2])

###############################################################################
# Shooting residual and finite‑difference Jacobian                            #
###############################################################################

def shooting_residual(
    z: np.ndarray,
    *,
    x0: np.ndarray,
    xf: np.ndarray,
    mu: float = 10.0,
    lam: float = 10.0,
    u_max: float = 1.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> np.ndarray:
    """Integrate ODE and return 3‑vector residual (terminal state error & H)."""
    p1_0, p2_0, T = z

    y0 = np.hstack((x0, [p1_0, p2_0]))

    sol = solve_ivp(
        rhs,
        (0.0, T),
        y0,
        method="Radau",
        rtol=rtol,
        atol=atol,
        args=(),
        dense_output=False,
        vectorized=False,
        kwargs=dict(mu=mu, lam=lam, u_max=u_max),
    )

    if not sol.success:
        # Return a *large* residual on integration failure so the outer solver
        # recognises trouble and backs off.
        return np.full(3, 1e3)

    xT = sol.y[:2, -1]
    p1T, p2T = sol.y[2:, -1]

    # Hamiltonian at t = T (with p3 ≡ -1)
    uT = control(p2T, lam, u_max)
    HT = p1T * xT[1] + p2T * (mu * (1 - xT[0] ** 2) * xT[1] - xT[0] + uT) - 1.0

    return np.hstack((xT - xf, HT))


def jacobian_fd(F, z: np.ndarray, eps: float = 1e-6, **Fargs) -> np.ndarray:
    """Simple central finite‑difference Jacobian."""
    n = len(z)
    J = np.zeros((n, n))
    f0 = F(z, **Fargs)
    for i in range(n):
        dz = np.zeros(n)
        dz[i] = eps
        fp = F(z + dz, **Fargs)
        fm = F(z - dz, **Fargs)
        J[:, i] = (fp - fm) / (2 * eps)
    return J

###############################################################################
# Quick smoke‑test                                                             #
###############################################################################

if __name__ == "__main__":
    x0 = np.array([2.0, 0.0])          # start on the right branch
    xf = np.array([-2.0, 0.0])         # target on the left branch

    # Crude initial guess: small negative p2 pushes control ≈ +u_max
    z0 = np.array([0.0, -0.1, 4.0])    # [p1_0, p2_0, T]

    r = shooting_residual(z0, x0=x0, xf=xf, mu=10.0, lam=5.0, u_max=1.0)
    print("‖residual‖ =", norm(r))

    # Numerical Jacobian demo
    J = jacobian_fd(shooting_residual, z0, x0=x0, xf=xf)
    print("cond(J) =", np.linalg.cond(J))
