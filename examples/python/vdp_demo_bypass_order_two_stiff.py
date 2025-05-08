import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
matplotlib.use("Agg")

# ---------------------------------------------------------
# Modified VdP amplitude residual  (now nonlinear in μ)
# ---------------------------------------------------------
def F(R, mu):          return mu*R - 0.25*mu*R**3 + 0.10*mu**2 - 1.0
def J_R(R, mu):        return mu*(1.0 - 0.75*R**2)
def F_mu(R, mu):       return R - 0.25*R**3 + 0.20*mu
def F_mumu(R, mu):     return 0.20
# ---------------------------------------------------------
# Scalar Kelley pseudo-transient
# ---------------------------------------------------------
def kelly_scalar(mu, R, delta=2.0, beta=1.0, it=60):
    for _ in range(it):
        f = F(R, mu)
        if abs(f) < 1e-10:
            return R, True
        R += -beta * f / (J_R(R, mu) + 1.0/delta)
    return R, False

def jump_quadratic(R_hat, mu_hat, overshoot=1.05):
    rho, sig, tau = F(R_hat, mu_hat), F_mu(R_hat, mu_hat), F_mumu(R_hat, mu_hat)
    a, b, c = 0.5*tau, sig, rho
    disc = b*b - 4*a*c
    if disc < 0:
        raise ValueError("no real root – adjust start point")
    dmu = (-b + math.sqrt(disc)) / (2*a)
    return mu_hat + overshoot*dmu
# ---------------------------------------------------------
# First- and second-order μ-jumps
# ---------------------------------------------------------
def mu_jump_linear(R_hat, mu_hat, overshoot=1.1):
    rho, sigma = F(R_hat, mu_hat), F_mu(R_hat, mu_hat)
    if abs(sigma) < 1e-12:
        raise ZeroDivisionError("σ≈0 ⇒ linear formula useless")
    return mu_hat + overshoot * (-rho / sigma)

def mu_jump_quadratic(R_hat, mu_hat, overshoot=1.05):
    rho, sigma, tau = F(R_hat, mu_hat), F_mu(R_hat, mu_hat), F_mumu(R_hat, mu_hat)
    # Solve ½ τ Δμ² + σ Δμ + ρ = 0   → choose root that increases μ
    a, b, c = 0.5*tau, sigma, rho
    disc = b*b - 4*a*c
    if disc < 0:
        raise RuntimeError("No real root – check your math!")
    dmu = (-b + np.sqrt(disc)) / (2*a)
    return mu_hat + overshoot * dmu

# ---------------------------------------------------------
# Demo from a DEEP infeasible μ
# ---------------------------------------------------------
R_hat, mu_hat = 2.0, 0.30          # the point you just evaluated
rho, sigma, tau = F(R_hat, mu_hat), F_mu(R_hat, mu_hat), F_mumu(R_hat, mu_hat)

# --- 1st-order jump -------------------------------------------------
mu_lin  = mu_hat + (-rho / sigma)          # overshoot = 1.0 for clarity
# → Δμ ≈ +16.52   ⇒   μ ≈ 16.82

# --- 2nd-order jump -------------------------------------------------
a, b, c = 0.5*tau, sigma, rho              # ½ τ Δμ² + σ Δμ + ρ = 0
disc    = b*b - 4*a*c
dmu_quad = (-b + math.sqrt(disc)) / (2*a)  # positive root
mu_quad  = mu_hat + dmu_quad               # ≈ 3.16

print(f"linear  jump μ  = {mu_lin:.4f}")
print(f"quadratic jump μ = {mu_quad:.4f}")
# Restart Kelley at each μ to see convergence behaviour
R_lin , ok_lin  = kelly_scalar(mu_lin , R_hat)
R_quad, ok_quad = kelly_scalar(mu_quad, R_hat)

print(f"Kelley @ μ_lin  converged={ok_lin}")
print(f"Kelley @ μ_quad converged={ok_quad}")