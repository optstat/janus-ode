import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------
# Modified VdP amplitude residual  (now nonlinear in μ)
# ---------------------------------------------------------
def F(R, mu, eps=1.0):
    return mu*R - 0.25*mu*R**3 + 0.10*mu**2 - eps          # +0.1 μ² term

def J_R(R, mu):            # ∂F/∂R
    return mu - 0.75*mu*R**2

def F_mu(R, mu):           # ∂F/∂μ  = σ
    return R - 0.25*R**3 + 0.20*mu

def F_mumu(R, mu):         # ∂²F/∂μ² = τ
    return 0.20

# ---------------------------------------------------------
# Scalar Kelley pseudo-transient
# ---------------------------------------------------------
def kelly_scalar(mu, R0, delta=2.0, beta=1.0,
                 tol=1e-10, max_iter=60):
    R, hist = float(R0), []
    for _ in range(max_iter):
        f = F(R, mu)
        hist.append(abs(f))
        if abs(f) < tol:
            return R, np.array(hist), True
        step = -beta * f / (J_R(R, mu) + 1.0/delta)
        R += step
    return R, np.array(hist), False            # stalled

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
mu0   = 0.20            # well inside infeasible band
R0    = 1.0

# (1) Kelley stalls
R_stall, res1, ok1 = kelly_scalar(mu0, R0)
print(f"Stalled at μ={mu0}: converged={ok1}, |F|={res1[-1]:.3e}")

# (2) Linear jump    (big overshoot)
mu_lin = mu_jump_linear(R_stall, mu0)
R_lin, res_lin, ok_lin = kelly_scalar(mu_lin, R_stall)
print(f"Linear jump to μ={mu_lin:.3f}: converged={ok_lin}")

# (3) Quadratic jump (smaller, just enough)
mu_quad = mu_jump_quadratic(R_stall, mu0)
R_quad, res_quad, ok_quad = kelly_scalar(mu_quad, R_stall)
print(f"Quadratic jump to μ={mu_quad:.3f}: converged={ok_quad}")

# ---------------------------------------------------------
# Plot residual histories
# ---------------------------------------------------------
plt.semilogy(res1, 'k--', label=f'μ={mu0} (stall)')
plt.semilogy(len(res1)+np.arange(len(res_lin)), res_lin, 'r-o',
             label=f'linear jump → μ={mu_lin:.3f}')
plt.semilogy(len(res1)+np.arange(len(res_quad)), res_quad, 'b-o',
             label=f'quadratic jump → μ={mu_quad:.3f}')
plt.xlabel('Iteration (cumulative)')
plt.ylabel('|F|')
plt.title('Kelley: wide infeasible band – linear vs. quadratic μ-jump')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.tight_layout()
plt.savefig('kelley_wide_jump.png', dpi=300)
plt.close()
