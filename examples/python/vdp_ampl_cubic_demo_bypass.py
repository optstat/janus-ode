import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")      
# -------------------------------
# Van-der-Pol amplitude residual
# -------------------------------
def F(R, mu, eps=1.0):
    return -0.25 * mu * R**3 + mu * R - eps          # cubic

def J(R, mu):
    return mu * (1.0 - 0.75 * R**2)                  # ∂F/∂R

def F_s(R, mu):
    return -0.25 * R**3 + R                          # ∂F/∂mu

# --------------------------------
# Scalar Kelley pseudo-transient
# --------------------------------
def kelly_scalar(mu, R0, delta=2.0, beta=1.0,
                 tol=1e-10, max_iter=60):
    R     = float(R0)
    histR = []
    histF = []
    for _ in range(max_iter):
        f = F(R, mu)
        histR.append(R)
        histF.append(f)
        if abs(f) < tol:
            return R, np.array(histR), np.array(histF), True
        step = -beta * f / (J(R, mu) + 1.0/delta)
        R += step
    return R, np.array(histR), np.array(histF), False   # no conv

# --------------------------------
# Minimal Δμ* jump computation
# --------------------------------
def minimal_mu_jump(R_hat, mu_hat, overshoot=1.1):
    rho   = F(R_hat, mu_hat)       # w = 1 in scalar case
    sigma = F_s(R_hat, mu_hat)
    if abs(sigma) < 1e-12:
        raise ZeroDivisionError("σ≈0: wide infeasible band")
    return mu_hat + overshoot * (-rho / sigma)

# --------------------------------
# Demo
# --------------------------------
mu_infeasible = 1.0        # inside  (0 , μ★≈1.299)
R0            = 1.0

# 1) Plain Kelley -> stall
R_stall, trajR1, trajF1, ok1 = kelly_scalar(mu_infeasible, R0)

print(f"Plain Kelley at μ={mu_infeasible}: converged={ok1}")
print(f"  final residual |F| = {abs(trajF1[-1]):.3f}")

# 2) Compute jump
mu_new = minimal_mu_jump(R_hat=R_stall, mu_hat=mu_infeasible)
print(f"Minimal jump → new μ = {mu_new:.4f}")

# 3) Kelley after jump
R_good, trajR2, trajF2, ok2 = kelly_scalar(mu_new, R_stall)
print(f"Kelley after jump: converged={ok2}")
print(f"  residual at root = {abs(trajF2[-1]):.1e}")

# 4) Plot residual magnitude
plt.semilogy(np.abs(trajF1), 'r-o', label=f'μ={mu_infeasible} (stall)')
plt.semilogy(len(trajF1) + np.arange(len(trajF2)),
             np.abs(trajF2), 'b-o',
             label=f'jump to μ={mu_new:.3f}')
plt.xlabel("Iteration (cumulative)")
plt.ylabel("|F|")
plt.title("Kelley residual: stall vs. minimal μ-jump")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig("kelley_stall_jump.png", dpi=300)
plt.close()
