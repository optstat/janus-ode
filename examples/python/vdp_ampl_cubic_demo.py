# --------------------------------------------
# Kelley pseudo-transient solver (scalar case)
# Van-der-Pol amplitude cubic demo
# --------------------------------------------
import math

# residual  F(R, μ)  and its Jacobian  J = ∂F/∂R
def F(R, mu, eps=1.0):
    return -0.25 * mu * R**3 + mu * R - eps

def J(R, mu):
    return mu * (1.0 - 0.75 * R**2)

def kelly_scalar(mu, R0, delta=2.0, beta=1.0,
                 tol=1e-10, max_iter=50):
    """
    Pseudo-transient Kelley corrector for the scalar equation F(R,μ)=0
        (  J(R,μ)  +  1/delta  ) * ΔR  =  -β F
    """
    R = float(R0)
    for _ in range(max_iter):
        f = F(R, mu)
        if abs(f) < tol:
            return R
        step = -beta * f / (J(R, mu) + 1.0 / delta)
        R += step
    raise RuntimeError("Kelley failed to converge")

# ------ demo: μ > μ* so two positive roots exist -----------------
mu  = 1.5                          # continuation parameter
# choose a start near the upper branch
root_upper = kelly_scalar(mu, R0=1.6, delta=5.0)
# choose a start nearer the lower branch
root_lower = kelly_scalar(mu, R0=0.9, delta=1.0)

print(f"μ = {mu}")
print(f"  upper-branch root = {root_upper:.10f}  F = {F(root_upper, mu):.2e}")
print(f"  lower-branch root = {root_lower:.10f}  F = {F(root_lower, mu):.2e}")
