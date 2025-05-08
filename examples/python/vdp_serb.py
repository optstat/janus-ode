import numpy as np

μ          = 1000.0
BETA       = 0.4          # exponent in Kelley formula (0.4–0.6 typical)
DELTA0     = 1e-8         # very small initial pseudo-time step
DELTA_MIN  = 1e-12
DELTA_MAX  = 1e4
TOL        = 1e-10        # residual tolerance
MAXIT      = 10

# --------  problem definition  ---------------------------------------------
def F(x):
    x1, x2 = x
    return np.array([x2,
                     μ * (1.0 - x1**2) * x2 - x1])

def J(x):
    x1, x2 = x
    return np.array([[0.0,                     1.0],
                     [-1.0 - 2.0 * μ * x1 * x2,
                      μ * (1.0 - x1**2) ]])

# --------  SER-B PTC solver  -----------------------------------------------
def ptc_serB(x0, beta=BETA, delta0=DELTA0,
             delta_min=DELTA_MIN, delta_max=DELTA_MAX,
             tol=TOL, maxit=MAXIT, verbose=True):
    """
    Kelley SER-B pseudo-transient continuation with an embedded
    Newton correction at *every* iteration.
    """
    x      = x0.astype(float)
    Δ      = delta0
    Fnrm   = np.linalg.norm(F(x))

    for k in range(maxit):
        # (1) SER step  :  (I/Δ + J) s = −F
        A = np.eye(len(x)) / Δ + J(x)
        s = np.linalg.solve(A, -F(x))
        x += s

        # (2) Newton correction (un-shifted)
        δ = np.linalg.solve(J(x), -F(x))
        x += δ

        Fval  = F(x)
        Fnrm_new = np.linalg.norm(Fval)

        if verbose:
            print(f"{k:03d}:  ‖F‖ = {Fnrm_new:9.2e}   Δ = {Δ:9.2e}")

        # (3) smooth Kelley update
        ratio = Fnrm_new / max(Fnrm, 1e-30)
        Δ = np.clip(Δ * ratio**beta, delta_min, delta_max)

        Fnrm = Fnrm_new
        if Fnrm < tol:
            if verbose:
                print("      ✓ converged")
            break
    return x

# --------  demo  ------------------------------------------------------------
if __name__ == "__main__":
    for guess in ([1.0, 1.0],
                  [0.8, 0.0],
                  [-2.0, 2.0],
                  [10.0, 10.0]):
        root = ptc_serB(np.array(guess), verbose=True)
        print(f"start {str(guess):>12}  →  root {root}   ‖F‖={np.linalg.norm(F(root)):.1e}")
