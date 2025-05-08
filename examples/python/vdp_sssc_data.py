"""
generate_deltas.py  –  “greedy SSSC” that records the maximal Δλ allowed
at each outer step.  No ML code here; the resulting CSV is training data.
---------------------------------------------------------------------------
"""

import numpy as np
from numpy.linalg import solve, norm, eigvals, cond

# ─────────────────────────── User - level parameters ──────────────────────────
MU          = 10.0                 # Van-der-Pol μ
SHIFT_GAIN  = 0.7                  # γ in Akella’s α = γ (ρ+1)
RHO_IN      = 0.95                 # inner SER-B acceptance
RHO_OUT     = 0.98                 # outer acceptance
DELTA0      = 1e-2                 # initial pseudo-time step for SER-B
LAM_TOL     = 1e-6                 # stop when λ ≥ 1-LAM_TOL
STEP0       = 0.05                 # starting outer step
STEP_MAX    = 0.5                  # ceiling in the growth search
# -----------------------------------------------------------------------------


# === Residuals and Jacobians ==================================================
def F(x):
    return np.array([x[1], MU*(1 - x[0]**2)*x[1] - x[0]])

def JF(x):
    x1, x2 = x
    return np.array([[0., 1.],
                     [-2*MU*x1*x2 - 1.,  MU*(1 - x1**2)]])

def G (x, lam): return lam*F(x) + (1-lam)*x
def JG(x, lam): return lam*JF(x) + (1-lam)*np.eye(2)


# === SER-B inner corrector  (bug-fixed) ======================================
def serB_corrector(x0, Gfun, Jfun,
                   beta=0.4, delta0=DELTA0, tol=1e-10, maxit=100):

    x, Δ = x0.copy(), delta0
    V_old = 0.5*norm(Gfun(x))**2

    for _ in range(maxit):
        A = np.eye(2)/Δ + Jfun(x)
        try:
            s = solve(A, -Gfun(x))
        except np.linalg.LinAlgError:
            Δ *= 0.5;  continue

        x_trial = x + s
        # one Newton polish
        try:
            d = solve(Jfun(x_trial), -Gfun(x_trial))
            if norm(d) > 1.:  d /= norm(d)
            x_trial += d
        except np.linalg.LinAlgError:
            pass

        V_new = 0.5*norm(Gfun(x_trial))**2
        if V_new > max(5e-2, RHO_IN*V_old):
            Δ *= 0.5
            if Δ < 1e-14:  return x, False
            continue

        ratio = V_new / V_old
        x, V_old = x_trial, V_new
        if V_new < 0.5*tol**2:  return x, True

        Δ = np.clip(Δ * ratio**(beta/2), 1e-14, 1e6)

    return x, False


# === Search for the largest admissible Δλ ====================================
def max_step(x, lam, seed_step, step_max):
    """Return (Δλ_max, x_new, lam_new) or (None, None, None) if seed fails."""
    def attempt(step):
        lam_new = min(1.0, lam + step)
        ρ  = max(abs(eigvals(JG(x, lam_new))))
        α  = SHIFT_GAIN*(ρ+1)*np.sqrt(max(0., 1-lam))
        Gs = lambda z: G(z, lam_new) + α*(z - x)
        JGs= lambda z: JG(z, lam_new) + α*np.eye(2)
        x_new, ok = serB_corrector(x, Gs, JGs)
        return ok, x_new, lam_new

    step = seed_step
    ok, x_new, lam_new = attempt(step)
    if not ok:  return None, None, None

    # geometric growth until failure
    grow = 1.5
    best_step, best_x, best_lam = step, x_new, lam_new
    while best_step < step_max:
        ok, x_new, lam_new = attempt(best_step*grow)
        if not ok:  break
        best_step, best_x, best_lam = best_step*grow, x_new, lam_new
    return best_step, best_x, best_lam


# === Main outer loop that logs the data ======================================
def generate_one_run(x0, verbose=False):
    """Return X (N×5) and y (N×1) for a single λ-march from 0→1."""
    lam, x, step  = 0.0, x0.copy(), STEP0
    feats, targets = [], []

    for _ in range(200):                       # hard outer max-iters
        Δλ, x_new, lam_new = max_step(x, lam, step, STEP_MAX)

        if Δλ is None:                         # even tiny step failed
            step *= 0.5
            if step < 1e-15:
                raise RuntimeError("Solver stalled before λ=1")
            continue

        # ------- feature vector ---------------------------------------
        ρ  = max(abs(eigvals(JG(x, lam))))
        κ  = cond(JG(x, lam))
        feats.append([lam, x[0], x[1],
                      np.log10(ρ+1e-12), np.log10(κ+1e-12)])
        targets.append(Δλ)
        # ---------------------------------------------------------------

        if verbose:
            print(f"λ={lam:.3f} → {lam_new:.3f}   Δλ={Δλ:.3e}")

        lam, x, step = lam_new, x_new, Δλ      # greedy move
        if lam >= 1.0 - LAM_TOL:  break

    return np.array(feats,  dtype=np.float32), \
           np.array(targets, dtype=np.float32).reshape(-1, 1)


# === Convenience driver: many runs, CSV dump ===========================
def build_dataset(n_runs=1000, save_csv=None, seed=0):
    rng = np.random.default_rng(seed)
    Xs, ys = [], []

    for k in range(n_runs):
        # random initial guesses – wide on purpose
        x0 = rng.uniform(low=[4.0,0.5], high=[5.0, 1.0])
        X, y = generate_one_run(x0)
        Xs.append(X);  ys.append(y)

    X_all = np.vstack(Xs)
    y_all = np.vstack(ys)

    if save_csv:
        data = np.hstack([X_all, y_all])
        hdr  = "lam,x1,x2,log_rho,log_kappa,delta_max"
        np.savetxt(save_csv, data, delimiter=",", header=hdr, comments="")
        print(f"Wrote {data.shape[0]} rows to {save_csv}")

    return X_all, y_all


# === Quick demo ========================================================
if __name__ == "__main__":
    X, y = build_dataset(n_runs=1000, save_csv="deltas.csv", seed=42)
    print("Feature shape:", X.shape, "  Target shape:", y.shape)