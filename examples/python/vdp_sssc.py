import numpy as np, time
from numpy.linalg import solve, norm, eigvals, cond

μ  = 10.0
ρ_in, ρ_out = 0.95, 0.98
ε  = 1e-12  # regularization for SER-B

def F(x):  return np.array([x[1], μ*(1-x[0]**2)*x[1] - x[0]])
def JF(x):
    x1,x2 = x
    return np.array([[0.,1.],[-2*μ*x1*x2-1., μ*(1-x1**2)]])
def G (x,lam): return lam*F(x) + (1-lam)*x
def JG(x,lam): return lam*JF(x)+ (1-lam)*np.eye(2)

# ───── SER-B (now returns the inner-iteration count) ─────────────────
def serB_corrector(x0, Gfun, Jfun, beta=0.4, delta0=1e-2,
                   tol=1e-10, maxit=100, rho_in=ρ_in):
    x, Δ = x0.copy(), delta0
    V_old = 0.5*norm(Gfun(x))**2
    inner = 0                                   # ← counter

    for _ in range(maxit):
        inner += 1
        A = np.eye(len(x))/Δ + Jfun(x)+ ε*np.eye(len(x))
        try:  s = solve(A, -Gfun(x))
        except np.linalg.LinAlgError:
            Δ *= 0.5;  continue

        x_trial = x + s
        try:
            δ = solve(Jfun(x_trial), -Gfun(x_trial))
            if norm(δ) > 1.: δ /= norm(δ)
            x_trial += δ
        except np.linalg.LinAlgError:
            pass

        V_new = 0.5*norm(Gfun(x_trial))**2
        if V_new > max(5e-2, rho_in*V_old):
            Δ*=0.5
            if Δ<1e-14: return x, False, inner
            continue

        ratio = V_new / V_old
        x, V_old = x_trial, V_new
        if V_new < 0.5*tol**2: return x, True, inner
        Δ = np.clip(Δ*ratio**(beta/2), 1e-14, 1e6)

    return x, False, inner

# ───── plain Newton polish ───────────────────────────────────────────
def refine(x, maxit=5):
    for _ in range(maxit):
        try:    dx = solve(JF(x), -F(x))
        except np.linalg.LinAlgError:
            break
        x += dx
        if norm(F(x)) < 1e-12: break
    return x

# ───── Akella SSSC with instrumentation ─────────────────────────────
def sssc_akella(x0, n_max=200, lam_tol=1e-6,
                gamma_shift=0.7, step0=0.05, step_max=0.5,
                verbose=True):
    lam, step, x = 0.0, step0, x0.copy()
    outer_it = inner_it = rejected = 0
    t0 = time.perf_counter()

    while lam < 1.0 - lam_tol and n_max:
        outer_it += 1
        lam_new = min(1.0, lam + step)

        ρ = max(abs(eigvals(JG(x, lam_new))))
        α = gamma_shift*(ρ+1)*np.sqrt(max(0.,1-lam))
        Gs = lambda z: G(z, lam_new) + α*(z - x)
        JGs= lambda z: JG(z, lam_new)+ α*np.eye(2)

        x_new, ok, inner_now = serB_corrector(x, Gs, JGs)
        inner_it += inner_now

        if not ok or \
           norm(F(x_new)) > ρ_out*norm(F(x))+1e-12 or \
           0.5*norm(G(x_new,lam_new))**2 > max(1e-6,
                ρ_out*0.5*norm(G(x,lam))**2):
            step *= 0.5
            rejected += 1
            if step < 1e-15:  raise RuntimeError("Δλ too small")
            continue

        lam, x, n_max = lam_new, x_new, n_max-1
        ρ  = max(abs(eigvals(JG(x, lam))))
        α  = gamma_shift * (ρ + 1)          # ← add this line
        condJ = cond(JG(x, lam) + α * np.eye(2))
        # 🔍 Print detailed debug info
        if verbose:
            print(f"λ={lam:.3f}, ‖F‖={norm(F(x)):.2e}, ρ={ρ:.2f}, cond(JG+αI)={condJ:.2e}")
 
        if   condJ < 5e2: step = min(step_max, step*1.8)
        elif condJ < 2e3: step = min(step_max, step*1.2)

    
    if lam >= 1.0 - lam_tol:
        ρ  = max(abs(eigvals(JG(x, lam))))  # Recompute with final lam
        α  = gamma_shift * (ρ + 1)
        x = refine(x)
    wall = time.perf_counter() - t0
    meta = dict(outer=outer_it, inner=inner_it,
                rej=rejected, time=wall, F=norm(F(x)))
    return x, meta

# ───── smoke-test ───────────────────────────────────────────────────
if __name__ == "__main__":
    x0 = np.array([5., 0.5])
    root, meta = sssc_akella(x0, verbose=False)
    print("root =", root, "\nmeta =", meta)