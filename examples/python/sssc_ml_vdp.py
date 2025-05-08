#!/usr/bin/env python3
# sssc_ml_vdp.py  – ML–driven SSSC with metrics
# -------------------------------------------------------------
import numpy as np, time, torch, torch.nn as nn
from numpy.linalg import solve, norm, eigvals, cond

# ---------------- Van-der-Pol system --------------------------
MU = 10.0
def F(x):  return np.array([x[1], MU*(1-x[0]**2)*x[1] - x[0]])
def JF(x):
    x1,x2 = x
    return np.array([[0.,1.],[-2*MU*x1*x2-1., MU*(1-x1**2)]])
def G (x,lam): return lam*F(x) + (1-lam)*x
def JG(x,lam): return lam*JF(x)+ (1-lam)*np.eye(2)

# ----------- Newton polish ------------------------------------
def refine(x, maxit=5):
    for _ in range(maxit):
        try: dx = solve(JF(x), -F(x))
        except np.linalg.LinAlgError: break
        x += dx
        if norm(F(x)) < 1e-12: break
    return x

# ---------------- SER-B inner corrector (with counter) ---------
RHO_IN = 0.95
def serB_corrector(x0, Gfun, Jfun, beta=0.4, delta0=1e-2,
                   tol=1e-10, maxit=100):
    x, Δ = x0.copy(), delta0
    V_old = 0.5*norm(Gfun(x))**2
    inner = 0                                 # ← counter

    for _ in range(maxit):
        inner += 1
        A = np.eye(2)/Δ + Jfun(x)
        try:  s = solve(A, -Gfun(x))
        except np.linalg.LinAlgError:
            Δ *= 0.5; continue
        x_trial = x + s
        try:
            d = solve(Jfun(x_trial), -Gfun(x_trial))
            if norm(d) > 1.: d /= norm(d)
            x_trial += d
        except np.linalg.LinAlgError:
            pass
        V_new = 0.5*norm(Gfun(x_trial))**2
        if V_new > max(5e-2, RHO_IN*V_old):
            Δ *= 0.5
            if Δ < 1e-14: return x, False, inner
            continue
        ratio = V_new / V_old
        x, V_old = x_trial, V_new
        if V_new < 0.5*tol**2: return x, True, inner
        Δ = np.clip(Δ*ratio**(beta/2), 1e-14, 1e6)
    return x, False, inner

# ---------------- load trained network ------------------------
class DeltaNet(nn.Module):
    def __init__(self, n_in=5, n_h=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_h), nn.Tanh(),
            nn.Linear(n_h, n_h),  nn.Tanh(),
            nn.Linear(n_h, 1),    nn.Softplus())
    def forward(self, x): return self.net(x)

ckpt   = torch.load("delta_net.pt", map_location="cpu")
net    = DeltaNet(); net.load_state_dict(ckpt["model_state"]); net.eval()
x_mean, x_std = ckpt["x_mean"], ckpt["x_std"]
@torch.no_grad()
def predict_delta(feat):
    z = (feat - x_mean)/x_std
    return net(torch.tensor(z, dtype=torch.float32)).item()

# ---------------- ML-driven outer SSSC with metrics ------------
SHIFT_GAIN = 0.7
STEP_MAX, STEP_MIN = 0.5, 1e-15
RHO_OUT,   LAM_TOL = 0.98, 1e-6

def sssc_ml(x0, verbose=True):
    lam, x = 0.0, x0.copy()
    outer_it = inner_it = rejected = 0
    t0 = time.perf_counter()

    for k in range(200):
        outer_it += 1
        rho   = max(abs(eigvals(JG(x, lam))))
        kappa = cond(JG(x, lam))
        feat  = np.array([lam, x[0], x[1],
                          np.log10(rho+1e-12),
                          np.log10(kappa+1e-12)], np.float32)
        step  = min(STEP_MAX, predict_delta(feat))
        first_try = True

        while step >= STEP_MIN:
            lam_new = min(1.0, lam + step)
            alpha   = SHIFT_GAIN*(rho+1)*np.sqrt(max(0., 1-lam))
            Gs = lambda z: G(z, lam_new) + alpha*(z - x)
            JGs= lambda z: JG(z, lam_new)+ alpha*np.eye(2)
            x_new, ok, inner_now = serB_corrector(x, Gs, JGs)
            inner_it += inner_now

            if ok and \
               norm(F(x_new)) <= RHO_OUT*norm(F(x))+1e-12 and \
               0.5*norm(G(x_new,lam_new))**2 <= \
               max(1e-6, RHO_OUT*0.5*norm(G(x,lam))**2):
                break        # success
            step *= 0.5
            if first_try:
                rejected += 1
                first_try = False
        else:
            raise RuntimeError("Δλ floor reached — stalled")

        if verbose:
            print(f"{k:2d}: λ {lam:.3f}→{lam_new:.3f}  Δλ={step:.2e}  "
                  f"‖F‖={norm(F(x_new)):.2e}")

        lam, x = lam_new, x_new
        if lam >= 1.0 - LAM_TOL:
            x = refine(x)
            break

    wall = time.perf_counter() - t0
    meta = dict(outer=outer_it, inner=inner_it,
                rej=rejected, time=wall, F=norm(F(x)))
    return x, meta

# ---------------- smoke-test ------------------------------------
if __name__ == "__main__":
    x0 = np.array([5., 0.5])
    root, meta = sssc_ml(x0, verbose=False)
    print("root =", root, "\nmeta =", meta)