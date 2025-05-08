#!/usr/bin/env python3
import math, numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ---------- problem data -----------------------------------------
x0        = np.array([2.0, -1.0])
UMAX      = 1.0
W         = 1.0
NT_BASE   = 150
ABS_TOL   = 1e-10
SER_RATIO = 0.7

DELTA0, DELTA_MAX = 25.0, 25.0      # large Δ ⇒ gradient-like first step
FD_EPS            = 1e-6

# ---------- helper functions -------------------------------------
def t_lower_bound(x1, x2, umax):
    disc = math.sqrt(x2**2 + 2*umax*abs(x1))
    return 2*(disc - x2)/umax

def control_opt(l2):                 # λ₀ = −1 convention
    u =  l2 / W
    return max(-UMAX, min(UMAX, u))

def NT_for(tf):
    return int(max(NT_BASE, 40*tf))

# ---------- forward / backward sweeps ----------------------------
def forward(l2_path, tf, x0):
    t = np.linspace(0.0, tf, NT_for(tf))
    l2_of_t = interp1d(t, l2_path, kind='cubic', fill_value="extrapolate")
    def rhs(_, x):
        return [x[1], -x[0] + control_opt(l2_of_t(_))]
    y = solve_ivp(rhs, (t[0], t[-1]), x0, t_eval=t, method='Radau',
                  rtol=1e-8, atol=1e-8).y.T
    return t, y

def backward(lam_tf, tf, X_path):
    t = np.linspace(0.0, tf, NT_for(tf))
    x_of_t = interp1d(t, X_path, axis=0, kind='cubic',
                      fill_value="extrapolate")
    def rhs(_, l):
        return [+l[1], -l[0]]
    y = solve_ivp(rhs, (t[-1], t[0]), lam_tf, t_eval=t[::-1],
                  method='Radau', rtol=1e-8, atol=1e-8).y.T[:, ::-1]
    return y

# ---------- residual for fixed tf --------------------------------
def residual(z, tf, x0):
    lam_tf = np.array(z)
    t      = np.linspace(0.0, tf, NT_for(tf))

    # fixed-point for λ₂-path (unchanged)
    l2 = np.full_like(t, lam_tf[1])
    for _ in range(40):
        _, X = forward(l2, tf, x0)
        L    = backward(lam_tf, tf, X)
        newl2 = L[:, 1]
        if np.max(np.abs(newl2 - l2)) < 1e-7:
            break
        l2 = 0.5*newl2 + 0.5*l2

    Xf = X[-1]
    return np.array([Xf[0], Xf[1]])     # <<< only 2 components

def jac_fd(z, tf, x0):
    F0 = residual(z, tf, x0)
    J  = np.zeros((2, 2))
    for k in range(2):
        dz = z.copy(); dz[k] += FD_EPS
        J[:, k] = (residual(dz, tf, x0) - F0) / FD_EPS
    return J, F0

# ---------- fixed-point iteration for λ₂-path --------------------
def solve_fixed_tf(tf, x0, z0):
    z, delta = z0.copy(), DELTA0
    for _ in range(50):
        J, F = jac_fd(z, tf, x0)
        if np.linalg.norm(F) < ABS_TOL:
            return z, True
        A = J + np.eye(2)/delta           # 3×2
        s, *_ = np.linalg.lstsq(A, -F, rcond=None)
        step, best = 1.0, z.copy()
        for _ in range(25):
            z_new = z + step*s
            if np.linalg.norm(residual(z_new, tf, x0)) \
               <= SER_RATIO*np.linalg.norm(F):
                z = z_new; break
            step *= 0.5
        else:
            return best, False
    return z, False

# ---------- outer time-homotopy ----------------------------------
t_min = t_lower_bound(*x0, UMAX)
T0    = 1.6 * t_min
steps = 20
grid  = np.linspace(T0, t_min, steps)

# analytic unit-norm seed λ(tf)
norm    = np.linalg.norm(x0)
z_guess = np.array([-x0[0]/norm, -x0[1]/norm])

print(f"t_min = {t_min:.4f}   starting T0 = {T0:.4f}")
for k, tf in enumerate(grid):
    z_guess, ok = solve_fixed_tf(tf, x0, z_guess)
    print(f"step {k:2d} | tf={tf:6.3f} | ‖F‖="
          f"{np.linalg.norm(residual(z_guess,tf,x0)):.2e} | {'OK' if ok else 'FAIL'}")
    if not ok:
        tf = grid[max(0,k-1)]          # last successful horizon
        break

print("\n≈ Minimum-time solution")
print(f"tf*     = {tf:.6f} s")
print(f"λ1(tf*) = {z_guess[0]:+.6f}")
print(f"λ2(tf*) = {z_guess[1]:+.6f}")