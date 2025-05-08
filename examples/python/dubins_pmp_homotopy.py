#!/usr/bin/env python3
# ------------------------------------------------------------------
# Dubins-vehicle minimum-time PMP solver
#   – Kelley pseudo-transient continuation (Ψtc)
#   – SER-B step-size controller
# Unknowns:  z = [ λ1(tf), λ2(tf), v ]  (v maps to tf)
# Terminal conditions:
#   x1(tf) = 0     (reach origin)
#   x2(tf) = 0
#   λ1(tf)^2 + λ2(tf)^2 = 1   (costate scale fix)
# Free variables:
#   heading θ(tf)  free  ⇒  λ3(tf) = 0 (enforced explicitly)
# ------------------------------------------------------------------
import math, numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# ───── problem data ───────────────────────────────────────────────
X0_easy  = np.array([1.0, 0.0, np.pi-0.5])    # facing goal
X0_hard  = np.array([1.0, 0.0, 0.0])      # facing wrong way
TARGET = np.array([0.0, 0.0])           # origin
W    = 0.1                                 # curvature penalty
NT   = 100                              # grid points in [0, tf]
RTOL = ATOL = 1e-7                      # ODE tolerances
UMAX = 1.0                            # max steering angle
ABS_TOL = 1e-8           # put up with machine noise
def X0_of(alpha: float):
    return (1.0-alpha)*X0_easy + alpha*X0_hard
# ───── time-map v ↦ tf (keeps tf ∈ (0,T_MAX]) ────────────────────
T_MAX = 20.0
def tf_of(v: float) -> float:
    return float(np.clip(v, 1e-6, T_MAX))   # identity + clipping

# ───── Kelley / SER-B parameters ─────────────────────────────────
DELTA0       = 1.0
DELTA_MIN    = 1e-8
DELTA_MAX    = 1.0
NEWTON_IT    = 100
SUCCESS_RATIO = 0.7
FD_EPS       = 1e-8

# ───── helper functions ──────────────────────────────────────────
def bang(l3: float) -> float:                # optimal control
    u= -l3 / W
    return u

def H(x1, x2, th, l1, l2, l3, u):           # Hamiltonian
    return l1*math.cos(th) + l2*math.sin(th) + l3*u + 0.5*W*u*u + 1.0

# ───── global work arrays (resized when tf changes) ──────────────
t_grid = np.linspace(0.0, 1.0, NT)
X_traj = np.zeros((NT, 3))      # ← initialise to zeros
L_traj = np.zeros((NT, 3))      # ← initialise to zeros

# ───── backward (costate) sweep λ̇ = −∂H/∂x ──────────────────────
def backward(lam_tf: np.ndarray) -> np.ndarray:
    x_of_t = interp1d(t_grid, X_traj, axis=0,
                      kind='cubic', fill_value="extrapolate")
    def rhs(t, l):
        l1, l2, l3   = l
        x1, x2, th   = x_of_t(t)
        sinth, costh = math.sin(th), math.cos(th)
        dl1 = 0.0                           # ∂H/∂x1 = 0
        dl2 = 0.0                           # ∂H/∂x2 = 0
        dl3 = -(l1 * -sinth + l2 * costh)   # ∂H/∂θ
        return [dl1, dl2, dl3]

    sol = solve_ivp(rhs, (t_grid[-1], t_grid[0]), lam_tf,
                    method='Radau', t_eval=t_grid[::-1],
                    rtol=RTOL, atol=ATOL)
    lam = sol.y.T[:, ::-1]                 # forward order
    if lam.shape[0] < NT:                  # pad if early exit
        pad = np.repeat(lam[-1][None, :], NT - lam.shape[0], axis=0)
        lam = np.vstack((lam, pad))
    return lam

# ───── forward (state) sweep ẋ = f(x,u) ─────────────────────────
def forward(lam3_path: np.ndarray, x0: np.ndarray) -> np.ndarray:
    l3_of_t = interp1d(t_grid, lam3_path,
                       kind='cubic', fill_value="extrapolate")
    def rhs(t, x):
        x1, x2, th = x
        u  = bang(l3_of_t(t))
        return [math.cos(th), math.sin(th), u]

    sol = solve_ivp(rhs, (t_grid[0], t_grid[-1]), x0,
                    method='Radau', t_eval=t_grid,
                    rtol=RTOL, atol=ATOL)
    x = sol.y.T
    if x.shape[0] < NT:                    # pad if early exit
        pad = np.repeat(x[-1][None, :], NT - x.shape[0], axis=0)
        x   = np.vstack((x, pad))
    return x

# ───── fixed-point iteration between F and B sweeps ─────────────
def sweep_fp(lam_tf: np.ndarray, x0,
             tol: float = 1e-6, kmax: int = 40, 
             ) -> None:
    lam3 = np.full(NT, lam_tf[2])   # = 0 anyway, but explicit
    for _ in range(kmax):
        X_traj[:] = forward(lam3, x0)
        L_new      = backward(lam_tf)
        diff       = np.max(np.abs(L_new[:, 2] - lam3))
        L_traj[:]  = L_new
        lam3       = L_traj[:, 2]
        if diff < tol:
            return
    raise RuntimeError("F–B sweeps did not converge")

# ───── residual ────────────────────────────────────────────────
def residual(z: np.ndarray, alpha: float) -> np.ndarray:
    lam_tf = np.array([z[0], z[1], 0.0])      # λ3(tf)=0 fixed
    tf     = tf_of(z[2])

    global t_grid
    t_grid = np.linspace(0.0, tf, NT)

    x0 = X0_of(alpha)
    sweep_fp(lam_tf, x0)

    x1_f, x2_f, _      = X_traj[-1]
    l1_f, l2_f, _      = L_traj[-1]

    return np.array([
        x1_f,                          # position error x
        x2_f,                          # position error y
        l1_f*l1_f + l2_f*l2_f - 1.0    # costate-norm constraint
    ])

def jac_fd(z: np.ndarray, alpha: float):
    F0 = residual(z, alpha)
    J  = np.zeros((3, 3))
    for k in range(3):
        dz      = z.copy(); dz[k] += FD_EPS
        J[:, k] = (residual(dz, alpha) - F0) / FD_EPS
    return J, np.linalg.norm(F0)

# ───── one Kelley + SER-B step with Armijo ─────────────────────
def kelly_serb(z: np.ndarray, delta: float, alpha: float):
    F0  = residual(z, alpha)
    nF0 = np.linalg.norm(F0)    
    if nF0 < ABS_TOL:        # already good enough
        return z, nF0, delta, True
    J, _ = jac_fd(z, alpha)

    # Newton/Ψtc direction ONCE
    s = np.linalg.solve(J + np.eye(3)/delta, -F0)

    best_z, best_n = z.copy(), nF0
    step = 1.0
    for _ in range(20):                        # Armijo line search
        z_new = z + step * s
        nF    = np.linalg.norm(residual(z_new, alpha))

        if nF < best_n:                        # keep best
            best_z, best_n = z_new, nF

        if nF <= SUCCESS_RATIO * nF0:          # accepted
            return z_new, nF, delta, True

        step *= 0.5                            # shrink step
        if step < 1e-8:
            break                              # give up
    delta = min(delta / step, DELTA_MAX)   # accelerate next Newton call
    return best_z, best_n, delta, False

# ───── driver ──────────────────────────────────────────────────
def solve_dubins():
    z, delta = np.array([-1.0, 0.0, 1.0]), DELTA0
    alpha, dalpha = 0.0, 0.1

    while alpha < 1.0:
        print(f"\nα finished at {alpha:.4f} (Δα under limit)")
        if dalpha < 1e-6:                     # ← new guard
            break
        alpha = min(1.0, alpha + dalpha)
        print(f"\nα = {alpha:.2f}")

        retry = 0
        while True:                                    # attempt this α
            z, nF, delta, ok = kelly_serb(z, delta, alpha)
            print(f"  try {retry:2d}: ‖F‖={nF:.3e}, δ={delta:.1e}, ok={ok}")
            if ok and nF < 1e-9:                       # success
                break
            retry += 1
            if retry >= 4:                             # four strikes…
                dalpha *= 0.5                          # shrink α-step
                if dalpha < 1e-6:             # <-- nothing left to shrink
                    print("\nHomotopy finished: Δα under limit")
                    return z                  # <<< EXIT with current solution

                alpha  -= dalpha                       # roll back
                delta   = DELTA0                       # reset δ
                print(f"    back-off → α={alpha:.2f}, Δα={dalpha:.3f}")
                retry   = 0                            # restart attempts
    return z

# ───── main ────────────────────────────────────────────────────
if __name__ == "__main__":
    z_star = solve_dubins()

    tf_star = tf_of(z_star[2])
    print("\nDubins minimum-time solution")
    print(f"λ1(tf) = {z_star[0]: .6f}")
    print(f"λ2(tf) = {z_star[1]: .6f}")
    print(f"λ3(tf) = 0.0   (fixed)")
    print(f"t_f     = {tf_star:.6f}   (T_max = {T_MAX})")
