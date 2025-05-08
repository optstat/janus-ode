#!/usr/bin/env python3
# ----------------------------------------------------------------------
# Minimum-time forced Van-der-Pol  –  indirect SSSC with SciPy Radau
# Costates integrated *backwards* (stable manifold)
# ----------------------------------------------------------------------
import time, contextlib, math
import numpy as np
from numpy.linalg import solve, norm, eigvals, cond
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# --------------------------- problem constants -----------------------
mu, u_max, beta_q = 10.0, 2.0, 1e-3
x_init            = np.array([2.0, 0.0])       # initial (=final) state

# ----------------- optimal control law & Hamiltonian -----------------
def control_law(p2, lam):
    u_quad = -(1.0 - lam) * p2 / beta_q
    u_bb   = -u_max * np.sign(p2)
    return np.clip(u_quad + lam * (u_bb - u_quad), -u_max, u_max)

def hamiltonian(x, p, u):
    x1, x2 = x;  p1, p2 = p
    return p1*x2 + p2*(mu*(1 - x1**2)*x2 - x1 + u) + 1.0

# ----------------------- RHS for state & costate ---------------------
def rhs_state(t, x, lam, p2_dummy=0.0):
    x1, x2 = x
    # p2_dummy is only here so rhs_state matches scipy's signature
    u = np.clip(-u_max, u_max, p2_dummy)   # placeholder; u supplied elsewhere
    return [x2, mu*(1 - x1**2)*x2 - x1 + u]

def rhs_aug_state(t, y, lam):
    """Augmented RHS used for forward solve (state only)."""
    x1, x2, p2 = y         # p2 carried along only to re-create u(t) later
    u  = control_law(p2, lam)
    dx1 = x2
    dx2 = mu*(1 - x1**2)*x2 - x1 + u
    dp2 = 0.0              # p2 is frozen here – just book-keeping
    return [dx1, dx2, dp2]

def rhs_costate_back(t, p, lam, x_spl):
    """Backward costate RHS   dp/dt = -∂H/∂x   (stable)."""
    x  = x_spl(t)
    x1, x2   = x
    p1, p2   = p
    dp1 = -p2*(2*mu*x1*x2 + 1)
    dp2 =  p1 + mu*(1 - x1**2)*p2
    return [dp1, dp2]      # already 'minus' sign ⇒ integrate tf→0

# ------------------ forward + backward integration -------------------
def integrate_vdp(z, lam, rtol=1e-8, atol=1e-9):
    """
    z = [p1_tf, p2_tf, tf]
    Return x(tf), p(0), H(tf).
    """
    p1_tf, p2_tf, tf = z

    # ---------- 1. forward state solve: 0 → tf -----------------------
    y0   = np.array([*x_init, p2_tf])        # carry p2(tf) along
    solx = solve_ivp(rhs_aug_state, [0, tf], y0,
                     args=(lam,), method="Radau",
                     rtol=rtol, atol=atol)
    if not solx.success:
        raise RuntimeError("State Radau failed: " + solx.message)

    t_grid = solx.t
    x_grid = solx.y[:2]                      # (2, N) array
    # cubic spline for x(t) so costate RHS has smooth access
    x_spl  = interp1d(t_grid, x_grid, kind="cubic", axis=1,
                      fill_value="extrapolate")

    x_tf = x_grid[:, -1]

    # ---------- 2. backward costate solve: tf → 0 --------------------
    p_tf  = np.array([p1_tf, p2_tf])
    solp  = solve_ivp(rhs_costate_back, [tf, 0], p_tf,
                      args=(lam, x_spl), method="Radau",
                      rtol=rtol, atol=atol)
    if not solp.success:
        raise RuntimeError("Costate Radau failed: " + solp.message)

    p0 = solp.y[:, -1]

    # ---------- 3. Hamiltonian at tf (for free-time residual) --------
    H_tf = hamiltonian(x_tf, p_tf, control_law(p2_tf, lam))
    return x_tf, p0, H_tf

# ---------------------- residual & Jacobian --------------------------
def F(z, lam):
    x_tf, _, H_tf = integrate_vdp(z, lam)
    r       = np.empty(3)
    r[:2]   = x_tf - x_init
    r[ 2]   = H_tf
    return r

def JF(z, lam, rel_eps=1e-6, abs_eps=1e-8):
    J  = np.empty((3, 3))
    f0 = F(z, lam)
    for j in range(3):
        step     = abs_eps + rel_eps*abs(z[j])
        dz       = np.zeros_like(z);  dz[j] = step
        J[:, j]  = (F(z + dz, lam) - f0) / step
    return J

# ---------------------- SER-B corrector (unchanged) ------------------
ρ_in, ρ_out, ε_reg = 0.95, 0.98, 1e-12
def serB_corrector(x0, Gfun, Jfun, beta=0.4, delta0=1e-2,
                   tol=1e-10, maxit=100, rho_in=ρ_in):
    x, Δ = x0.copy(), delta0
    V_old = 0.5*norm(Gfun(x))**2
    inner = 0
    m, I_m = x.size, np.eye(x.size)

    for _ in range(maxit):
        inner += 1
        A = I_m/Δ + Jfun(x) + ε_reg*I_m
        try:
            s = solve(A, -Gfun(x))
        except np.linalg.LinAlgError:
            Δ *= 0.5;  continue

        x_trial = x + s
        with contextlib.suppress(np.linalg.LinAlgError):
            δ = solve(Jfun(x_trial), -Gfun(x_trial))
            δ /= max(1., norm(δ));  x_trial += δ

        V_new = 0.5*norm(Gfun(x_trial))**2
        if V_new > max(5e-2, rho_in*V_old):
            Δ *= 0.5
            if Δ < 1e-14:  return x, False, inner
            continue

        ratio, x, V_old = V_new/V_old, x_trial, V_new
        if V_new < 0.5*tol**2:  return x, True, inner
        Δ = np.clip(Δ*ratio**(beta/2), 1e-14, 1e6)

    return x, False, inner

# ----------------------- SSSC outer continuation ---------------------
def sssc_continuation(z0, *, step0=0.05, step_max=0.5,
                      lam_tol=1e-8, gamma_shift=0.4, n_max=200,
                      verbose=True):
    m, I_m = z0.size, np.eye(z0.size)

    def G (z, lam):  return lam*F(z, lam) + (1 - lam)*z
    def JG(z, lam):  return lam*JF(z, lam) + (1 - lam)*I_m

    lam, step, z = 0.0, step0, z0.copy()
    outer_it = inner_it = rejected = 0
    t0 = time.perf_counter()

    while lam < 1 - lam_tol and n_max:
        outer_it += 1
        lam_new = min(1.0, lam + step)
        ρ       = max(abs(eigvals(JG(z, lam_new))))
        α       = gamma_shift*(ρ + 1)*math.sqrt(max(0.0, 1 - lam))

        if verbose:
            print(f"[outer {outer_it:03d}] call SER-B  α={α:.3e}  "
                  f"λ={lam:.3f}→{lam_new:.3f}  ‖F‖={norm(F(z,1)):.3e}")

        Gs  = lambda zz: G(zz, lam_new) + α*(zz - z)
        JGs = lambda zz: JG(zz, lam_new) + α*I_m
        z_new, ok, inner_now = serB_corrector(z, Gs, JGs)

        if verbose:
            print(f"[outer {outer_it:03d}] SER-B {'ok' if ok else 'fail'}  "
                  f"inner={inner_now:02d}  ‖F‖={norm(F(z_new,1)):.3e}")

        inner_it += inner_now

        if (not ok or
            norm(F(z_new,1)) > ρ_out*norm(F(z,1)) + 1e-12 or
            0.5*norm(G(z_new,lam_new))**2 >
            max(1e-6, ρ_out*0.5*norm(G(z,lam))**2)):
            step *= 0.5;  rejected += 1
            if step < 1e-15:  raise RuntimeError("Δλ too small");  continue

        lam, z, n_max = lam_new, z_new, n_max-1
        ρ   = max(abs(eigvals(JG(z, lam))))
        α   = gamma_shift*(ρ + 1)
        condJ = cond(JG(z, lam) + α*I_m)

        if verbose:
            print(f"λ={lam:5.3f}  ‖F‖={norm(F(z,1)):8.2e} "
                  f"ρ={ρ:8.2e}  cond={condJ:9.2e}")

        if   condJ < 5e2:  step = min(step_max, step*1.8)
        elif condJ < 2e3:  step = min(step_max, step*1.2)

    wall = time.perf_counter() - t0
    meta = dict(outer=outer_it, inner=inner_it, rejected=rejected,
                time=wall, F=norm(F(z,1)))
    return z, meta

# -------------------------------- main --------------------------------
if __name__ == "__main__":
    z0 = np.array([0.1, 0.1, 6.0])          # guess for (p1_tf, p2_tf, tf)

    root, meta = sssc_continuation(z0, verbose=True)
    p_tf, tf = root[:2], root[2]

    print("\n==== solution ===========================================")
    print(f"  p1(tf) = {p_tf[0]: .6f}")
    print(f"  p2(tf) = {p_tf[1]: .6f}")
    print(f"  t_f    = {tf: .6f}  s")
    print("residual ‖F‖ =", f"{meta['F']:.2e}")
    print("outer / inner / rejected =", meta['outer'],
          meta['inner'], meta['rejected'])
    print("wall-time %.3f s" % meta['time'])
