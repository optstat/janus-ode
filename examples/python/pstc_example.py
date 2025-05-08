import numpy as np
from numpy.linalg import norm, solve

def F(u, mu):
    """
    The van der Pol 'residual' function:
        F(y, z) = ( z,  mu*(1 - y^2)*z - y ).
    We want to solve F(u) = 0 for a steady state.
    """
    y, z = u
    return np.array([
        z,
        mu*(1 - y**2)*z - y
    ], dtype=float)

def DF(u, mu):
    """
    Jacobian (2x2) of F at u = (y, z):
    
        DF = [ [dF1/dy, dF1/dz],
               [dF2/dy, dF2/dz] ].
               
    F1 = z,
    F2 = mu*(1 - y^2)*z - y.
    
    dF1/dy = 0,    dF1/dz = 1
    dF2/dy = mu*(-2y)*z - 1
    dF2/dz = mu*(1 - y^2)
    """
    y, z = u
    return np.array([
        [0.0,            1.0],
        [mu*(-2.0*y)*z - 1.0,   mu*(1.0 - y**2)]
    ], dtype=float)

def pseudo_transient_continuation(
    u0, mu, delta0=1e-1, 
    method="SER-A", 
    delta_max=np.inf, 
    tol=1e-10, max_iter=200
):
    """
    Pseudo-Transient Continuation for solving F(u) = 0,
    with two possible SER update rules:
        method = 'SER-A' or 'SER-B'.

    Parameters
    ----------
    u0         : initial guess, shape (2,)
    mu         : van der Pol parameter
    delta0     : initial pseudo-time step
    method     : 'SER-A' or 'SER-B'
    delta_max  : cap on delta (can be np.inf)
    tol        : convergence tolerance on ||F(u)||
    max_iter   : maximum number of iterations
    
    Returns
    -------
    u          : final approximate solution
    history    : list of (iter, ||F||, delta) for each iteration
    """
    u = np.array(u0, dtype=float)
    delta = delta0
    
    hist = []
    residual = F(u, mu)
    res_norm = norm(residual, 2)
    
    for k in range(max_iter):
        hist.append((k, res_norm, delta))
        
        if res_norm < tol:
            break
        
        # Construct the matrix M = (1/delta)*I + DF(u)
        J = DF(u, mu)
        M = J.copy()
        M[0,0] += 1.0/delta
        M[1,1] += 1.0/delta
        
        # Solve M * step = F(u)
        # (Note sign: we do u_next = u - M^{-1} F(u))
        step = solve(M, residual)
        
        u_next = u - step
        residual_next = F(u_next, mu)
        res_next_norm = norm(residual_next, 2)
        
        # --- Update delta according to SER-A or SER-B ---
        if method.upper() == "SER-A":
            # SER-A: delta+ = min( delta * ||F(u)|| / ||F(u_next)||, delta_max )
            if res_next_norm != 0:
                new_delta = delta * (res_norm / res_next_norm)
            else:
                # if res_next_norm is 0, we essentially found the solution
                new_delta = delta_max
            delta = min(new_delta, delta_max)
        
        elif method.upper() == "SER-B":
            # SER-B: delta+ = max( delta / ||u_next - u||, delta_max ) in [43]
            # but we usually interpret "max(..., delta_max)" or "min(..., delta_max)" carefully.
            # The reference snippet has:
            #    δ+ = max( δc / ||u+ − uc||, δmax ) 
            # We'll follow that exactly:
            step_size = norm(u_next - u, 2)
            if step_size > 0:
                new_delta = delta / step_size
            else:
                # no movement => presumably at solution
                new_delta = delta_max
            # The snippet in the paper: "δ+ = max(..., δmax)". 
            # Usually we'd do "min(..., δmax)" to cap it, but the text example says "max(..., δmax)".
            # We'll replicate the formula as stated:
            delta = max(new_delta, delta_max)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Prepare next iteration
        u = u_next
        residual = residual_next
        res_norm = res_next_norm
    
    # Final iteration info
    hist.append((k+1, res_norm, delta))
    
    return u, hist

# ---------- DEMO / TEST -------------
if __name__ == "__main__":
    import sys
    
    # We'll demonstrate with mu=1, which has a known steady state at (y,z)=(0,0).
    mu = 1.0
    
    # Let's pick an initial guess somewhat away from the solution:
    u0 = np.array([1.0, 1.0])  # e.g., (y,z) = (1, 1)
    
    print("=== SER-A Demonstration ===")
    sol_A, hist_A = pseudo_transient_continuation(
        u0, mu, delta0=1e-2, method="SER-A", delta_max=1e6, tol=1e-10, max_iter=50
    )
    
    for (k, r, d) in hist_A[:10]:
        print(f"Iter={k:2d}, ||F||={r:.2e}, delta={d:.2e}")
    print("... (omitting further lines) ...")
    print(f"Final: u=({sol_A[0]:.6f}, {sol_A[1]:.6f}), ||F(u)||={norm(F(sol_A, mu)):.2e}")
    
    print("\n=== SER-B Demonstration ===")
    sol_B, hist_B = pseudo_transient_continuation(
        u0, mu, delta0=1e-2, method="SER-B", delta_max=1e3, tol=1e-10, max_iter=50
    )
    
    for (k, r, d) in hist_B[:10]:
        print(f"Iter={k:2d}, ||F||={r:.2e}, delta={d:.2e}")
    print("... (omitting further lines) ...")
    print(f"Final: u=({sol_B[0]:.6f}, {sol_B[1]:.6f}), ||F(u)||={norm(F(sol_B, mu)):.2e}")
