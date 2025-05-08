import numpy as np
from scipy.integrate import solve_ivp

mu_d = 10000.0

def vanderpol(t, y, mu=mu_d):
    """
    Van der Pol ODE:
        y = [x1, x2]
        dy/dt = [ x2,
                  (1 - x1^2)*x2 - x1 ]
    """
    x1, x2 = y
    dx1dt = x2
    dx2dt = mu*(1.0 - x1**2)*x2 - x1
    return [dx1dt, dx2dt]

def outer_ode(tau, T, mu=mu_d):
    """
    Outer ODE for pseudo-time approach:
        dT/dtau = - x2( T )
    where x2(T) is obtained by integrating Van der Pol from t=0..T.
    """
    Tval = T  # T is stored as an array of length 1
    # Integrate Van der Pol from 0..Tval with initial conditions x(0) = (0, 2).
    # Note: must ensure Tval >= 0 for the solver, or handle negative carefully.
    if Tval < 0:
        # In a real application, you might clip or treat negative T specially.
        return [0.0]

    sol = solve_ivp(
        fun=lambda t, y: vanderpol(t, y, mu),
        t_span=[0, Tval],
        y0=[0, 2],
        method='Radau'
    )
    print(f"Van der Pol solution from 0 to {Tval} = {sol}")
    # x2(Tval) is sol.y[1, -1]
    x2_T = sol.y[1, -1] if sol.t[-1] == Tval else np.nan  # check if success
    
    # dT/dtau = - x2(Tval)
    return [x2_T]

def solve_pseudotransient(mu=mu_d, T0=1.0, tau_max=10.0):
    """
    Solve the outer ODE from tau=0..tau_max with T(0)=T0.
    Returns the final T at 'end' of pseudo-time.
    """
    # Wrap outer ODE in a function for solve_ivp
    def f(tau, T):
        return outer_ode(tau, T, mu)

    # Solve with a simple method on [0..tau_max]
    sol = solve_ivp(
        f,
        [0, tau_max],      # pseudo-time interval
        [T0],              # initial T
        dense_output=True, # so we can sample
        method='Radau'
    )
    print(f"First solution from pseudo-transient approach = {sol}")
    # Final T
    T_final = sol.y[0, -1]
    return T_final, sol

if __name__ == "__main__":
    # Example usage
    T_final, outer_sol = solve_pseudotransient(mu=mu_d, T0=0.0, tau_max=50)
    print(f"Final T from pseudo-transient approach = {T_final:.6f}")
    #Check the solution by integrating the vdp oscillator
    # from 0 to T_final
    sol = solve_ivp(
        fun=lambda t, y: vanderpol(t, y, mu=mu_d),
        t_span=[0, T_final],
        y0=[0, 2]
    )
    print(f"Van der Pol solution from 0 to {T_final} = {sol}")