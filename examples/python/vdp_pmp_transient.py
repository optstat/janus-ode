import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


class VDPPMPSolver:
    """
    Skeleton class to demonstrate:
      - Forward integration of state variables
      - Backward integration of costates
      - Bang-bang control rule (separate function, not used in first guess)
      - The Hamiltonian in this problem is as follows:
        H = p1*x2 + p2*u*((1-x1^2)*x2 - x1)  + 1
    """
    def __init__(self, x0, pT, t0, tF, umax, umin):
        """
        Initialize the solver with:
          x0   : initial state [x1(0), x2(0), x3(0)]
          pT   : costate values at final time [p1(T), p2(T), p3(T)]
          t0   : initial time
          tF   : final time
          umax : maximum control value
          umin : minimum control value

        We'll store a naive guess of the control (u = umax) for all t.
        """
        self.x0   = np.array(x0, dtype=float)
        self.pF   = np.array(pT, dtype=float)
        self.t0   = t0
        self.tF   = tF
        self.umax = umax
        self.umin = umin

        # For our naive first guess, just set u(t) = umax everywhere.
        # (You could store a piecewise or time-discretized guess if desired.)
        self.u_guess = umax

        # Containers for solutions (filled after calling solve_forward/solve_backward).
        self.sol_x = None
        self.sol_p = None
        self.run_count = 0
        self.W = 0.01
        

    def first_guess_control(self, t):
        """
        Naive first guess for the control: u = umax for all t.
        """
        return self.u_guess

    def f_state(self, t, x, u):
        """
        Right-hand side of the state ODEs:
          x1' = x2
          x2' = u * ((1 - x1^2)*x2 - x1)
          x3' = 1.0
          since x3 = t
        """
        x1, x2, x3 = x
        dx1 = x2
        dx2 = u * ((1 - x1**2)*x2 - x1)
        dx3 = 1.0
        return [dx1, dx2, dx3]

    def f_costate(self, t, p, x, u):
        """
        H = p1*x2 + p2*u*((1-x1^2)*x2 - x1) + 1
        Right-hand side of the costate ODEs:
          p1' = -p2 * u * (-2*x1*x2 - 1)
          p2' = -p1 - p2 * u * (1 - x1^2)
          p3' = 0.0
        """
        p1, p2, p3 = p
        x1, x2, x3 = x
        dp1 = -p2 * u * (-2*x1*x2 - 1)
        dp2 = -p1 - p2 * u * (1 - x1**2)
        dp3 = 0.0
        return [dp1, dp2, dp3]

    def bang_bang_control(self, x, p):
        """
        Bang-bang control rule:
          if p2 * (1 - x1^2) * x2 - x1 < 0  ==>  u = umax
          else                             ==>  u = umin
        """
        x1, x2, _ = x
        _, p2, _  = p
        check = p2 * ((1 - x1**2) * x2 - x1)
        if check < 0.0:
            return self.umax
        else:
            return self.umin

    def solve_forward(self, num_points=10):
        """
        Solve the state equations forward in time with a *fixed* guessed control (u_guess).
        This is just a naive first pass to get an initial guess of x(t).
        """
        def ode_forward(t, x):
            if self.run_count == 0:
                self.u_guess = self.first_guess_control(t)
            else:
                p_interp = interp1d(
                          self.sol_p['t'], self.sol_p['p'], kind='cubic',
                          fill_value="extrapolate")

                self.u_guess = self.bang_bang_control(x, p_interp(t))
            return self.f_state(t, x, self.u_guess)
        t_span = (self.t0, self.tF)
        t_eval = np.linspace(self.t0, self.tF, num_points)
        

        sol = solve_ivp(
            ode_forward,
            t_span,
            y0=self.x0,
            t_eval=t_eval,
            method='Radau',
            atol = 1.0e-8,
            rtol = 1.0e-4
        )

        self.sol_x = sol
        self.run_count += 1

        return sol

    def solve_backward(self, num_points=10):
        """
        Solve the costate equations backward in time, again using the same naive control guess (u_guess).
        We'll integrate from tF down to t0.
        """
        # First, make sure we have the forward solution for x(t).
        if self.sol_x is None:
            raise ValueError("State solution not found. Call solve_forward first.")

        # Interpolate x(t) from the forward solution, so we can evaluate x at any t.
        x_interp = interp1d(
            self.sol_x.t, self.sol_x.y, kind='cubic',
            fill_value="extrapolate"
        )

        # We want dp/dt from tF down to t0, so either:
        #   (1) Integrate forward in an auxiliary variable tau = tF - t, or
        #   (2) Flip the sign of the ODE and integrate from t0 to tF.
        # Here we'll do (2) for simplicity: define dp/dtau = -dp/dt, then integrate from tau=0 to tau=(tF - t0).
        
        def ode_backward(tau, p):
            # t runs backward: t = tF - tau
            t = self.tF - tau
            x = x_interp(t)  # x(t)
            u = self.bang_bang_control(x, p)
            # dp/dt
            dpdt = self.f_costate(t, p, x, u)
            # dp/dtau = - dp/dt
            return [dpdt[0], dpdt[1], dpdt[2]]

        # "Initial" condition at tau = 0 (which corresponds to t = tF):
        p_init = self.pF

        # We'll integrate up to tau = (tF - t0), which corresponds to t = t0.
        tau_span = (0, self.tF - self.t0)
        tau_eval = np.linspace(0, self.tF - self.t0, num_points)

        sol = solve_ivp(
            ode_backward,
            tau_span,
            y0=p_init,
            t_eval=tau_eval,
            method='Radau',
            atol = 1.0e-8,
            rtol = 1.0e-4
        )

        
        # Build a small dict-like object to store final result:
        self.sol_p = {
            't': sol.t,
            'p': sol.y
        }
        return self.sol_p

    def solve_all(self):
        """
        Convenience method to run forward solve for x and then backward solve for p.
        """
        self.solve_forward()
        print(self.sol_x)
        self.solve_backward()
        print(self.sol_p)
        

#Function takes as input the fixed parameters and retuns the variable parameters
#For this problem
#We have for the Hamiltoninian
#H(x,p,u) = p1*x2 + p2*u*((1-x1^2)*x2 - x1) +1
#We have tf as inputs
#and we return x1f, x2f, x3f, p10, p20, p30, Hf as outputs
#Note the solver retains the state from previous runs so we can use the previous solution to solve the next one
def F(x10, x20, x30, p1f, p2f, p3f, tf, solver):
    """
    Wrapper function to call the solver and return the solutions.
    """
    #The solver has state from previous runs so it contains the previous solution
    #Update the solver state while keeping the previous solution for interpolation
    solver.x0 = [x10, x20, x30]
    solver.pF = [p1f, p2f, p3f]
    solver.tF = tf
    solver.solve_all()
    x1f = solver.sol_x.y[0, -1]
    x2f = solver.sol_x.y[1, -1]
    x3f = solver.sol_x.y[2, -1]
    p10 = solver.sol_p['p'][0, -1]
    p20 = solver.sol_p['p'][1, -1]
    p30 = solver.sol_p['p'][2, -1]
    p1factual = solver.sol_p['p'][0, 0]
    p2factual = solver.sol_p['p'][1, 0]
    p3factual = solver.sol_p['p'][2, 0]
    print(f'tf = {tf}')
    print(f'p1f = {p1f}, p1factual = {p1factual}')
    print(f'p2factual = {p2factual}, p2f = {p2f}')
    print(f'p3factual = {p3factual}, p3f = {p3f}')
    print(f'p10 = {p10}, p20 = {p20}, p30 = {p30}')
    assert np.isclose(p1f, p1factual)
    assert np.isclose(p2f, p2factual)
    assert np.isclose(p3f, p3factual)
    uactual = solver.bang_bang_control([x1f, x2f, x3f], [p1factual, p2factual, p3factual])
    Hf = p1factual*x2f + p2factual*uactual*((1 - x1f**2)*x2f - x1f) + 1
    return x1f, x2f, x3f, p10, p20, p30, Hf 


def pseudo_transient_relaxation():
    #First solve the forward and backward problems imposing only the condition x1f = 0.0
    #Leaving the rest of the conditions free
    #Set up the outer ODE
    x10 = 1.0
    x20 = 1.0
    x30 = 0.0 #Initial time
    p2f = 0.0
    p3f = 1.0 #The costate is here is the time costate which is 1.0
    def outer_dyns(t, y, solver):
        p1f = y[0]
        tf  = y[1]
        
        #We would like to solve for dx/dtau = M^{-1}*F(x)
        #Here M is the Sensitivity Jacobian so we have to solve for the 
        #System M*dx/dtau = F(x) using LU decomposition.  
        #This is an expensive operation but for this system we can solve it directly 
        #We will use Finite Difference to estimate M and LU decomposition to solve for dx/dtau
        #There are two variables to solve for x1f and tf and two constraints
        x1f, x2f, x3f, p10, p20, p30, Hf = F(x10, x20, x30, p1f, p2f, p3f, tf, solver)
        #Estimate M using central finite difference
        h = 1.0e-10*np.fmax(1.0, np.abs(p1f))
        x1f_ph, x2f_ph, x3f_ph, p10_ph, p20_ph, p30_ph, Hf_ph = F(x10, x20, x30, p1f+h, p2f, p3f, tf, solver)
        x1f_mh, x2f_mh, x3f_mh, p10_mh, p20_mh, p30_mh, Hf_mh = F(x10, x20, x30, p1f-h, p2f, p3f, tf, solver)
        M11 = (x1f_ph - x1f_mh)/(2*h)
        M21 = (Hf_ph - Hf_mh)/(2*h)
        h = 1.0e-10*np.fmax(1.0, np.abs(tf))   
        #Now vary the second input tf
        x1f_ph, x2f_ph, x3f_ph, p10_ph, p20_ph, p30_ph, Hf_ph = F(x10, x20, x30, p1f, p2f, p3f, tf+h, solver)
        x1f_mh, x2f_mh, x3f_mh, p10_mh, p20_mh, p30_mh, Hf_mh = F(x10, x20, x30, p1f, p2f, p3f, tf-h, solver)
        M12  = (x1f_ph - x1f_mh)/(2*h)
        M22 = (Hf_ph - Hf_mh)/(2*h)
        #Now we have the Sensitivity Jacobian M
        #Now we have to solve the system M*dx/dtau = F(x)
        #Construct the Matrix M and do LU decomposition
        M = np.array([[M11, M12], [M21, M22]])
        print(f'M = {M}')
        print(f'p1f = {p1f}, tf = {tf}')
        Fx = np.array([x1f, Hf])
        #Solve for dx/dtau
        alpha = 1.0
        dx = np.linalg.solve(M, Fx)


        print(f'dx = {dx}')
        return -alpha*dx

    solver = VDPPMPSolver([x10, x20, x30], [0.0, p2f, p3f], 0.0, 0.1, 3.0, 1.0)
    y0 = [0.0, 1.0] 
    t_span = (0, 10)
    sol = solve_ivp(
        #Add tolerances to make the solution more accurate
        outer_dyns,
        t_span,
        y0=y0,
        args=(solver,),
        method='Radau',
        atol=1.0e-8,
        rtol=1.0e-4
    )

# -----------------------------------------------------------------------------
# Example usage (replace with your own in a separate script or notebook):
#
if __name__ == "__main__":
    pseudo_transient_relaxation()
