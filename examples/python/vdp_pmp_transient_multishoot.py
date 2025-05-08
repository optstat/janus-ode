import numpy as np
import os
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
import contextlib

sys.stdout.reconfigure(line_buffering=True)  # Python 3.7+





class VDPPMPSolver:
    r"""
    Skeleton class to demonstrate:
      - Solution of a minimum time problem using the Pontryagin Minimum Principle using
      - multiple shooting, homotopy and augmented langrange methods
      - Bang-bang control rule (separate function, not used in first guess)
      - The Hamiltonian in this two dimensional problem is as follows:
        H = p1*x2 + p2*u*((1-x1^2)*x2 - x1)  + 1.0 +W/2*u^2
        with terminal condtions 
        \Phi(x_f) = (x1f-x10)^2 + (x2f-x20)^2
        In other words we want to complete a full cycle of the VDP oscillator in minimum time
        The problem is interesting because it is highly nonlinear, and has only a small set of
        reachable sets
        The problem has the following original and transversality conditions
        x10 is given
        x20 is given
        p1f is free
        p2f is free
        #In each iteration mu is fixed and lambda_1 and lambda_2 are to be determined
        #by the pseudo-transient continuation method
        There are three constraints and three unknowns
        The three unknowns are
          p1f, p2f, fT
        The three constraints are
          x1f=x10
          x2f=x20 
          Hf=0.0 
        The dynamics are
        \dot{x1} = x2
        \dot{x2} = u*((1-x1^2)*x2 - x1)
        The costate equations are using 
        H = p1*x2 + p2*u*((1-x1^2)*x2 - x1)  + 1.0 +W/2*u^2
        \dot{p1} = p2*u*(2*x1*x2 + 1) 
        \dot{p2} = -p1 - p2*u*(1-x1^2) 
        We therefore have a system of three equations in three unknowns
        x2(T)=x20, x1(T)=x10, H(T)=0 are the constraints
        p1(T), x4(0), T are the unknowns
        So we have a 2D BVP system as defined by the PMP
        The optimal control is given by the bang-bang control rule
        
        Bounded by umin and umax

        The numerical method is as follows
      - Stabilization of the stiffness using multiple shooting
      - The segments proceed from x0i to xfi and pfi to p0i in reverse
      - x0i--------------------->xfi x0_{i+1}---------------------->xf_{i+1}
                                  |                                    |
                                  |                                    |
        p0i<---------------------pfi p0_{i+1}<----------------------pf_{i+1}
        - The endpoints of final segments can be determined by the Pontryagin Minimum Principle
        - The initial state variables are fixed
        - The endpoints of the intermediate segments are determined by the continuity of the costates
        - and the state variables
        - We will use the representation [Ns, Nt, d] for the state and costate variables
        - The class uses a time marching scheme followed by a Pseudo-Transient Continuation Method
        - as the inner loop with umin and umax as the parameters being varied in the upper loop 
    """
    def __init__(self, 
                 x0, # Initial state for the first segment
                 pF, # Final costate for the last segment (estimate)
                 Ns,
                 Nt,
                 umin,  # Minimum control value
                 umax,   # Maximum control value
                 tf=1.0,
                 alpha=0.1 #Smoothing parameter for the control
                ): # Number of time points in each segment
        """
        Initialize the solver with:
          x0s  : initial states for each segment [x1(0), x2(0)] except for the first segment
          pFs  : Final costates values at each segment [p1(T), p2(T)] except for the last segment
          ts   : Time sequence for each segment where the solution will be evaluated
          umax : maximum control value
          umin : minimum control value
        """
        #Make sure the dimension of the inputs is 2D
        self.Ns      = Ns# Number of segments
        self.d       = x0.shape[0] # Dimension of the state and costate variables
        self.Nt      = Nt # Number of time points in each segment
        #The same for forward and backward sweep
        #Create a linear structure for the first guess
        self.run_count = 0
        self.atol      = 1.0e-5
        self.rtol      = 1.0e-3
        self.sol_x     = np.zeros((self.Ns, self.Nt, self.d)) #State solution
        self.sol_p     = np.zeros((self.Ns, self.Nt, self.d)) #Costate solution
        self.sol_t     = np.zeros((self.Ns, self.Nt)) #Time discretization for each segment
        self.con_x     = np.zeros((self.Ns-1, self.d)) #State segment deltas
        self.con_p     = np.zeros((self.Ns-1, self.d)) #Costate segment deltas
        self.con_pmp   = np.zeros((3,)) #Endpoint constraints. There are three variables p1f, p2f and tF and three constraints
        #Create a guess of linear interpolation for the state and costate
        # Construct the solution with what we know
        self.sol_x[0, 0, 0]  = x0[0]  # Initial state for the first segment
        self.sol_x[0, 0, 1]  = x0[1]  # Initial state for the first segment  
        self.sol_p[-1,-1, 0] = pF[0]  # Final costate for the last segment
        self.sol_p[-1,-1, 1] = pF[1]
        self.sol_x_ref = np.zeros((self.Ns*self.Nt, self.d)) #Reference state solution
        self.sol_t_ref = np.zeros((self.Ns*self.Nt))  # Time discretization for each segment
        self.ref_interp = None
        self.umin            = umin
        self.umax            = umax
        self.x0              = x0
        self.count           = 0
        self.t0              = 0.0 # Start time for the first segment
        self.tF              = tf
        self.alpha           = alpha  # Smoothing parameter for the control
        self.stop1           = False  # Flags used for debugging
        self.stop2           = False
        self.W               = 10.0
        self.p1f            = 0.0
        self.p2f            = 0.0
        #Change this to be your data directory
        home_directory       = os.path.expanduser("~")
        self.data_dir        = os.path.abspath(home_directory+"/Applications/research/janus-ode/examples/python/data/")



    ## Initialize the variables for the system
    ## x0s are the initial states for the intermediate segments
    ## pFs are the final costates for the intermediate segments
    def init_variables(self, x0s, pFs, p1F, p2F, tF):
        self.sol_x[1:, 0, :]   = x0s  # Initial states for the intermediate segments from 1 onwards
        #We have to be careful with the time state variable x3
        self.sol_p[:-1, -1, :] = pFs  # Final costates for the intermediate segments excluding last segment
        self.sol_p[-1, -1, 0]  = p1F
        self.sol_p[-1, -1, 1]  = p2F
        self.p2f = p2F
        self.p1f = p1F
        #We need to adjust the time for all segments
        self.sol_t       = np.zeros((self.Ns, self.Nt))  # Time discretization for each segment
        self.tF          = tF
        

        self.t0      = 0.0 # Start time for the first segment
        
        dt = (self.tF - self.t0) / self.Ns  # Time step for each segment
        for i in range(self.Ns):
            self.sol_t[i, :] = np.linspace(self.t0 + i*dt, \
                                           self.t0 + (i+1)*dt, \
                                           self.Nt)
        
        #Generate the reference trajectory for the state only
        print(f'x0s   = {x0s}')
        print(f'pFs   = {pFs}')
        print(f'sol_x = {self.sol_x}')
        print(f'sol_p = {self.sol_p}')
        print(f'sol_t = {self.sol_t}')
        

    def trajectory_difference(self, x, y, norm_type='infinity'):
        """
        Computes the difference between two trajectories x and y in R^d,
        sampled at N+1 discrete time points.

        Parameters
        ----------
        x : np.ndarray, shape (N+1, d)
            First trajectory. x[k] is the state (dimension d) at the k-th time step.
        y : np.ndarray, shape (N+1, d)
            Second trajectory. y[k] is the state (dimension d) at the k-th time step.
            norm_type : str, optional
            Which norm to use. Options:
            - 'infinity': discrete sup norm (max over time of Euclidean distance)
            - 'l2': discrete L^2 norm over time
            - 'l1': discrete L^1 norm over time

        Returns
        -------
        float
            Scalar measure of the difference between x and y according to the chosen norm.
        """
        # Ensure x and y have the same shape
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape.")
        diffs = np.linalg.norm(x - y, axis=1)  # shape (N,) for N time points
        if norm_type == 'infinity':
            return np.max(diffs)
        elif norm_type == 'l2':
            return np.sqrt(np.sum(diffs**2))
        elif norm_type == 'l1':
            return np.sum(diffs)
        else:
            raise ValueError("Invalid norm_type. Choose 'infinity', 'l2', or 'l1'.")
        

    def f_state(self, t, x, u):
        """
        Right-hand side of the state ODEs:
          x1' = x2
          x2' = u * ((1 - x1^2)*x2 - x1)
          x3' = 1.0
          since x3 = t
        """
        x1, x2 = x
        dx1 = x2
        dx2 = u * ((1 - x1**2)*x2 - x1)
        return [dx1, dx2]

    def f_costate(self, t, p, x, u):
        """
        H = p1*x2 + p2*u*((1-x1^2)*x2 - x1) + 1.0 
        Right-hand side of the costate ODEs:
          p1' = -p2 * u * (-2*x1*x2 - 1)
          p2' = -p1 - p2 * u * (1 - x1^2)
          p3' = 0.0
        """
        p1, p2   = p
        x1, x2   = x
        dp1 = -p2 * u * (-2*x1*x2 - 1) 
        dp2 = -p1 - p2 * u * (1 - x1**2) 
        return [dp1, dp2]

    def calc_control(self, t, x, p):
        """
        Calculate the optimal control u* using the Pontryagin Minimum Principle.
        """
        x1,  x2 = x
        p1,  p2 = p
        #H = p1*x2 + p2*u*((1-x1^2)*x2 - x1) + 1.0 + W/2*u^2
        u_star     = -p2 * ((1 - x1**2) * x2 - x1)/self.W 
        u_star     = np.clip(u_star, self.umin, self.umax)  # Ensure u_star is within bounds
        return u_star


    def solve_forward(self, i : int):
        """
        Solve the state equations forward in time for segment i
        """
        p_interp = interp1d(
                            self.sol_t[i, :], \
                            self.sol_p[i, :, :].T, \
                            kind='cubic', \
                            fill_value="extrapolate")

        def ode_forward(t, x):
            u_star = self.calc_control(t, x, p_interp(t))       
            return self.f_state(t, x, u_star)
        t0 = self.sol_t[i, 0] #Start time for the segment
        tF = self.sol_t[i, -1] #End time for the segment
        t_span = (t0, tF) #There is no explicit dependence on time in the dynamics
        t_eval = np.linspace(t0, tF, self.Nt) #Evaluate the solution at these points
        x0 = self.sol_x[i, 0, :] #Initial state for the segment
        sol = solve_ivp(
                ode_forward,
                t_span,
                y0=x0,
                t_eval=t_eval,
                method='Radau',
                atol = self.atol,
                rtol = self.rtol
            )
        self.sol_x[i, :, :] = sol.y.T #Store the entire solution for the segment
        if i == self.Ns - 1: #We completed a whole run across all segments
            self.run_count += 1

    def solve_backward(self, i:int):
        """
        Solve the costate equations backward in time, 
        again using the same naive control guess (u_guess).
        We'll integrate from tF down to t0 by inverting the time variable.
        """
        # First, make sure we have the forward solution for x(t).
        if self.sol_x is None:
            raise ValueError("State solution not found. Call solve_forward first.")
        #Interpolate x(t) from the forward solution, so we can evaluate x at any t.
        #Invert the time so that we can integrate from tF to t0 for the segment
        sol_x_segm = self.sol_x[i, :, :]
        x_interp = interp1d(
                            self.sol_t[i, :], \
                            sol_x_segm.T, \
                            kind='cubic',\
                            fill_value="extrapolate"
                            )
        #We integrate time from tF to t0 while the dynamics are integrated backwards
        t0 = self.sol_t[i, 0] #Start time for the segment
        tF = self.sol_t[i, -1] #End time for the segment
        t_span = (tF, t0) #We are integrating backwards in time
        t_eval = np.linspace(tF, t0, self.Nt)
            
        def ode_costate(t, p):
            # t runs "forward" from tF to t0 with costate dynamics as defined by -partial H/partial x
            x = x_interp(t)  
            u = self.calc_control(t, x, p)
            #print(f'costate t = {t}, x = {x}, p = {p}, u = {u}')
            # dp/dt
            dpdt = self.f_costate(t, p, x, u)
            return [dpdt[0], dpdt[1]]
        #start at the end
        pF = self.sol_p[i, -1, :] #Final costate for the segment
        sol = solve_ivp(
                ode_costate,
                t_span,
                y0=pF,
                t_eval=t_eval,
                method='Radau',
                atol = self.atol,
                rtol = self.rtol)
        self.sol_p[i, :, :] = sol.y[:, ::-1].T  #Reverse and transpose the solution to match the time order

    def solve_segment(self, i:int):
        max_iterations = 100  
        for _ in range(max_iterations):
            x_old = np.copy(self.sol_x[i, :, :])
            p_old = np.copy(self.sol_p[i, :, :])

            # Forward/backward
            self.solve_backward(i)
            self.solve_forward(i)

            xdiff = self.trajectory_difference(self.sol_x[i, :, :], x_old)
            pdiff = self.trajectory_difference(self.sol_p[i, :, :], p_old)
            
            if xdiff < 1.0e-8 and pdiff < 1.0e-8:
                break
            



    def solve_all(self):
        """
        Solve the forward and backward problems for all segments
        """
        for i in range(self.Ns):
            self.solve_segment(i)

        
        #Also calculate the constraints violations.  This is very simple
        for i in range(self.Ns-1):
            self.con_x[i, :] = self.sol_x[i, -1, :]-self.sol_x[i+1, 0, :] 
            self.con_p[i, :] = self.sol_p[i, -1, :]-self.sol_p[i+1, 0, :] 
        
        
        p1f = self.sol_p[-1, -1, 0]
        p2f = self.sol_p[-1, -1, 1]
        
        #The expected final condition is the same as the initial condition
        xf  = self.sol_x[-1, -1, :]
        x1f = self.sol_x[-1, -1, 0]
        x2f = self.sol_x[-1, -1, 1]

        pf = self.sol_p[-1, -1, :]
         
        ustar = self.calc_control(self.tF, xf, pf)
        #H=p1f*x2f + p2f*ustar*((1-x1f**2)*x2f - x1f) + 1.0
        self.Hf = p1f*x2f + p2f*ustar*((1-x1f*x1f)*x2f - x1f) + 1.0 
        #Now calculate the constraints
        self.con_pmp[0] = self.sol_x[-1, -1, 0]-self.x0[0] #x1f
        self.con_pmp[1] = self.sol_x[-1, -1, 1]-self.x0[1] #x2f 
        self.con_pmp[2] = self.Hf #Hamiltonian at the final time which should be equal to 0.0
    

    def finite_diff_step(self, x, rel_power=1.0/3.0, abs_floor=1e-8):
        """
        Returns a step size for finite difference on 'x' that
        balances relative scale and a minimum absolute floor.
        """
        eps = np.finfo(float).eps  # ~2.220446049250313e-16
        return max((eps**rel_power)*abs(x), abs_floor)        

    #This function is the TPBVP formulation that takes as input
    #    p1f, p2f, tF
    #    It returns the differences between the segments endpoints
    #    and the differences of the costates between segments
    #    as well as the transversality condition at the final time
    def F(self, x0s, pFs, p1f, p2f, tF):
        self.init_variables(x0s, pFs, p1f, p2f, tF)  # Set the initial conditions for the segments
        self.solve_all()
        #Concatenate the constraints into a single array
        con_x = self.con_x.flatten()  # Flatten the state segment deltas
        con_p = self.con_p.flatten()  # Flatten the costate segment deltas
        con_pmp = self.con_pmp  # Endpoint constraints
        # Combine all constraints into a single array
        constraints = np.concatenate((con_x, con_p, con_pmp))
        return constraints

    def extract_variables(self, xs):
        x0s = xs[0:(self.Ns-1)*self.d].reshape((self.Ns-1, self.d))  # Initial states for Intermediate segments
        pFs = xs[(self.Ns-1)*self.d:(self.Ns-1)*self.d + (self.Ns-1)*self.d].reshape((self.Ns-1, self.d))  # Final costates for Intermediate segments
        p1f = xs[-3]  
        p2f = xs[-2]
        tF = np.log1p(np.exp(xs[-1]))  
        assert tF > 0.0, "Final time must be positive"
        return x0s, pFs, p1f, p2f, tF



        
    def compute_jacobian(self, xs):
        """
        Builds the Jacobian dF/dx with finite differences,
        exactly as in your original pseudo_dyns method,
        but now we only do one step (not integrate in pseudo-time).
        """
        # Number of constraints:
        MN = 2*(self.Ns-1)*self.d + 3
        M  = np.zeros((MN, MN))
            # ----- snapshot current solver state -----
        sol_x0 = self.sol_x.copy()
        sol_p0 = self.sol_p.copy()
        sol_t0 = self.sol_t.copy()

        x0, pF,  p1, p2, t = self.extract_variables(xs)
        base_con = self.F(x0, pF, p1, p2, t)
        print(f'base_con = {base_con}')

        # Finite differences:
        for i in range(MN):
            xsp = np.copy(xs)
            # step size
            h = self.finite_diff_step(xsp[i])
            xsp[i] += h
            x0sp, pFsp, p1pF, p2pF, tpF = self.extract_variables(xsp)
            if i == MN -3:
                self.stop1 = True
            conp = self.F(x0sp, pFsp, p1pF, p2pF, tpF)
            if i == MN -3:
                sol_xc = np.copy(self.sol_x)
                sol_pc = np.copy(self.sol_p)


            xsm = np.copy(xs)
            xsm[i] -= h
            x0sm, pFsm, p1mF, p2mF, tmF = self.extract_variables(xsm)
            conm = self.F(x0sm, pFsm, p1mF, p2mF, tmF)
            if i == MN -3:
                print(f'sol_x before = {sol_xc}')
                print(f'sol_p before = {sol_pc}')
                print(f'sol_x after = {self.sol_x}')
                print(f'sol_p after = {self.sol_p}')
            

            M[:, i] = (conp - conm)/(2*h)
        # ----- restore snapshot -----
        self.sol_x[:] = sol_x0
        self.sol_p[:] = sol_p0
        self.sol_t[:] = sol_t0
        return M, base_con

    # ------------------------------------------------------------------
    #  Augmented‑Lagrangian pre‑solve (à la LANCELOT) to estimate p1_f, p2_f
    # ------------------------------------------------------------------
    def estimate_constraints_auglag(
        self,
        tF_guess: float,
        p_init=np.array([0.0, 0.0]),
        lambda_init=None,
        mu_init: float = 1.0,
        max_outer: int = 8,
        tol: float = 1.0e-6,
    ):
        """Return an approximate value for intermediate values for x and p
        as well as the terminal costates p1_f, p2_f that satisfies the PMP
        endpoint constraints using an Augmented Lagrangian strategy
        similar to the *outer* loop of LANCELOT.

        Only the three endpoint equalities are enforced:
            x1(T) = x1(0),  x2(T) = x2(0),  H(T) = 0.
        (Segment‑continuity constraints are *not* imposed – SER‑B will
        handle those.)
        This will estimate the solution using single shooting
        and the Pontryagin Minimum Principle.
        """
        if lambda_init is None:
            lam = np.zeros(3)
        else:
            lam = np.asarray(lambda_init, dtype=float)
        mu = float(mu_init)
        p = np.asarray(p_init, dtype=float).copy()

        # helper: evaluate endpoint constraints given p = [p1, p2]
        def constraint_vector(p_vec: np.ndarray) -> np.ndarray:
            p1, p2 = p_vec
            # zero intermediate segments => x0s & pFs empty
            empty = np.empty((0, self.d))
            self.F(self.x0s, self.pFs, p1, p2, tF_guess)
            return self.con_pmp.copy()  # shape (3,)

        for _ in range(max_outer):
            # (1) unconstrained sub‑problem minimisation
            def inner_obj(p_vec: np.ndarray) -> float:
                c = constraint_vector(p_vec)
                phi = lam + mu * c
                return 0.5 * np.dot(phi, phi) / mu

            res = minimize(inner_obj, p, method="L-BFGS-B")
            p = res.x

            # (2) update multipliers and penalty
            c = constraint_vector(p)
            if np.linalg.norm(c, 2) < tol:
                break
            lam += mu * c
            mu *= 10.0  # classic Powell update (LANCELOT style)

        return float(p[0]), float(p[1])



    def psitc_sera(self, 
                   ref_guess_filename, 
                   max_iter=1000, 
                   tol=5.0e-5, 
                   delta_init=1e-2, 
                   delta_max=1.0):
        """
        Pseudo-Transient Continuation (PTC) with SER-A update for delta.

        Parameters
        ----------
        max_iter   : int
            Maximum number of Newton/PTC iterations.
        tol        : float
            Convergence tolerance on the norm of F.
        delta_init : float
            Initial pseudo-time step size.
        delta_max  : float
            Maximum allowed pseudo-time step size (SER-A cap).

        Notes
        -----
        The update rule is:
            delta_{k+1} = min( delta_k * (||F(u_k)|| / ||F(u_{k+1})|| ), delta_max ).
        """

        # 1) Build an initial guess for the boundary unknowns (x0s, pFs, p1f, tF):
        #    For example, random or from a file if it exists.
        x0s = np.random.uniform(-1.0, 1.0, (self.Ns-1, self.d))
        pFs = np.random.uniform(-1.0, 1.0, (self.Ns-1, self.d))
        p1f = 1.0
        p2f = 1.0
        tF  = 1.0

        # (Optional) Check if there's a saved solution you want to use:
        filename = ref_guess_filename
        ref_path = os.path.join(self.data_dir, filename)
        if os.path.exists(ref_path):
            print(f'Loading solution from file {filename}')
            saved = np.load(ref_path)
            sol_x = saved['sol_x']
            sol_p = saved['sol_p']
            sol_t = saved['sol_t']
            for i in range(self.Ns-1):
                x0s[i, :] = sol_x[i+1, 0, :]
                pFs[i, :] = sol_p[i, -1, :]
            tF = sol_t[-1, -1]
            p1f = sol_p[-1, -1, 0]
            p2f = sol_p[-1, -1, 1]


        # Flatten everything into one vector: xs
        xs = np.concatenate((x0s.ravel(), pFs.ravel(), [p1f], [p2f], [tF]))

        # 2) Initialize pseudo-time step delta
        delta = delta_init

        # 3) Evaluate F at the initial guess (needed to compute ratio later)
        x0_now, pF_now, p1_now, p2_now, tF_now = self.extract_variables(xs)
        Fval = self.F(x0_now, pF_now, p1_now, p2_now, tF_now)
        normF_old = np.linalg.norm(Fval, 2)

        # 4) Main iteration loop
        for iteration in range(max_iter):
            # (a) Build Jacobian M = dF/dx at the current guess
            M, baseF = self.compute_jacobian(xs)
            normF = np.linalg.norm(baseF, 2)


            # Check convergence
            print(f"Iteration {iteration:2d}, ||F|| = {normF:.3e}, delta = {delta:.3e}")
            if normF < tol:
                print("Converged!")
                break

            # (b) Solve for Newton step:  M * dx = -F
            dx = np.linalg.solve(M, -baseF)
            print(f"dx = {dx}")


            # (c) Form the tentative new guess:
            #     SER-A method applies the fraction 'delta' to the step
            xs_new = xs + delta*dx

            # (d) Evaluate the constraint norm at the new guess
            x0_new, pF_new, p1F_new, p2F_new, tF_new = self.extract_variables(xs_new)
            Fval_new = self.F(x0_new, pF_new, p1F_new, p2F_new, tF_new)
            normF_new = np.linalg.norm(Fval_new, 2)

            # (e) Update delta via SER-A
            #     delta_{k+1} = min( delta_k * (normF_old / normF_new), delta_max )
            #     Provided normF_new != 0 to avoid dividing by zero
            if normF_new > 1e-15:
                delta_new = delta * (normF_old / normF_new)
                delta = min(delta_new, delta_max)
            else:
                # If F is extremely small, you can just keep delta as is or set it to delta_max
                delta = delta_max

            # (f) Accept the update, store the new normF as 'old' for next iteration
            xs = xs_new
            normF_old = normF_new

        # 5) Final solution: force one last forward/backward to store in solver arrays
        x0_fin, pF_fin, p1F_fin, p2F_fin, tF_fin = self.extract_variables(xs)
        self.F(x0_fin, pF_fin, p1F_fin, p2F_fin, tF_fin)

        print("=====================================")
        print("SER-A PTC finished.")
        print(f"Final F = {self.con_x}, {self.con_p}, {self.con_pmp}")
        print(f"Hf      = {self.Hf}")
        print(f"tF      = {tF_fin}")
        print("=====================================")
    
    
    
    def psitc_serb(
        self,
        ref_guess_filename,
        max_iter=10000,
        tol=5.0e-5,
        delta_init=1.0e-5,
        delta_max=1.0,
        beta=0.3
    ):
        """
        Pseudo-Transient Continuation (PTC) with a SER-B-style update for delta.

        Parameters
        ----------
        max_iter : int
            Maximum number of Newton/PTC iterations.
        tol : float
            Convergence tolerance on the norm of F.
        delta_init : float
            Initial pseudo-time step size.
        delta_max : float
            Maximum allowed pseudo-time step size.
        beta : float
            The exponent used in the SER-B update rule (typical range: 0.3-0.8).

        Notes
        -----
        The SER-B update rule often takes the form:
            delta_{k+1} = min( delta_max,
                            delta_k * (||F(u_k)|| / ||F(u_{k+1})||)^beta )
        which is more robust than the linear ratio in SER-A.
        """

        p1f = 0.0
        p2f = 0.0
        #tF  = np.log(np.exp(xs[-1])+1.0)
        tF_init = 6.0
        tFv = np.log(np.expm1(tF_init))

        x0s = np.ones((self.Ns - 1, self.d))
        pFs = np.ones((self.Ns - 1, self.d))
        #Interpolate the costate variables
        for i in range(self.Ns-1):
            pFs[i,:] = np.array([p1f, p2f])*(i+1)/self.Ns
        dt = (self.tF - self.t0) / self.Ns  # Time step for each segment

        for i in range(self.Ns):
            self.sol_t[i, :] = np.linspace(self.t0 + i*dt, \
                                           self.t0 + (i+1)*dt, \
                                           self.Nt)
            if i > 0:
                #Assume a perfect circle for the state space guess
                x0s[i-1, :] = np.cos(i/self.Ns*2.0*np.pi)*self.x0        
        # 1) Build an initial interpolation
        #This is a limit cycle so we can generate a guess just by 
        #running the state space forward with a fixed control


        ref_path = os.path.join(self.data_dir, ref_guess_filename)
        
        if os.path.exists(ref_path):
            print(f'Loading solution from file {ref_guess_filename}')
            saved = np.load(ref_path)
            sol_x = saved['sol_x']
            sol_p = saved['sol_p']
            sol_t = saved['sol_t']
            for i in range(self.Ns - 1):
                x0s[i, :] = sol_x[i + 1, 0, :]
                pFs[i, :] = sol_p[i, -1, :]
            p1f = sol_p[-1, -1, 0]
            p2f = sol_p[-1, -1, 1]
            tFv = sol_t[-1, -1]
            

        # Flatten everything into one vector
        xs = np.concatenate((x0s.ravel(), pFs.ravel(), [p1f], [p2f], [tFv]))
        print(f'xs= {xs}')
    

        # 2) Initialize pseudo-time step delta
        delta = delta_init

        # 3) Evaluate F at the initial guess
        x0_now, pF_now, p1F_now, p2F_now, tF_now = self.extract_variables(xs)
        print('p1F_now = ', p1F_now)
        print('p2F_now = ', p2F_now)
        
        print(f"tF_now = {tF_now}")
        
        

        Fval = self.F(x0_now, pF_now, p1F_now, p2F_now, tF_now)
        print(f'Fval = {Fval}')
        
        
        


        normF_old = np.linalg.norm(Fval, 2)

        # 4) Main iteration loop
        for iteration in range(max_iter):

            # (a) Build Jacobian M at the current guess
            M, baseF = self.compute_jacobian(xs)
            print(f'M = {M}')
            print(f'baseF = {baseF}')
        
            normF = np.linalg.norm(baseF, 2)

            # Check convergence
            print(f"Iteration {iteration:2d}, ||F|| = {normF:.3e}, delta = {delta:.3e}"), 
            if normF < tol:
                print("Converged!")
                break
            # (b) Solve for Newton step:  M * dx = -F
            dx = np.linalg.solve(M, -baseF)
            print(f"dx = {dx}")
            # (c) Tentative new guess: SER-B applies 'delta * dx' but with a specialized update
            xs_new = xs + delta * dx

            # (d) Evaluate the constraint norm at the new guess
            x0_new, pF_new, p1F_new, p2F_new, tF_new = self.extract_variables(xs_new)
            print(f'x0_new = {x0_new}')
            print(f'pF_new = {pF_new}')
            print(f"tF_new = {tF_new}")
            print(f'p1F_new = {p1F_new}')
            print(f'p2F_new = {p2F_new}')
        
            Fval_new = self.F(x0_new, pF_new, p1F_new, p2F_new, tF_new)
            normF_new = np.linalg.norm(Fval_new, 2)
             
            # (e) Update delta via SER-B:
            #     delta_{k+1} = min( delta_max,
            #                        delta_k * (normF_old / normF_new)^beta )
            #
            #     Guard against tiny or zero normF_new to avoid divide by zero:
            if normF_new > 1e-15:
                ratio = normF_old / normF_new
                delta_new = delta * (ratio ** beta)
                delta = min(delta_new, delta_max)
            else:
                delta = delta_max

            # (f) Accept the update
            xs = xs_new
            normF_old = normF_new

        # 5) Final solution
        x0_fin, pF_fin, p1F_fin, p2F_fin, tF_fin = self.extract_variables(xs)
        self.F(x0_fin, pF_fin, p1F_fin, p2F_fin, tF_fin)

        print("=====================================")
        print("SER-B PTC finished.")
        print(f"Final F = {self.con_x}, {self.con_p}, {self.con_pmp}")
        print(f"Hf      = {self.Hf}")
        print(f"tF      = {tF_fin}")
        print("=====================================")
    
def format_number(num):
  """Formats a float to a string with two decimal places, replacing '.' with 'p'.

  Args:
    num: The float number to format.

  Returns:
    The formatted string.
  """
  formatted_string = f"{num:.2f}"
  return formatted_string.replace('.', 'p')

def run_solver(i):
    """
        Each call to run_solver will print to its own log file. 
    """
    all_results = {}
    for j in range(1): #This is the upper stiffness for umax
        Ns = 2
        Nt = 25
        umin = -1.0
        umax = 1.0+i*0.1
        uminstr = format_number(umin)
        umaxstr = format_number(umax)
        x02 = 2.0+0.01*j
        x01name = format_number(x02)

        log_filename = f"process_log_{x01name}_{uminstr}_{umaxstr}.txt"
        with open(log_filename, "w") as lf, contextlib.redirect_stdout(lf):
            x0 = np.array([0.0, x02])  # Initial state for the first segment
            pF = np.array([1.0, 0.0])  # Final costate for the last segment
            solver = VDPPMPSolver(x0, pF, Ns, Nt, umin, umax)
            solver.umin = umin
            solver.umax = umax
            uminname = format_number(umin)
            umaxname = format_number(umax)
            umaxrefname = str(round(umax-0.1,1)).replace('.', 'p')
            name = format_number(x02)
            ref_guess_filename = f'vdp_pmp_solution_{name}_{Ns}_{Nt}_{uminname}_{umaxrefname}.npz'
            #ref_guess_filename = f'vdp_pmp_solution_2p17_2_25_1p0_1p0.npz'
            os.makedirs(solver.data_dir, exist_ok=True)
            curr_filename = os.path.join(
                            solver.data_dir, 
                            f"vdp_pmp_solution_{name}_{Ns}_{Nt}_{uminname}_{umaxname}.npz")

            print('curr_filename = ', curr_filename)

            print(f'Attempting to load reference guess from {ref_guess_filename}')
            solver.psitc_serb(ref_guess_filename=ref_guess_filename)

            np.savez(curr_filename, 
                    sol_x=solver.sol_x, 
                    sol_p=solver.sol_p, 
                    sol_t=solver.sol_t)

            results = {
                'solver' : i,
                'index': j,
                'sol_x': solver.sol_x,
                'sol_p': solver.sol_p,
                'sol_t': solver.sol_t,
                'con_x': solver.con_x,
                'con_p': solver.con_p,
                'con_pmp': solver.con_pmp,
                'Hf': solver.Hf,
                'tF': solver.tF,
                'umin': umin,
                'umax': umax,
                'run_count': solver.run_count
            }
            all_results[j] = results

    return all_results

if __name__ == "__main__":
    max_workers = 1 # Adjust based on available CPU cores
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_solver, i): i for i in range(1)}

        for future in as_completed(futures):
            results = future.result()
            keys = results.keys()
            #Results is a dictionary 
            for key in keys:
                i = results[key]['solver']
                j = results[key]['index']
                print(f"\nResults for solver {i} iteration {j}:")
                print(f"solver.sol_x = {results[key]['sol_x']}")
                print(f"solver.sol_p = {results[key]['sol_p']}")
                print(f"solver.sol_t = {results[key]['sol_t']}")
                print(f"solver.con_x = {results[key]['con_x']}")
                print(f"solver.con_p = {results[key]['con_p']}")
                print(f"solver.con_pmp = {results[key]['con_pmp']}")
                print(f"solver.Hf = {results[key]['Hf']}")
                print(f"solver.tF = {results[key]['tF']}")
                print(f"solver.umin = {results[key]['umin']}")
                print(f"solver.umax = {results[key]['umax']}")
                print(f"solver.run_count = {results[key]['run_count']}")

