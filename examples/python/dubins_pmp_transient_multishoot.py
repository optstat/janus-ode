import numpy as np
import os
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d






class VDPPMPSolver:
    """
    Skeleton class to demonstrate:
      - Forward integration of state variables
      - Backward integration of costates
      - Bang-bang control rule (separate function, not used in first guess)
      - The Hamiltonian in this problem is as follows:
        H = p1*cos(x3)+ p2*sin(x3)+p3*u+p4
      - Stabilization of the stiffness using multiple shooting
      - The segments proceed from x0i to xfi and pfi to p0i in reverse
      - x0i--------------------->xfi x0_{i+1}---------------------->xf_{i+1}
                                  |                                    |
                                  |                                    |
        p0i<---------------------pfi p0_{i+1}<----------------------pf_{i+1}
        - The endpoints of final segments are determined by the Pontryagin Minimum Principle
        - The initial state variables are fixed
        - The endpoints of the intermediate segments are determined by the continuity of the costates
        - and the state variables
        - The method used to solve the TPBVP is the Pseudo-Transient Continuation Method
        - We will use the representation [Ns, Nt, d] for the state and costate variables
    """
    def __init__(self, 
                 x0, # Initial state for the first segment
                 Ns,
                 Nt,
                 umin,  # Minimum control value
                 umax   # Maximum control value 
                ): # Number of time points in each segment
        """
        Initialize the solver with:
          x0s  : initial states for each segment [x1(0), x2(0), x3(0)] except for the first segment
          pFs  : Final costates values at each segment [p1(T), p2(T), p3(T)] except for the last segment
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
        self.atol    = 1.0e-5
        self.rtol    = 1.0e-3
        self.sol_x   = np.zeros((self.Ns, self.Nt, self.d)) #State solution
        self.sol_p   = np.ones((self.Ns, self.Nt, self.d)) #Costate solution
        self.con_x   = np.zeros((self.Ns-1, self.d)) #State segment deltas
        self.con_p   = np.zeros((self.Ns-1, self.d)) #Costate segment deltas
        self.con_pmp = np.zeros((2,)) #Endpoint constraints. There are two variables p1f and tF and two constraints
        #Create a guess of linear interpolation for the state and costate
        # First segment is fixed
        self.sol_x[0, 0, 0] = x0[0]  # Initial state for the first segment
        self.sol_x[0, 0, 1] = x0[1]  # Initial state for the first segment  
        self.sol_x[0, 0, 2] = x0[2]  # Initial state for the first segment
        self.umin = umin
        self.umax = umax
        self.count = 0
        self.x0 = x0

    
    ## Initialize the variables for the system
    def init_variables(self, x0s, pFs, p1F, tF):
        """
        Set the initial conditions for the segments.
        """
        #Make sure the dimension of the inputs is 2D
        assert x0s.ndim == 2
        assert pFs.ndim == 2
        assert x0s.shape[0] == pFs.shape[0]
        assert x0s.shape[1] == pFs.shape[1]
        self.x0s = x0s
        self.pFs = pFs
        # Set the initial state for the first segment
        #Set the intermediate segments for states 
        for i in range(1, self.Ns):
            self.sol_x[i, 0, :] = x0s[i-1, :]  # Interpolated initial state for segment i
        #Set the intermediate segments for costates
        for i in range(self.Ns-1):
            self.sol_p[i, -1, :] = pFs[i, :]
        # Set the final state for the last segment
        self.sol_p[-1, -1, 0] = p1F  # Final costate for the last segment
        self.sol_p[-1, -1, 1] = 0.0  # Zero because x2f is free  
        self.sol_p[-1, -1, 2] = 1.0  
        #We need to adjust the time for all segments
        self.sol_t       = np.zeros((self.Ns, self.Nt))  # Time discretization for each segment
        self.tF  = tF

        self.t0      = 0.0 # Start time for the first segment
        
        dt = (self.tF - self.t0) / self.Ns  # Time step for each segment
        for i in range(self.Ns):
            self.sol_t[i, :] = np.linspace(self.t0 + i*dt, \
                                           self.t0 + (i+1)*dt, \
                                           self.Nt)

        
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
            self.u_star = self.bang_bang_control(x, p_interp(t))            
            return self.f_state(t, x, self.u_star)
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
                method='BDF',
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
            u = self.bang_bang_control(x, p)
            # dp/dt
            dpdt = self.f_costate(t, p, x, u)
            return [dpdt[0], dpdt[1], dpdt[2]]
        #start at the end
        pF = self.sol_p[i, -1, :] #Final costate for the segment
        sol = solve_ivp(
                ode_costate,
                t_span,
                y0=pF,
                t_eval=t_eval,
                method='BDF',
                atol = self.atol,
                rtol = self.rtol
            )
        self.sol_p[i, :, :] = sol.y[:, ::-1].T  #Reverse and transpose the solution to match the time order

    def solve_segment(self, i:int):
        max_iterations = 100  
        for _ in range(max_iterations):
            x_old = self.sol_x[i, :, :].copy()
            p_old = self.sol_p[i, :, :].copy()

            # Forward/backward
            self.solve_forward(i)
            self.solve_backward(i)

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
            self.con_x[i, :] = self.sol_x[i+1, 0, :]-self.sol_x[i, -1, :] 
            self.con_p[i, :] = self.sol_p[i+1, 0, :]-self.sol_p[i, -1, :] 
        #Now calculate the PMP violations
        self.con_pmp[0] = self.sol_x[-1, -1, 0] #x1f
        x1f = self.sol_x[-1, -1, 0]
        x2f = self.sol_x[-1, -1, 1]
        p1f = self.sol_p[-1, -1, 0]
        p2f = self.sol_p[-1, -1, 1]
        xf = self.sol_x[-1, -1, :]
        pf = self.sol_p[-1, -1, :]
        ustar = self.bang_bang_control(xf, pf)
        self.Hf = p1f*x2f + p2f*ustar*((1-x1f**2)*x2f - x1f) + 1.0
        self.con_pmp[1] = self.Hf #Hamiltonian at the final time which should be equal to 0.0

    def finite_diff_step(self, x, rel_power=1.0/3.0, abs_floor=1e-8):
        """
        Returns a step size for finite difference on 'x' that
        balances relative scale and a minimum absolute floor.
        """
        eps = np.finfo(float).eps  # ~2.220446049250313e-16
        return max((eps**rel_power)*abs(x), abs_floor)        

    #This function is the TPBVP formulation that takes as input
    #1.  The segments initial states and 
    #2.  Final costate estimates 
    #3.  Time discretization for all segments
    #4.  Maximum and minimum control values
    # It returns the differences between the segments endpoints
    #    and the differences of the costates between segments
    #    as well as the transversality condition at the final time
    def F(self, x0s, pFs, p1f, tF):
        self.x0s = x0s
        self.pFs = pFs
        self.tF  = tF
        if tF < 0.0:
            tf = 0.01
        self.p1f = p1f
        self.init_variables(x0s, pFs, p1f, tF)  # Set the initial conditions for the segments
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
        p1f = xs[-2]  # Final costate for the last segment
        tF  = xs[-1]  # Final time for the last segment
        return x0s, pFs, p1f, tF



    def pseudo_dyns(self, t, xs):
        #There are 2*(Ns-1)*d+2 variables and 2*(Ns-1)*d+2 constraints
        MN = 2*(self.Ns-1)*self.d + 2 # Number of constraints
        M = np.zeros((MN, MN))  # Sensitivity matrix
        for i in range(MN):
            #Use finite differences to calculate the derivatives
            xsp = xs.copy()
            h = self.finite_diff_step(xsp[i])
            xsp[i] += h  # Small perturbation for finite difference
            x0sp, pFsp, p1pF, tpF = self.extract_variables(xsp)  # Extract perturbed values
            # Now we can call the F function with the perturbed values
            conp = self.F(x0sp, pFsp, p1pF, tpF)  # Calculate the constraints for the perturbed xs
            xsm = xs.copy()
            xsm[i] -= h 
            x0sm, pFsm, p1mF, tmF = self.extract_variables(xsm)
            # Now we can call the F function with the perturbed values
            conm = self.F(x0sm, pFsm, p1mF, tmF)  # Calculate the constraints for the perturbed xs
            # Calculate the constraints for perturbed xs
            M[:, i] = (conp - conm) / (2*h)  # Central difference approximation
        #Solver for dxdtau
        # dx/dtau = M^{-1} * F(x)
        x0, pF, p1, t = self.extract_variables(xs)  # Extract the current values
        con = self.F(x0, pF, p1, t)  # Calculate the constraints for the current xs
        print(f'con = {con}')
        dxdtau = np.linalg.solve(M, -con)  # Solve the linear system
        if self.count == 10:
           pass
        self.count += 1
        return dxdtau   
        #We need to calculate the sensitivity matrix do this using finite differences

    


    def psitc(self):
        #H = p1*x2 + p2*u*((1-x1^2)*x2 - x1) + 1
        #x1f = 0.0 is the only state constraint
        #Terminal costate for the last segment
        #p2f = 0.0 since x2f is free
        #p3f = 1.0 since tf is free but dpsi/dtf = 1.0
        #Construct x0s, pFs and ts in a flattened array
        # Initial states for each segment
        x0s = np.random.uniform(-1.0, 1.0, (self.Ns-1, self.d))
        # Final costates for each segment
        pFs = np.random.uniform(-1.0, 1.0, ((self.Ns-1, self.d)))
        tF = 1.0  # Final time for the last segment
        p1f = 1.0  # Final costate for the last segment
        
        #If file vdp_pmp_solution_3_5_1p0_1p0.npz exists, load it
        if os.path.exists('vdp_pmp_solution_5_25_1p0_1p0.npz'):
            #Load the solution from the file
            print('Loading solution from file vdp_pmp_solution_5_25_1p0_1p0.npz')
            sol_x = np.load('vdp_pmp_solution_3_5_1p0_1p0.npz')['sol_x']
            sol_p = np.load('vdp_pmp_solution_3_5_1p0_1p0.npz')['sol_p']
            sol_t = np.load('vdp_pmp_solution_3_5_1p0_1p0.npz')['sol_t']
            for i in range(self.Ns-1):
                x0s[i, :] = sol_x[i+1, 0, :]
                pFs[i, :] = sol_p[i, -1, :]
            tF = sol_t[-1, -1]/2
        # Time discretization for each segment
        #Do linear interpolation for the segments
        #for i in range(Ns-1):
        #    fact = (i + 1) / float(Ns)  # Linear interpolation factor
        #    x0s[i, :] = (1.0 - fact) * x0 + fact * xF  # Interpolated initial state for segment i
        #    pFs[i, :] = (1.0 - fact) * p0 + fact * pF  # Interpolated final costate for segment i
        #Flatten the inputs into a single array
        xs = np.concatenate((x0s.flatten(), pFs.flatten(),   (p1f,), (tF,)))  # Flatten the initial states and final costates
        #Solve the problem using the pseudo-transient relaxation method using the Radau ode solver
        #and the pseudo-dynamics defined by the F function
        sol = solve_ivp(
                self.pseudo_dyns,  # The pseudo-dynamics function
                [0, 10.0],  # Time span for the relaxation process
                xs,  # Initial guess for the states and costates
                method='Radau',  # Radau method for stiff problems
                first_step=0.1,  # Initial step size
                atol=1.0e-5,
                rtol=1.0e-3
            )

    


# -----------------------------------------------------------------------------
# Example usage (replace with your own in a separate script or notebook):
#
if __name__ == "__main__":
    x0 = np.array([1.0, 0.0, 0.0])  # Initial state for the first segment
    Ns = 5  # Number of segments
    Nt = 25    # Number of time points in each segment
    umin = 1.0  # Minimum control value
    umax = 1.0  # Maximum control value
    # Create an instance of the solver
    solver = VDPPMPSolver(x0, Ns, Nt, umin, umax)
    solver.psitc()
    #Save the solution to a file
    np.savez('vdp_pmp_solution_5_25_1p0_1p0.npz', 
              sol_x=solver.sol_x, 
              sol_p=solver.sol_p, 
              sol_t=solver.sol_t)
    # Print the results
    print(f'solver.sol_x = {solver.sol_x}')
    print(f'solver.sol_p = {solver.sol_p}')
    print(f'solver.sol_t = {solver.sol_t}')
    print(f'solver.con_x = {solver.con_x}')
    print(f'solver.con_p = {solver.con_p}')
    print(f'solver.con_pmp = {solver.con_pmp}')
    print(f'solver.Hf = {solver.Hf}')
    print(f'solver.run_count = {solver.run_count}')