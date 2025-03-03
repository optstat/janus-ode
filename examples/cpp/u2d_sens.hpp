#ifndef U2D_PMP_SOLVER_HPP
#define U2D_PMP_SOLVER_HPP
#include "../../src/cpp/janus_ode_common.hpp"
/**
 * PETSC is used to solve the augmented system
 * This is a universal solver without knowledge of the underlying system
 * It uses the Pseudo Transient Continuation method to solve the forward 
 * Pontryagin Minimum Principle.  In such a formulation the costates are
 * appended to the state variables and the entire system is propagated forward.
 * The resulting two point BVP system is solved using a pseudo transient method.
 * In addition the use a preconditioner in CVODES and the GMRES method to 
 * calculate the inverse of the sensitivity matrix.
 * The advantage of using the inverse method is that the inverse sensitivity 
 * matrix can be constructed one row a time in the forward direction avoiding the
 * adjoint method which is very unstable and difficult to construct 
 * for very stiff systems.
 * Proof for the inverse sensitivity method is
 * for the ODE x'(t) = f(x(t),t) with x(0) = x0
 * Define the forward sensitivity as M(t) = d x(t)/d x0
 * The the inverse sensitivity is given by N(t) = d x0/d x(t)
 * so that M(t)N(t) = I
 * d/dt(M(t)N(t)) = 0
 * d/dt(M(t))N(t) + M(t)d/dt(N(t)) = 0
 * But M(t)*N(t) = J(t)*M(t)*N(t) = J(t)I = J(t)
 * We therefore have J(t)+M(t)*N'(t) = 0
 * or N'(t) = -M(t)^-1*J(t)
 * But M^-1(t) = N(t) so we simply have
 * N'(t) = -N(t)*J(t) with N(0) = I
 * which is the forward sensitivity equation which can be solved
 * one row at a time in the forward direction.
 * In this implementation even though the 2D system is low dimensional
 * we demosntrate how to construct the forward sensitivity equations
 * without needing to form either N(t) or J(t) fully but row by row
 * in the forward direction using a sparse Jacobian
 * The Jacobian structure is
 * 
\begin{pmatrix}
\partial (x{\prime}) / \partial x & \partial (x{\prime}) / \partial N \\
\partial (N{\prime}) / \partial x & \partial (N{\prime}) / \partial N
\end{pmatrix}.
 *Concretely:
	1.	\partial(x{\prime})/\partial x: This is \partial f/\partial x\bigl(t, x\bigl), i.e. the same Jacobian you’d supply for a standard stiff ODE in x.
	2.	\partial(x{\prime})/\partial N: Typically zero, because x{\prime}(t) does not depend on N(t).
	3.	\partial(N{\prime})/\partial x: This is \partial\bigl[-N(t) \, J(t)\bigr]/\partial x. You get derivatives of -N(t) \, J(t) w.r.t. x. Part of this includes -\,N(t)\,\partial(J)\!/\!\partial x.
	4.	\partial(N{\prime})/\partial N: This is \partial\bigl[-N(t)\,J(t)\bigr]/\partial N. You’ll have terms that represent “how changing N changes the product -N\,J.
  *
  * We have to flatten the sensitivity matrix into a vector yielding a system of n+n^2 equations
  * With the jacobian being n+n^2 by n+n^2.  
  * The jacobian is however mostly zero except for the 4th and 5th rows which are the state equations
  */  
namespace u2d
{
  /**
   * Forward only PMP solution
   */
  namespace vdpforpmp
  {

    const int NS = 6+6*6;                   // Augmented system states plus costates plus the 6*6 sensitivity states
    sunrealtype RTOL = SUN_RCONST(1.0e-6);  /* scalar relative tolerance            */
    sunrealtype ATOL1 = SUN_RCONST(1.0e-8); /* vector absolute tolerance components */
    sunrealtype ATOL2 = SUN_RCONST(1.0e-8);
    sunrealtype ATOL3 = SUN_RCONST(1.0e-8);
    sunrealtype ATOL4 = SUN_RCONST(1.0e-8);
    sunrealtype W = SUN_RCONST(1.0e-4);                    // This is regularization on the control
    sunrealtype alpha = SUN_RCONST(1.0);                   // This is regularization on the control
    sunrealtype sigma = SUN_RCONST(1.0e-4);                // Width of the time gaussian penalty term
    sunrealtype pi = SUN_RCONST(3.14159265358979323846);   // Pi
    sunrealtype gf = 1.0 / sqrt(2.0 * pi * sigma * sigma); // Gaussian factor
    sunbooleantype err_con;
    sunrealtype pbar[NS];

    static SUNMatrix J;
    SUNContext sunctx;
    static N_Vector y;
    static N_Vector *yS;
    static SUNLinearSolver LS;

    // Global initialization of the CVODES and KINSOL solvers

    class UserData
    {
    public:
      N_Vector ts;   // Data times
      N_Vector x1s;  // Data states for x1
      N_Vector x2s;  // Data states for x2
      int NDATA = 0; // Number of data points

      UserData(torch::Tensor ts,
               torch::Tensor x1s,
               torch::Tensor x2s,
               SUNContext sunctx)
      {
        this->ts = N_VNew_Serial(ts.size(0), sunctx);
        this->x1s = N_VNew_Serial(x1s.size(0), sunctx);
        this->x2s = N_VNew_Serial(x2s.size(0), sunctx);
        for (int i = 0; i < ts.size(0); i++)
        {
          Ith(this->ts, i + 1) = ts[i].item<double>();
          Ith(this->x1s, i + 1) = x1s[i].item<double>();
          Ith(this->x2s, i + 1) = x2s[i].item<double>();
        }
        this->NDATA = ts.size(0);
      }

      ~UserData()
      {
        N_VDestroy_Serial(ts);
        N_VDestroy_Serial(x1s);
        N_VDestroy_Serial(x2s);
      }
    };

    /* Functions Called by the Solver */

    static int f(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data);

    static int Jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                   void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

    static int fS(int Ns, sunrealtype t, N_Vector y, N_Vector ydot, int iS,
                  N_Vector yS, N_Vector ySdot, void *user_data, N_Vector tmp1,
                  N_Vector tmp2);

    static int ewt(N_Vector y, N_Vector w, void *user_data);


    /*
     * Dynamics routine augmented with sensitivity equations
     * The Hamiltionian in this case is
     * H = l1*u1 + l2*u2 + l3+W/2*(u1^2 + u2^2) + \sum alpha*w_i(t)*(x1-x1_i)^2*(x2-x2_i)^2
     * where x1t_i and x2t_i are the data states the trajectory has to pass through
     * w_i(t) = gf*exp(-(t-t_i)^2/(2*sigma^2))
     */
    static int f(sunrealtype t,   // Current time
                 N_Vector y,      // Current state
                 N_Vector ydot,   // Current state derivative
                 void *user_data) // UserData that contain the current costate information
    {
      UserData data = *(UserData *)user_data;
      sunrealtype l1, l2, l3, x1, x2, x3;
      l1 = Ith(y, 1);
      l2 = Ith(y, 2);
      l3 = Ith(y, 3); // Time costate

      x1 = Ith(y, 4);
      x2 = Ith(y, 5);
      x3 = Ith(y, 6); // Time

      N_VConst(0.0, ydot);

      auto u1 = -l1 / W;
      auto u2 = -l2 / W;
      // Costate time derivatives
      Ith(ydot, 1) = u1;
      Ith(ydot, 2) = u2;
      // State time derivatives
      Ith(ydot, 4) = 0.0; // partial H by partial x1
      Ith(ydot, 5) = 0.0; // partial H by partial x2
      for (int i = 0; i < data.NDATA; i++)
      {
        auto wt = gf * exp(-(t - Ith(data.ts, i)) * (t - Ith(data.ts, i)) / (2 * sigma * sigma));
        Ith(ydot, 4) += -2 * alpha * (Ith(data.x1s, i + 1) - x1) * wt;
        Ith(ydot, 5) += -2 * alpha * (Ith(data.x2s, i + 1) - x2) * wt;
      }
      Ith(ydot, 6) = 1.0; // partial H by partial x3 (time)
      return (0);
    }

    /**
     * Jacobian.  Here the jacobian is mostly zero except for the
     */
    static int Jac(sunrealtype t,
                   N_Vector y,
                   N_Vector fy,
                   SUNMatrix J,
                   void *user_data,
                   N_Vector tmp1,
                   N_Vector tmp2,
                   N_Vector tmp3) // Temporary vectors
    {
      sunrealtype l1, l2, l3, x1, x2, x3;
      l1 = Ith(y, 1);
      l2 = Ith(y, 2);
      l3 = Ith(y, 3); // Time costate
      x1 = Ith(y, 4);
      x2 = Ith(y, 5);
      x3 = Ith(y, 6); // Time
      //The rest of the states are the various sensitivities
      N_Vector n2 = N_VNew_Serial(6*6, sunctx);
      UserData data = *(UserData *)user_data;
      SUNMatZero(J); // zero out all the entries
      // Only entries that are non zero
      // are in the 4th and 5th rows
      /*xd4 = Ith(ydot, 4) = 0.0; //partial H by partial x1
      xd5 = Ith(ydot, 5) = 0.0; //partial H by partial x2
      for ( int i=0; i < data.NDATA; i++ ) {
        auto wt = gf*exp(-(t-Ith(data.ts, i))*(t-Ith(data.ts, i))/(2*sigma*sigma));
        xd4 += -2*alpha * (Ith(data.x1s, i+1) - x1) *wt;
        xd5 += -2*alpha * (Ith(data.x2s, i+1) - x2) *wt;
      }*/
      // xd4
      std::cerr << "data.x1s=";
      janus::print_N_Vector(data.x1s);
      std::cerr << "data.x2s=";
      janus::print_N_Vector(data.x2s);
      std::cerr << "data.ts=";
      janus::print_N_Vector(data.ts);
      std::cerr << "NDATA=" << data.NDATA << std::endl;
      exit(1);
      for (int i = 0; i < data.NDATA; i++)
      {

        auto wt = gf * exp(-(t - Ith(data.ts, i+1)) * (t - Ith(data.ts, i+1)) / (2 * sigma * sigma));
        IJth(J, 4, 1) += 2 * alpha * wt;
        IJth(J, 4, 2) += 2 * alpha * wt;
      }
      // xd5
      for (int i = 0; i < data.NDATA; i++)
      {
        auto wt = gf * exp(-(t - Ith(data.ts, i+1)) * (t - Ith(data.ts, i+1)) / (2 * sigma * sigma));
        IJth(J, 5, 1) += -2 * alpha * (Ith(data.x1s, i + 1) - x1) * wt;
        IJth(J, 5, 2) += -2 * alpha * (Ith(data.x2s, i + 1) - x2) * wt;
      }

      return (0);
    }


    /*
     * EwtSet function. Computes the error weights at the current solution.
     */
    static int ewt(N_Vector y, N_Vector w, void *user_data)
    {
      int i;
      sunrealtype yy, ww, rtol, atol[4];

      rtol = RTOL;
      atol[0] = ATOL1;
      atol[1] = ATOL2;
      atol[2] = ATOL3;
      atol[3] = ATOL4;

      for (i = 1; i <= 4; i++)
      {
        yy = Ith(y, i);
        ww = rtol * ABS(yy) + atol[i - 1];
        if (ww <= 0.0)
        {
          return (-1);
        }
        Ith(w, i) = 1.0 / ww;
      }
      return (0);
    }

    /*
     * Perform forward sensitivity analysis using the CVODES solver
     * and return the final state and costate sensitivities
     * The method takes in the initial costates and initial state conditions
     * y0in: Initial augmented state
     * tsin: Data times
     * xsin: Data states for x1 and x2
     *
     */
    std::tuple<torch::Tensor, torch::Tensor> for_sens(torch::Tensor y0in,  // Initial augmented state
                                                      torch::Tensor tsin,  // Data times
                                                      torch::Tensor xsin // Data states for x1 and x2
    )
    {
      /* Create the SUNDIALS context that all SUNDIALS objects require */
      auto retval = SUNContext_Create(SUN_COMM_NULL, &sunctx);
      TORCH_CHECK(sunctx != NULL, "SUNContext_Create failed");
      sunrealtype T0 = 0.0; // Initial time
      y = N_VNew_Serial(NS, sunctx);
      Ith(y, 1) = y0in[0].item<double>(); // l1
      Ith(y, 2) = y0in[1].item<double>(); // l2
      Ith(y, 3) = y0in[2].item<double>(); // l3

      Ith(y, 4) = y0in[3].item<double>(); // x1
      Ith(y, 5) = y0in[4].item<double>(); // x2
      Ith(y, 6) = y0in[5].item<double>(); // x3
      TORCH_CHECK(y != NULL, "N_VNew_Serial failed");
      /* Call CVodeCreate to create the solver memory and specify the
       * Backward Differentiation Formula */
      void *cvode_mem = CVodeCreate(CV_BDF, sunctx);
      TORCH_CHECK(cvode_mem != NULL, "CVodeCreate failed");
      /* Call CVodeInit to initialize the integrator memory and specify the
       * user's right hand side function in y'=f(t,y), the initial time T0, and
       * the initial dependent variable vector y. */
      retval = CVodeInit(cvode_mem, f, T0, y);
      TORCH_CHECK(retval == CV_SUCCESS, "CVodeInit failed");
      /* Call CVodeWFtolerances to specify a user-supplied function ewt that sets
       * the multiplicative error weights w_i for use in the weighted RMS norm */
      retval = CVodeWFtolerances(cvode_mem, ewt);

      TORCH_CHECK(retval == CV_SUCCESS, "CVodeWFtolerances failed");
      long int mxsteps = 1000; // or some larger number

      retval = CVodeSetMaxNumSteps(cvode_mem, mxsteps);
      TORCH_CHECK(retval == CV_SUCCESS, "CVodeSetMaxNumSteps failed");
      auto x1s = xsin.index({Slice(), 0});
      auto x2s = xsin.index({Slice(), 1});
      /* Attach user data */
      UserData data(tsin, x1s, x2s, sunctx);

      retval = CVodeSetUserData(cvode_mem, &data);
      // if (check_retval(&retval, "CVodeSetUserData", 1)) { return {}; }
      TORCH_CHECK(retval == CV_SUCCESS, "CVodeSetUserData failed");
      /* Create dense SUNMatrix for use in linear solves */
      J = SUNDenseMatrix(NS, NS, sunctx);
      /* Create another dense matrix for the sensitivites*/
      // if (check_retval((void*)A, "SUNDenseMatrix", 0)) { return {}; }
      TORCH_CHECK(J != NULL, "SUNDenseMatrix failed");
      /* Create dense SUNLinearSolver object */
      LS = SUNLinSol_SPGMR(y, SUN_PREC_LEFT, maxl, sunctx);

      // if (check_retval((void*)LS, "SUNLinSol_Dense", 0)) { return {}; }
      TORCH_CHECK(LS != NULL, "SUNLinSol_Dense failed");
      /* Attach the matrix and linear solver */
      retval = CVodeSetLinearSolver(cvode_mem, LS, J);
      TORCH_CHECK(retval == CV_SUCCESS, "CVodeSetLinearSolver failed");
      /* Set the user-supplied Jacobian routine Jac */
      retval = CVodeSetJacFn(cvode_mem, Jac);
      TORCH_CHECK(retval == CV_SUCCESS, "CVodeSetJacFn failed");
      /* Sensitivity-related settings */

      /* Set sensitivity initial conditions */
      yS = N_VCloneVectorArray(NS, y);
      TORCH_CHECK(yS != NULL, "N_VCloneVectorArray failed");
      for (int is = 0; is < NS; is++)
      {
        N_VConst(ONE, yS[is]);
      }

      /* Call CVodeSensInit1 to activate forward sensitivity computations
       * and allocate internal memory for COVEDS related to sensitivity
       * calculations. Computes the right-hand sides of the sensitivity
       * ODE, one at a time */
      retval = CVodeSensInit1(cvode_mem, NS, CV_STAGGERED, fS, yS);
      TORCH_CHECK(retval == CV_SUCCESS, "CVodeSensInit failed");

      /* Call CVodeSensEEtolerances to estimate tolerances for sensitivity
       * variables based on the rolerances supplied for states variables and
       * the scaling factor pbar */
      retval = CVodeSensEEtolerances(cvode_mem);
      TORCH_CHECK(retval == CV_SUCCESS, "CVodeSensEEtolerances failed");

      /* Set sensitivity analysis optional inputs */
      /* Call CVodeSetSensErrCon to specify the error control strategy for
       * sensitivity variables */
      retval = CVodeSetSensErrCon(cvode_mem, err_con);
      TORCH_CHECK(retval == CV_SUCCESS, "CVodeSetSensErrCon failed");
      /* Set parameter scaling factor */
      pbar[0] = 1.0;
      pbar[1] = 1.0;
      pbar[2] = 1.0;
      pbar[3] = 1.0;
      pbar[4] = 1.0;
      pbar[5] = 1.0;

      /* Call CVodeSetSensParams to specify problem parameter information for
       * sensitivity calculations */
      retval = CVodeSetSensParams(cvode_mem, NULL, pbar, NULL);
      TORCH_CHECK(retval == CV_SUCCESS, "CVodeSetSensParams failed");
      /* Set the final time */
      sunrealtype TF = tsin[-1].item<double>();

      retval = CVode(cvode_mem, TF, y, &T0, CV_NORMAL);
      //Get the sensitivity information
      retval = CVodeGetSens(cvode_mem, &TF, yS);
      //We need to cast this in the form F(x) = 0 so we can use a nonolinear solver to solve for the initial conditions
      torch::Tensor sensitivity = torch::zeros({NS, 1}, torch::kDouble);
      sensitivity.index_put_({0}, Ith(yS[0], 1));
      sensitivity.index_put_({1}, Ith(yS[1], 1));
      sensitivity.index_put_({2}, Ith(yS[2], 1));
      sensitivity.index_put_({3}, Ith(yS[3], 1));
      sensitivity.index_put_({4}, Ith(yS[4], 1));
      sensitivity.index_put_({5}, Ith(yS[5], 1));
      //Get the final state
      torch::Tensor final_state = torch::zeros({NS, 1}, torch::kDouble);
      final_state.index_put_({0}, Ith(y, 1));
      final_state.index_put_({1}, Ith(y, 2));
      final_state.index_put_({2}, Ith(y, 3));
      final_state.index_put_({3}, Ith(y, 4));
      final_state.index_put_({4}, Ith(y, 5)); 
      final_state.index_put_({5}, Ith(y, 6));  

      auto res = std::make_tuple(final_state, sensitivity);
      /* Free yS */
      N_VDestroyVectorArray(yS, NS);
      /* Free y */
      N_VDestroy(y);
      /* Free J */
      SUNMatDestroy(J);
      /* Free LS */
      SUNLinSolFree(LS);
      /* Free cvode_mem */
      CVodeFree(&cvode_mem);
      /* Free the SUNDIALS context */
      SUNContext_Free(&sunctx);
      /* Free user data */
      return res;
      
    }

  }

} // namespace u2d
#endif