#ifndef FORWARD_VDP_CVODES_SOLVER_HPP
#define FORWARD_VDP_CVODES_SOLVER_HPP
#include "../../src/cpp/janus_ode_common.hpp"

namespace u2d
{
  namespace vdp
  {
    /**
     * Global variables
     */
    /* Functions Called by the Solver */
    
    typedef struct
    {
      sunrealtype mu; /* Initial conditions */
    } UserData;
   
    

    static int f(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data);

    static int Jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                   void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

    static int ewt(N_Vector y, N_Vector w, void *user_data);

    int NP = 2;

    // These will be reused in the sensitivity calculations
    sunrealtype RTOL = SUN_RCONST(1.0e-6);  /* scalar relative tolerance            */
    sunrealtype ATOL1 = SUN_RCONST(1.0e-8); /* vector absolute tolerance components */
    sunrealtype ATOL2 = SUN_RCONST(1.0e-8);
    sunrealtype x10 = SUN_RCONST(2.0);
    sunrealtype x20 = SUN_RCONST(0.0);
    sunrealtype tf = 100.0; // Final time
    int Nt = 1000;
    sunrealtype dt = tf / Nt; // We will generate 1000 data points
    static UserData data;
    static SUNMatrix J;
    static N_Vector y;
    static SUNLinearSolver LS;
    static SUNContext sunctx;
    static sunrealtype t, tout;
    static sunbooleantype err_con;
    static bool init = false; // Flag to check if the solver has been initialized
    /*
     * Dynamics routine augmented with sensitivity equations
     * for the ground truth model
     */
    static int f(sunrealtype t,   // Current time
                 N_Vector y,      // Current state
                 N_Vector ydot,   // Current state derivative
                 void *user_data) // UserData that contain the current costate information
    {
      UserData data = *(UserData *)user_data;
      sunrealtype mu = data.mu;
      sunrealtype x1, x2, xd1, xd2;
      x1 = Ith(y, 1);
      x2 = Ith(y, 2);

      xd1 = Ith(ydot, 1) = x2;
      xd2 = Ith(ydot, 2) = mu * (1 - x1 * x1) * x2 - x1;
      return (0);
    }

    /**
     * Jacobian routine for the ground truth model
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
      sunrealtype x1, x2;
      x1 = Ith(y, 1);
      x2 = Ith(y, 2);
      UserData data = *(UserData *)user_data;
      sunrealtype mu = data.mu;

      IJth(J, 1, 1) = 0.0;
      IJth(J, 1, 2) = 1.0;
      // mu*(1 - x1 * x1)*x2 - x1;
      IJth(J, 2, 1) = -2 * mu * x1 * x2 - 1.0;
      IJth(J, 2, 2) = mu * (1 - x1 * x1);
      return (0);
    }


     /*
  * EwtSet function. Computes the error weights at the current solution.
  */
 
 static int ewt(N_Vector y, N_Vector w, void* user_data)
 {
   int i;
   sunrealtype yy, ww, rtol, atol[4];
 
   rtol    = RTOL;
   atol[0] = ATOL1;
   atol[1] = ATOL2;
 
   for (i = 1; i <= 2; i++)
   {
     yy = Ith(y, i);
     ww = rtol * ABS(yy) + atol[i - 1];
     if (ww <= 0.0) { return (-1); }
     Ith(w, i) = 1.0 / ww;
   }
 
   return (0);
 }



    /**
     * Return the data with a time vector
     */
    std::tuple<torch::Tensor, torch::Tensor> gen_data(torch::Tensor x0in,
                                                      torch::Tensor muin,
                                                      torch::Tensor ftin,
                                                      int Nt)
    {
      torch::Tensor yres = torch::zeros({Nt+1, 2});
      torch::Tensor tres = torch::zeros({Nt+1});
      sunrealtype x10 = x0in[0].item<double>();
      sunrealtype x20 = x0in[1].item<double>();
      

      //Put this in user data
      data.mu = muin.item<double>();
      sunrealtype ft = ftin.item<double>();
      sunrealtype dt = ft/ Nt; // We will generate 1000 data points
      // Initialize CVODES
      SUNContext sunctx;
      /* Create the SUNDIALS context that all SUNDIALS objects require */
      auto retval = SUNContext_Create(SUN_COMM_NULL, &sunctx);
      TORCH_CHECK(sunctx != NULL, "SUNContext_Create failed");

      y = N_VNew_Serial(NP, sunctx);
      TORCH_CHECK(y != NULL, "N_VNew_Serial failed");
      /* Call CVodeCreate to create the solver memory and specify the
       * Backward Differentiation Formula */
      void *cvode_mem = CVodeCreate(CV_BDF, sunctx);
      TORCH_CHECK(cvode_mem != NULL, "CVodeCreate failed");
      /* Call CVodeInit to initialize the integrator memory and specify the
       * user's right hand side function in y'=f(t,y), the initial time T0, and
       * the initial dependent variable vector y.
       */
      Ith(y, 1) = x10;
      Ith(y, 2) = x20;

      sunrealtype T0 = 0.0; // Initial time
      retval = CVodeInit(cvode_mem, f, T0, y);
      TORCH_CHECK(retval == CV_SUCCESS, "CVodeInit failed");
      /* Call CVodeWFtolerances to specify a user-supplied function ewt that sets
       * the multiplicative error weights w_i for use in the weighted RMS norm */
      retval = CVodeWFtolerances(cvode_mem, ewt);
      TORCH_CHECK(retval == CV_SUCCESS, "CVodeWFtolerances failed");
      long int mxsteps = 1000; // or some larger number

      retval = CVodeSetMaxNumSteps(cvode_mem, mxsteps);
      TORCH_CHECK(retval == CV_SUCCESS, "CVodeSetMaxNumSteps failed");
      /* Attach user data */
      retval = CVodeSetUserData(cvode_mem, &data);
      // if (check_retval(&retval, "CVodeSetUserData", 1)) { return {}; }
      TORCH_CHECK(retval == CV_SUCCESS, "CVodeSetUserData failed");
      /* Create dense SUNMatrix for use in linear solves */
      auto J = SUNDenseMatrix(NP, NP, sunctx);
      /* Create another dense matrix for the sensitivites*/
      // if (check_retval((void*)A, "SUNDenseMatrix", 0)) { return {}; }
      TORCH_CHECK(J != NULL, "SUNDenseMatrix failed");
      /* Create dense SUNLinearSolver object */
      SUNLinearSolver LS = SUNLinSol_Dense(y, J, sunctx);
      // if (check_retval((void*)LS, "SUNLinSol_Dense", 0)) { return {}; }
      TORCH_CHECK(LS != NULL, "SUNLinSol_Dense failed");
      /* Attach the matrix and linear solver */
      retval = CVodeSetLinearSolver(cvode_mem, LS, J);
      TORCH_CHECK(retval == CV_SUCCESS, "CVodeSetLinearSolver failed");
      /* Set the user-supplied Jacobian routine Jac */
      retval = CVodeSetJacFn(cvode_mem, Jac);
      TORCH_CHECK(retval == CV_SUCCESS, "CVodeSetJacFn failed");
      tres.index_put_({0}, 0.0);
      yres.index_put_({0, 0}, x10);
      yres.index_put_({0, 1}, x20);
      /* Initialize y */
      
      for (int i = 0; i < Nt; i++)
      {
        

        sunrealtype TF = (i + 1) * dt; // Final time
        sunrealtype T = (i)*dt;



        // Take a signle step forward in time
        retval = CVode(cvode_mem, TF, y, &T, CV_NORMAL);
        tres.index_put_({i + 1}, T);
        yres.index_put_({i + 1, 0}, Ith(y, 1));
        yres.index_put_({i + 1, 1}, Ith(y, 2));
        // populate the time and the data
      }
      return std::make_tuple(tres, yres); // This now contains the data.
    }

  } // namespace vdp
} // namespace u2d


#endif