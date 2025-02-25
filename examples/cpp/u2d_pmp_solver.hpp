#ifndef U2D_PMP_SOLVER_HPP
#define U2D_PMP_SOLVER_HPP
#include "../../src/cpp/janus_ode_common.hpp"
/**
 * This current example uses forward mode staggered sensitivity analysis
 * to propagate the costates and utilizes a control method to learn the vdp system
 * from data.
 * This is control centered method to do machine learning on the vdp system by generating the
 * dynamics as data.
 * It can learn any 2D
 */
namespace u2d
{
  /**
   * Forward only PMP solution
   */
  namespace vdpforpmp
  {

    int NS = 4; //Augmented system states plus costates
    sunrealtype RTOL = SUN_RCONST(1.0e-6);  /* scalar relative tolerance            */
    sunrealtype ATOL1 = SUN_RCONST(1.0e-8); /* vector absolute tolerance components */
    sunrealtype ATOL2 = SUN_RCONST(1.0e-8);
    sunrealtype ATOL3 = SUN_RCONST(1.0e-8);
    sunrealtype ATOL4 = SUN_RCONST(1.0e-8);
    sunrealtype W = SUN_RCONST(1.0e-4); // This is regularization on the control
    static SUNMatrix J;
    SUNContext sunctx;
    static N_Vector y;
    static SUNLinearSolver LS;

    //Global initialization of the CVODES and KINSOL solvers
    
    typedef struct
    {
      sunrealtype mu; /* Initial conditions */
    } UserData;


    /* Functions Called by the Solver */

  static int f(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data);

  static int Jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                 void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

  static int fS(int Ns, sunrealtype t, N_Vector y, N_Vector ydot, int iS,
                N_Vector yS, N_Vector ySdot, void *user_data, N_Vector tmp1,
                N_Vector tmp2);

  static int ewt(N_Vector y, N_Vector w, void *user_data);
  

  /**
   * Forward PMP method for the VDP oscillator
   * Method takes in the initial costates and initial state conditions
   * and returns the optimal final costates and state conditions
   * together with the sensitivity Jacobian using the low memory 
   * staggered forward sensitivity method in CVODES
   */
  std::tuple<torch::Tensor, torch::Tensor> pmp_sens(torch::Tensor x0in,
                            torch::Tensor x0fin,
                            torch::Tensor t0in,
                            torch::Tensor ftin,
                            int Nt);

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
      sunrealtype l1, l2, x1, x2, xd1, xd2, xd3, xd4;
      l1 = Ith(y, 1);
      l2 = Ith(y, 2);
      x1 = Ith(y, 3);
      x2 = Ith(y, 4);

      auto u1 = -l1/W;
      auto u2 = -l2/W;
      //The costates are simple
      xd1 = Ith(ydot, 1) = u1;
      xd2 = Ith(ydot, 2) = u2;
      xd3 = Ith(ydot, 3) = x2;
      xd4 = Ith(ydot, 4) = mu * (1 - x1 * x1) * x2 - x1;
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
sunrealtype l1, l2, x1, x2;
l1 = Ith(y, 1);
l2 = Ith(y, 2);
x1 = Ith(y, 3);
x2 = Ith(y, 4);
UserData data = *(UserData *)user_data;
sunrealtype mu = data.mu;

IJth(J, 1, 1) = -l1/W;
IJth(J, 1, 2) = 0.0;
IJth(J, 1, 3) = 0.0;
IJth(J, 1, 4) = 0.0;
IJth(J, 2, 1) = 0.0;
IJth(J, 2, 2) = -l2/W;
IJth(J, 2, 3) = 0.0;
IJth(J, 2, 4) = 0.0;
IJth(J, 3, 1) = 0.0;
IJth(J, 3, 2) = 0.0;
IJth(J, 3, 3) = 0.0;
IJth(J, 3, 4) = 1.0;
IJth(J, 4, 1) = 0.0;
IJth(J, 4, 2) = 0.0;
IJth(J, 4, 3) = -2 * mu * x1 * x2 - 1.0;
IJth(J, 4, 4) = mu * (1 - x1 * x1);
return (0);
}



  /*
   * fS routine. Compute costate r.h.s.
   * Ns is the number of costates computed.
   * t is the current value of the independent variable.
   * y is the current value of the dependent variable vector.
   * ydot is the current value of the derivative vector.
   * iS is the index of the sensitivity.
   * yS is the current value of the sensitivity vector.
   * ySdot is the output sensitivity derivative vector.
   * user_data is a pointer to user data - the UserData structure.
   * tmp1, tmp2, tmp3 are pointers to vectors of length NEQ.
   */
  static int fS(int Ns,          // Number of sensitivities
                sunrealtype t,   // Current time
                N_Vector y,      // Current data
                N_Vector ydot,   // Current derivative
                int iS,          // Index of the sensitivity
                N_Vector yS,     // Current sensitivity
                N_Vector ySdot,  // Output sensitivity derivative
                void *user_data, // UserData
                N_Vector tmp1,   // Temporary vector1
                N_Vector tmp2)   // Temporary vector2
  {
    UserData data;
    sunrealtype l1, l2;
    sunrealtype x1, x2;
    sunrealtype s1, s2, s3, s4;
    sunrealtype sd1, sd2, sd3, sd4;
    SUNContext sunctx;
    sunrealtype pbar[NS];
    sunrealtype t, tout;
    sunbooleantype sensi, err_con;
    SUNLinearSolver LS;
    N_Vector y, tmp1, tmp2, tmp3;

    data = *(UserData*)user_data;
    sunrealtype mu = data.mu;
    /*Calculate the local Jacobian */
    Jac(t, y, ydot, J, &data, tmp1, tmp2, tmp3);



    // We are going to propagate the costates as sensitivities
    // reusing the staggered method
    //This is simply the jacobian * the sensitivity matrix
    sd1 = 0.0;
    sd2 = 0.0;
    sd3 = 0.0;
    sd4 = 0.0;
    for ( int i=0; i < NS; i++ ) {
      sd1 += IJth(J, 1, i+1) * Ith(yS, i+1);
      sd2 += IJth(J, 2, i+1) * Ith(yS, i+1);
      sd3 += IJth(J, 3, i+1) * Ith(yS, i+1);
      sd4 += IJth(J, 4, i+1) * Ith(yS, i+1);
    }

    // This is done column wise the function provides a row from the jacobian
    // and a column from the sensitivity matrix which need to be multiplied

    Ith(ySdot, 1) = sd1;
    Ith(ySdot, 2) = sd2;
    Ith(ySdot, 3) = sd3;
    Ith(ySdot, 4) = sd4;

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
   * Solves the augmented PMP in the forward direction
   * and returns F(x) and the sensitivity jacobian partial F / partial y0
   * where y0 is the augmented costate and state vector
   * The Hamiltonian here is the simple universal control plus small regularization term
   * H = l1*u1 + l2*u2 + W/2 * (u1^2 + u2^2)
   * Here we have that
   * u1star = -l1/W
   * u2star = -l2/W
   * The dynamics are
   * \dot{l1} = -dH/dx1=0
   * \dot{l2} = -dH/dx2=0
   * \dot{x1} = u1star
   * \dot{x2} = u2star
   * \dot{t}  = 1
   * so we have a 5 x 5 sensitivy Jacobian that needs to be calculated
   * Since this is a minimum time problem we have
   * tf is the final time and it is part of the augmented state
   * The augmented state here is
   * yin = [l1, l2, x10, x20, tf]
   * The final augmented state is
   * yfin = [l1, l2, x1f, x2f, tf]
   * of which l1 l2 and tf are unknowns
   * We have the following boundary conditions
   * l1(tf) = l1(0)
   * l2(tf) = l2(0)
   * x1(tf) = x1f
   * x2(tf) = x2f
   * tf = tf
   */
  std::tuple<torch::Tensor, torch::Tensor> pmp_sens(torch::Tensor y0in, //Initial augmented state
                                                    torch::Tensor y0fin, //Final augmented state
                                                    torch::Tensor t0in, //Initial time
                                                    torch::Tensor ftin, //Final time
                                                    int Nt) {
    /* Create the SUNDIALS context that all SUNDIALS objects require */
    auto retval = SUNContext_Create(SUN_COMM_NULL, &sunctx);
    TORCH_CHECK(sunctx != NULL, "SUNContext_Create failed");
    sunrealtype T0 = t0in.item<double>();
    y = N_VNew_Serial(NS, sunctx);
    Ith(y, 1) = x0in[0].item<double>(); //l1
    Ith(y, 2) = x0in[1].item<double>(); //l2
    Ith(y, 3) = x0in[2].item<double>(); //x1
    Ith(y, 4) = x0in[3].item<double>(); //x2
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
    /* Attach user data */
    UserData data;
    data.mu = 100.0; // This is the parameter of the vdp system
    
    retval = CVodeSetUserData(cvode_mem, &data);
    // if (check_retval(&retval, "CVodeSetUserData", 1)) { return {}; }
    TORCH_CHECK(retval == CV_SUCCESS, "CVodeSetUserData failed");
    /* Create dense SUNMatrix for use in linear solves */
    J = SUNDenseMatrix(NS, NS, sunctx);
    /* Create another dense matrix for the sensitivites*/
    // if (check_retval((void*)A, "SUNDenseMatrix", 0)) { return {}; }
    TORCH_CHECK(J != NULL, "SUNDenseMatrix failed");
    /* Create dense SUNLinearSolver object */
    LS = SUNLinSol_Dense(y, J, sunctx);
    // if (check_retval((void*)LS, "SUNLinSol_Dense", 0)) { return {}; }
    TORCH_CHECK(LS != NULL, "SUNLinSol_Dense failed");
    /* Attach the matrix and linear solver */
    retval = CVodeSetLinearSolver(cvode_mem, LS, J);
    TORCH_CHECK(retval == CV_SUCCESS, "CVodeSetLinearSolver failed");
    /* Set the user-supplied Jacobian routine Jac */
    retval = CVodeSetJacFn(cvode_mem, Jac);
    TORCH_CHECK(retval == CV_SUCCESS, "CVodeSetJacFn failed");
  }

  std::tuple<torch::Tensor 


  } // namespace uvdp

} // namespace u2d
#endif