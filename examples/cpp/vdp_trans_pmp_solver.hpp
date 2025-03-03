#ifndef VDP_TRANS_PMP_SOLVER
#define VDP
#include "../../src/cpp/janus_ode_common.hpp"
#endif

/*
  /* Type : UserData contains the DNN data*/

  typedef struct
  {
    DynamicsNN model;
  } *UserData;

  /* Functions Called by the Solver */

  static int f(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data);

  static int Jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                 void *user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

  static int fS(int Ns, sunrealtype t, N_Vector y, N_Vector ydot, int iS,
                N_Vector yS, N_Vector ySdot, void *user_data, N_Vector tmp1,
                N_Vector tmp2);

  static int ewt(N_Vector y, N_Vector w, void *user_data);


// Compute Jacobian df/dx using automatic differentiation
torch::Tensor compute_jacobian(DynamicsNN& model, torch::Tensor x) {
  torch::Tensor f = model.forward(x);  // Forward pass
  int dim = x.size(1);  // Assuming batch size is first dimension

  torch::Tensor jacobian = torch::zeros({dim, dim});
  
  for (int i = 0; i < dim; ++i) {
      // Create a tensor with gradients only for the i-th output component
      torch::Tensor grad_outputs = torch::zeros_like(f);
      grad_outputs[0][i] = 1.0;  // Backpropagate only for one output dimension at a time

      // Compute gradients
      torch::Tensor grad = torch::autograd::grad({f}, {x}, {grad_outputs}, /*retain_graph=*/true, /*create_graph=*/true)[0];

      jacobian.slice(0, i, i + 1) = grad[0];  // Extract row
  }

  return jacobian;
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
    sunrealtype y1, y2;
    sunrealtype s1, s2, s3, s4;
    sunrealtype sd1, sd2, sd3, sd4;
    SUNContext sunctx;
    sunrealtype pbar[NS];
    sunrealtype t, tout;
    sunbooleantype sensi, err_con;
    SUNLinearSolver LS;
    N_Vector y;

    data = (UserData)user_data;
    /*Calculate the local Jacobian */
    Jac(t, y, ydot, J, data);



    // We are going to propagate the costates as sensitivities
    // reusing the staggered method
    //This is simply the jacobian * the sensitivity matrix
    sd1 = 0.0;
    sd2 = 0.0;
    sd3 = 0.0;
    sd4 = 0.0;
    for ( int i=0; i < NP; i++ ) {
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
    sunrealtype yy, ww, rtol, atol[3];

    rtol = RTOL;
    atol[0] = ATOL1;
    atol[1] = ATOL2;
    atol[2] = ATOL3;

    for (i = 1; i <= 3; i++)
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
  
   
  void init() {
    /* Create the SUNDIALS context that all SUNDIALS objects require */
    auto retval = SUNContext_Create(SUN_COMM_NULL, &sunctx);
    TORCH_CHECK(sunctx != NULL, "SUNContext_Create failed");
  
    y = N_VNew_Serial(NP, sunctx);
    TORCH_CHECK(y != NULL, "N_VNew_Serial failed");
    /* Call CVodeCreate to create the solver memory and specify the
     * Backward Differentiation Formula */
    cvode_mem = CVodeCreate(CV_BDF, sunctx);
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
    retval = CVodeSetUserData(cvode_mem, data);
    // if (check_retval(&retval, "CVodeSetUserData", 1)) { return {}; }
    TORCH_CHECK(retval == CV_SUCCESS, "CVodeSetUserData failed");
    /* Create dense SUNMatrix for use in linear solves */
    J = SUNDenseMatrix(NEQ, NEQ, sunctx);
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

  // 1) Define a custom autograd function
  struct VdpPropagate
      : public torch::autograd::Function<VdpPropagate>
  {
    // Forward pass: from input to output
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx,
        torch::autograd::variable_list &inputs)  //The inputs here constitute the initial state, initial costates and the final time
    {
      //The Initial costates are first
      auto ls = inputs[0];
      //The states are next
      auto xs = inputs[1];
      //The controls here are fed in one at a time
      //Which means the activation function can only take step
      auto us = inputs[2];
      //The final time this should be a scalar and there is no need to backpropagate
      auto ft = inputs[3];

      //Save the inputs for the backward pass
      ctx->save_for_backward({ls, xs, us, ft});

      int  M  = input.size(0); // Sample size
      int retval, iout;
      auto outputl  = torch::zeros_like(inputs[0]);
      auto outputx  = torch::zeros_like(inputs[1]);
      auto outputt  = torch::zeros_like(inputs[2]);
      //Initialize the solver if it has not been initialized
      if (!init)
      {
        init();
        init = true;
      }

      // Loop over the batch dimension
      // This should be done in parallel
      // but let's postpone this
      for (int i = 0; i < M; i++)
      {
        //Initialize the states
        sunrealtype y1 = ys[i][0].item<double>();
        sunrealtype y2 = ys[i][1].item<double>();
        //Initialize the costates
        sunrealtype l1 = ls[i][0].item<double>();
        sunrealtype l2 = ls[i][1].item<double>();

        sunrealtype TF = ft.item<double>(); // Final time
        sunrealtype TO = ZERO;



        /* Initialize y */
        Ith(y, 1) = l1;
        Ith(y, 2) = l2;
        Ith(y, 3) = y1;
        Ith(y, 4) = y2;

        /* Set the control */
        data->u = us[i].item<double>();
        //Take a signle step forward in time
        retval = CVode(cvode_mem, tout, y, &t, CV_ONE_STEP);
        // if (check_retval(&retval, "CVode", 1)) { break; }
        TORCH_CHECK(retval == CV_SUCCESS, "CVode failed");

        // We are only interested in the final state
        l1 = Ith(y, 1);
        l2 = Ith(y, 2);
        x1 = Ith(y, 3);
        x2 = Ith(y, 4);
        
        // Put the states in the output
        outputl.index_put_({i, 0}, l1);
        outputl.index_put_({i, 1}, l2);
        // Put the costates in the output
        outputx.index_put_({i, 0}, x1);
        outputx.index_put_({i, 1}, x2);
        
        outputt.index_put_({i}, t);
        
        // Cast the ys to torch tensors
        sunrealtype ySdata[4];
        for (int i = 0; i < 4; i++)
        {
          ySdata[i] = Ith(yS[0], i + 1);
        }
        gradsl.index_put_({i, 0}, ySdata[0]);
        gradsl.index_put_({i, 1}, ySdata[1]);
        gradsx.index_put_({i, 2}, ySdata[2]);
        gradsx.index_put_({i, 3}, ySdata[3]);
        
      }
      // We have to convert to torch tensors

      // Save the grads needed for backward
      ctx->save_for_backward({grads});
      free data;

      return {output};
    }

    // Backward pass: from grad_output to grad_input
    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext *ctx,
        std::vector<at::Tensor> &grad_output)
    {
      // Retrieve saved data
      auto saved = ctx->get_saved_variables();
      auto ls = saved[0];
      auto xs = saved[1];
      auto us = saved[2];
      auto ft = saved[3];
      //The gradients here are wrt ls, us only because the states and time are not parameters
      auto grads = grad_output[0];
      auto gradsl = torch::zeros_like(ls);
      auto gradsu = torch::zeros_like(us);
      //We are going to calculate jacobian vector products 
      //in the reverse to save memory
      int M = ls.size(0);
      // Loop over the batch dimension
      // This should be done in parallel
      // but let's postpone this
      for (int i = 0; i < M; i++)
      {
        //Initialize the states
        sunrealtype y1 = ys[i][0].item<double>();
        sunrealtype y2 = ys[i][1].item<double>();
        //Initialize the costates
        sunrealtype l1 = ls[i][0].item<double>();
        sunrealtype l2 = ls[i][1].item<double>();

        sunrealtype TF = ft.item<double>(); // Final time
        sunrealtype TO = ZERO;



        /* Initialize y */
        Ith(y, 1) = l1;
        Ith(y, 2) = l2;
        Ith(y, 3) = y1;
        Ith(y, 4) = y2;

        /* Set the control */
        data->u = us[i].item<double>();



        /* Set parameter scaling factor */
        pbar[0] = ONE;
        pbar[1] = ONE;
        pbar[2] = ONE;
        pbar[3] = ONE;

        /* Set sensitivity initial conditions */
        yS = N_VCloneVectorArray(NS, y);
        // if (check_retval((void*)yS, "N_VCloneVectorArray", 0)) { return {}; }
        TORCH_CHECK(yS != NULL, "N_VCloneVectorArray failed");
        //Initialize the sensitivity
        for (int is = 0; is < NS; is++)
        {
          N_VConst(ONE, yS[is]);
        }

        /* Call CVodeSensInit1 to activate forward sensitivity computations
         * and allocate internal memory for COVEDS related to sensitivity
         * calculations. Computes the right-hand sides of the sensitivity
         * ODE, one at a time */
        void *vcode_mem;
        int sensi_meth = CV_STAGGERED1;
        retval = CVodeSensInit1(cvode_mem, NS, sensi_meth, fS, yS);
        // if (check_retval(&retval, "CVodeSensInit", 1)) { return {}; }
        TORCH_CHECK(retval == CV_SUCCESS, "CVodeSensInit failed");

        /* Call CVodeSensEEtolerances to estimate tolerances for sensitivity
         * variables based on the rolerances supplied for states variables and
         * the scaling factor pbar */
        retval = CVodeSensEEtolerances(cvode_mem);
        // if (check_retval(&retval, "CVodeSensEEtolerances", 1)) { return {}; }
        TORCH_CHECK(retval == CV_SUCCESS, "CVodeSensEEtolerances failed");

        /* Set sensitivity analysis optional inputs */
        /* Call CVodeSetSensErrCon to specify the error control strategy for
         * sensitivity variables */
        retval = CVodeSetSensErrCon(cvode_mem, err_con);
        // if (check_retval(&retval, "CVodeSetSensErrCon", 1)) { return {}; }
        TORCH_CHECK(retval == CV_SUCCESS, "CVodeSetSensErrCon failed");

        /* Call CVodeSetSensParams to specify problem parameter information for
         * sensitivity calculations */
        retval = CVodeSetSensParams(cvode_mem, NULL, pbar, NULL);
        // if (check_retval(&retval, "CVodeSetSensParams", 1)) { return {}; }
        TORCH_CHECK(retval == CV_SUCCESS, "CVodeSetSensParams failed");
        retval = CVode(cvode_mem, tout, y, &t, CV_ONE_STEP);
        // if (check_retval(&retval, "CVode", 1)) { break; }
        TORCH_CHECK(retval == CV_SUCCESS, "CVode failed");

        retval = CVodeGetSens(cvode_mem, &t, yS);
        // if (check_retval(&retval, "CVodeGetSens", 1)) { break; }
        TORCH_CHECK(retval == CV_SUCCESS, "CVodeGetSens failed");

        // We are only interested in the final state
        l1 = Ith(y, 1);
        l2 = Ith(y, 2);
        x1 = Ith(y, 3);
        x2 = Ith(y, 4);
        
        // Put the states in the output
        outputl.index_put_({i, 0}, l1);
        outputl.index_put_({i, 1}, l2);
        // Put the costates in the output
        outputx.index_put_({i, 0}, x1);
        outputx.index_put_({i, 1}, x2);

        outputt.index_put_({i}, t);
        
        // Cast the ys to torch tensors
        sunrealtype ySdata[4];
        for (int i = 0; i < 4; i++)
        {
          ySdata[i] = Ith(yS[0], i + 1);
        }
        gradsl.index_put_({i, 0}, ySdata[0]);
        gradsl.index_put_({i, 1}, ySdata[1]);
        gradsx.index_put_({i, 2}, ySdata[2]);
        gradsx.index_put_({i, 3}, ySdata[3]);
        
      }

      auto grads = ctx->get_saved_variables();
      return grads;
    }
  };
