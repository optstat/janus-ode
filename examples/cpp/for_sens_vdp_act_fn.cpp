#include <torch/torch.h>
#include <iostream>
#include <cvodes/cvodes.h> /* prototypes for CVODES fcts., consts. */
#include <math.h>
#include <nvector/nvector_serial.h> /* access to serial N_Vector            */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */

#define Ith(v, i) NV_Ith_S(v, i - 1) /* i-th vector component i=1..NEQ */
#define IJth(A, i, j) \
  SM_ELEMENT_D(A, i - 1, j - 1) /* (i,j)-th matrix component i,j=1..NEQ */

/* Precision specific math function macros */

#if defined(SUNDIALS_DOUBLE_PRECISION)
#define ABS(x) (fabs((x)))
#elif defined(SUNDIALS_SINGLE_PRECISION)
#define ABS(x) (fabsf((x)))
#elif defined(SUNDIALS_EXTENDED_PRECISION)
#define ABS(x) (fabsl((x)))
#endif

/* Problem Constants */

#define NEQ   3               /* number of equations  */
#define Y1    SUN_RCONST(2.0) /* initial y components */
#define Y2    SUN_RCONST(2.0)
#define Y3    SUN_RCONST(1000.0) //This is mu
#define RTOL  SUN_RCONST(1.0e-4) /* scalar relative tolerance            */
#define ATOL1 SUN_RCONST(1.0e-6) /* vector absolute tolerance components */
#define ATOL2 SUN_RCONST(1.0e-6)
#define ATOL3 SUN_RCONST(1.0e-6)
#define T0    SUN_RCONST(0.0)  /* initial time           */
#define T1    SUN_RCONST(1.0)  /* first output time      */
#define TMULT SUN_RCONST(1000.0) /* output time factor     */
#define NOUT  2               

#define NP 3 /* number of problem parameters */
#define NS 3 /* number of sensitivities computed */

#define ZERO SUN_RCONST(0.0)
#define ONE  SUN_RCONST(1.0)

/* Type : UserData */

typedef struct
{
  sunrealtype p[3]; /* problem parameters */
}* UserData;

/* Functions Called by the Solver */

static int f(sunrealtype t, N_Vector y, N_Vector ydot, void* user_data);

static int Jac(sunrealtype t, N_Vector y, N_Vector fy, SUNMatrix J,
               void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3);

static int fS(int Ns, sunrealtype t, N_Vector y, N_Vector ydot, int iS,
              N_Vector yS, N_Vector ySdot, void* user_data, N_Vector tmp1,
              N_Vector tmp2);

static int ewt(N_Vector y, N_Vector w, void* user_data);


/* Prototypes of private functions */

static void ProcessArgs(int argc, char* argv[], sunbooleantype* sensi,
                        int* sensi_meth, sunbooleantype* err_con);

static void WrongArgs(char* name);

/* Private function to check function return values */

static int check_retval(void* returnvalue, const char* funcname, int opt);


/*
  *-------------------------------
  * Functions called by the solver
  *-------------------------------
  */
 
 /*
  * f routine. Compute function f(t,y).
  */
 
  static int f(sunrealtype t, //Current time
               N_Vector y,    //Current state
               N_Vector ydot, //Current state derivative
               void* user_data) //UserData
  {
    sunrealtype y1, y2, y3, yd1, yd2, yd3;
    UserData data;
    sunrealtype p1, p2, p3;
  
    y1   = Ith(y, 1);
    y2   = Ith(y, 2);
    y3   = Ith(y, 3);
    data = (UserData)user_data;
  
    yd1 = Ith(ydot, 1) = y2;
    yd2 = Ith(ydot, 2) = y3*(1-y1*y1)*y2-y1;
    yd3 = Ith(ydot, 3) = 0.0;
  
    return (0);
  }
  
  /*
   * Jacobian routine. Compute J(t,y) = df/dy. *
   */
  
  static int Jac(sunrealtype t, 
                 N_Vector y, 
                 N_Vector fy, 
                 SUNMatrix J,
                 void* user_data, 
                 N_Vector tmp1, 
                 N_Vector tmp2, 
                 N_Vector tmp3)
  {
    sunrealtype y1, y2, y3;
    UserData data;
  
    y1   = Ith(y, 1);
    y2   = Ith(y, 2);
    y3   = Ith(y, 3);
    data = (UserData)user_data;
    //dx1/dt = x2
    //dx2/dt = x3*(1-x1^2)*x2 - x1
    //dx3/dt = 0

    //This can be replaced with autograd from libtorch if needed
    IJth(J, 1, 1) = 0.0;
    IJth(J, 1, 2) = 1.0;
    IJth(J, 1, 3) = 0.0;
    IJth(J, 2, 1) = -2.0 * y1 * y3 * y2 - 1.0;
    IJth(J, 2, 2) = y3 * (1.0 - y1 * y1);
    IJth(J, 2, 3) = (1.0 - y1 * y1) * y2;
 
  
    return (0);
  }
  
  /*
   * fS routine. Compute sensitivity r.h.s.
   * Ns is the number of sensitivities computed.
   * t is the current value of the independent variable.
   * y is the current value of the dependent variable vector.
   * ydot is the current value of the derivative vector.
   * iS is the index of the sensitivity.
   * yS is the current value of the sensitivity vector.
   * ySdot is the output sensitivity derivative vector.
   * user_data is a pointer to user data - the UserData structure.
   * tmp1, tmp2, tmp3 are pointers to vectors of length NEQ.
   */
  static int fS(int Ns,  //Number of sensitivities 
                sunrealtype t,  //Current time
                N_Vector y, //Current state
                N_Vector ydot,  //Current state derivative
                int iS, //Index of the sensitivity
                N_Vector yS,  //Current sensitivity
                N_Vector ySdot,  //Output sensitivity derivative
                void* user_data, //UserData
                N_Vector tmp1, //Temporary vector1
                N_Vector tmp2) //Temporary vector2
  {
    UserData data;
    sunrealtype p1, p2, p3;
    sunrealtype y1, y2, y3;
    sunrealtype s1, s2, s3;
    sunrealtype sd1, sd2, sd3;
  
    data = (UserData)user_data;
    p1   = data->p[0];
    p2   = data->p[1];
    p3   = data->p[2];
  
    y1 = Ith(y, 1);
    y2 = Ith(y, 2);
    y3 = Ith(y, 3);
    
    s1 = Ith(yS, 1); // ∂y1 / ∂p_(iS) 
    s2 = Ith(yS, 2); // ∂y2 / ∂p_(iS)
    s3 = Ith(yS, 3); // ∂y3 / ∂p_(iS)
 
    //The sensitivity is with respect to the initial conditions
    //so there the calculations column wise are the same in all directions
    //so the iS is not used in the calculation
    sd1 = s2;
    sd2 = (-2.0*y1*y3*y2-1)*s1 + y3*(1-y1*y1)*s2 + (1-y1*y1)*y2*s3;
    sd3 = 0.0;
 
    
 
    //This is done column wise the function provides a row from the jacobian
    //and a column from the sensitivity matrix which need to be multiplied
  
    Ith(ySdot, 1) = sd1;
    Ith(ySdot, 2) = sd2;
    Ith(ySdot, 3) = sd3;
  
    return (0);
  }
  

  /*
   * EwtSet function. Computes the error weights at the current solution.
   */
  static int ewt(N_Vector y, N_Vector w, void* user_data)
  {
    int i;
    sunrealtype yy, ww, rtol, atol[3];
  
    rtol    = RTOL;
    atol[0] = ATOL1;
    atol[1] = ATOL2;
    atol[2] = ATOL3;
  
    for (i = 1; i <= 3; i++)
    {
      yy = Ith(y, i);
      ww = rtol * ABS(yy) + atol[i - 1];
      if (ww <= 0.0) { return (-1); }
      Ith(w, i) = 1.0 / ww;
    }
    return (0);
  }
  


// 1) Define a custom autograd function
struct MyCustomActivationFunction 
    : public torch::autograd::Function<MyCustomActivationFunction> 
{
    // Forward pass: from input to output
    static torch::autograd::variable_list forward(
        torch::autograd::AutogradContext *ctx, 
        torch::autograd::variable_list& inputs
    ) {
      UserData data;
      SUNContext sunctx;
      sunrealtype pbar[NS];
      sunrealtype t, tout;
      sunbooleantype sensi, err_con;
      auto input = inputs[0];
      auto ft = inputs[1];
      int M = input.size(0); //Sample size
      int retval, iout;
      auto output = torch::zeros_like(input);
      auto grads = torch::zeros_like(input);
      N_Vector* yS;
      for (int i = 0; i < M; i++) {
          double y1 = input[i][0].item<double>();
          double y2 = input[i][1].item<double>();
          double y3 = input[i][2].item<double>(); //This is the stiffness parameter mu
          //Sensitivities
          double p1 = 1.0;
          double p2 = 1.0;
          double p3 = 1.0;
          //propagate the sensitivities
          double tf = ft.item<double>();  //Final time
          double t = 0.0;
          /* User data structure */
          data = (UserData)malloc(sizeof *data);
          TORCH_CHECK(data != NULL, "malloc failed");

   
          /* Initialize sensitivity variables (Initial conditions) */
          data->p[0] = SUN_RCONST(1.0);
          data->p[1] = SUN_RCONST(1.0);
          data->p[2] = SUN_RCONST(1.0);
   
          /* Create the SUNDIALS context that all SUNDIALS objects require */
          auto retval = SUNContext_Create(SUN_COMM_NULL, &sunctx);
          TORCH_CHECK(sunctx != NULL, "SUNContext_Create failed");
   
          /* Initial conditions */
          auto y = N_VNew_Serial(NEQ, sunctx);
          //if (check_retval((void*)y, "N_VNew_Serial", 0)) { return {}; }
          TORCH_CHECK(y != NULL, "N_VNew_Serial failed");
   
          /* Initialize y */
          Ith(y, 1) = y1;
          Ith(y, 2) = y2;
          Ith(y, 3) = y3;
   
          /* Call CVodeCreate to create the solver memory and specify the
           * Backward Differentiation Formula */
          auto  cvode_mem = CVodeCreate(CV_BDF, sunctx);
          //if (check_retval((void*)cvode_mem, "CVodeCreate", 0)) { return {}; }
          TORCH_CHECK(cvode_mem != NULL, "CVodeCreate failed");
          /* Call CVodeInit to initialize the integrator memory and specify the
           * user's right hand side function in y'=f(t,y), the initial time T0, and
           * the initial dependent variable vector y. */
          retval = CVodeInit(cvode_mem, f, T0, y);
          //if (check_retval(&retval, "CVodeInit", 1)) { return {}; }
          TORCH_CHECK(retval == CV_SUCCESS, "CVodeInit failed");
          /* Call CVodeWFtolerances to specify a user-supplied function ewt that sets
           * the multiplicative error weights w_i for use in the weighted RMS norm */
          retval = CVodeWFtolerances(cvode_mem, ewt);
          //if (check_retval(&retval, "CVodeWFtolerances", 1)) { return {}; }
          TORCH_CHECK(retval == CV_SUCCESS, "CVodeWFtolerances failed");
          long int mxsteps = 1000;   // or some larger number
          retval = CVodeSetMaxNumSteps(cvode_mem, mxsteps);
          //if (check_retval(&retval, "CVodeSetMaxNumSteps", 1)) { return {}; }
          TORCH_CHECK(retval == CV_SUCCESS, "CVodeSetMaxNumSteps failed");
          /* Attach user data */
          retval = CVodeSetUserData(cvode_mem, data);
          //if (check_retval(&retval, "CVodeSetUserData", 1)) { return {}; }
          TORCH_CHECK(retval == CV_SUCCESS, "CVodeSetUserData failed");
          /* Create dense SUNMatrix for use in linear solves */
          auto A = SUNDenseMatrix(NEQ, NEQ, sunctx);
          //if (check_retval((void*)A, "SUNDenseMatrix", 0)) { return {}; }
          TORCH_CHECK(A != NULL, "SUNDenseMatrix failed"); 
          /* Create dense SUNLinearSolver object */
          auto LS = SUNLinSol_Dense(y, A, sunctx);
          //if (check_retval((void*)LS, "SUNLinSol_Dense", 0)) { return {}; }
          TORCH_CHECK(LS != NULL, "SUNLinSol_Dense failed"); 
          /* Attach the matrix and linear solver */
          retval = CVodeSetLinearSolver(cvode_mem, LS, A);
          //if (check_retval(&retval, "CVodeSetLinearSolver", 1)) { return {}; }
          TORCH_CHECK(retval == CV_SUCCESS, "CVodeSetLinearSolver failed");
          /* Set the user-supplied Jacobian routine Jac */
          retval = CVodeSetJacFn(cvode_mem, Jac);
          //if (check_retval(&retval, "CVodeSetJacFn", 1)) { return {}; }
          TORCH_CHECK(retval == CV_SUCCESS, "CVodeSetJacFn failed");
   
           /* Set parameter scaling factor */
           pbar[0] = data->p[0];
           pbar[1] = data->p[1];
           pbar[2] = data->p[2];
   
           /* Set sensitivity initial conditions */
           yS = N_VCloneVectorArray(NS, y);
           //if (check_retval((void*)yS, "N_VCloneVectorArray", 0)) { return {}; }
           TORCH_CHECK(yS != NULL, "N_VCloneVectorArray failed");
           for (int is = 0; is < NS; is++) { N_VConst(ONE, yS[is]); }
   
         /* Call CVodeSensInit1 to activate forward sensitivity computations
          * and allocate internal memory for COVEDS related to sensitivity
          * calculations. Computes the right-hand sides of the sensitivity
          * ODE, one at a time */
         void *vcode_mem;
         int sensi_meth = CV_STAGGERED1;
         retval = CVodeSensInit1(cvode_mem, NS, sensi_meth, fS, yS);
         //if (check_retval(&retval, "CVodeSensInit", 1)) { return {}; }
          TORCH_CHECK(retval == CV_SUCCESS, "CVodeSensInit failed");
   
         /* Call CVodeSensEEtolerances to estimate tolerances for sensitivity
          * variables based on the rolerances supplied for states variables and
          * the scaling factor pbar */
         retval = CVodeSensEEtolerances(cvode_mem);
         //if (check_retval(&retval, "CVodeSensEEtolerances", 1)) { return {}; }
          TORCH_CHECK(retval == CV_SUCCESS, "CVodeSensEEtolerances failed");
   
         /* Set sensitivity analysis optional inputs */
         /* Call CVodeSetSensErrCon to specify the error control strategy for
          * sensitivity variables */
         retval = CVodeSetSensErrCon(cvode_mem, err_con);
         //if (check_retval(&retval, "CVodeSetSensErrCon", 1)) { return {}; }
          TORCH_CHECK(retval == CV_SUCCESS, "CVodeSetSensErrCon failed");
   
         /* Call CVodeSetSensParams to specify problem parameter information for
          * sensitivity calculations */
         retval = CVodeSetSensParams(cvode_mem, NULL, pbar, NULL);
         //if (check_retval(&retval, "CVodeSetSensParams", 1)) { return {}; }
          TORCH_CHECK(retval == CV_SUCCESS, "CVodeSetSensParams failed");
   
   
   
         retval = CVode(cvode_mem, tout, y, &t, CV_NORMAL);
         //if (check_retval(&retval, "CVode", 1)) { break; }
          TORCH_CHECK(retval == CV_SUCCESS, "CVode failed");
   
         retval = CVodeGetSens(cvode_mem, &t, yS);
         //if (check_retval(&retval, "CVodeGetSens", 1)) { break; }
          TORCH_CHECK(retval == CV_SUCCESS, "CVodeGetSens failed");
         output.index_put_({i, 0}, y1);
         output.index_put_({i, 1}, y2);
         output.index_put_({i, 2}, y3);
         //Cast the ys to torch tensors
         sunrealtype ySdata[3];
         for (int i = 0; i < 3; i++) {
              ySdata[i] = Ith(yS[0], i + 1);
         }
         grads.index_put_({i, 0}, ySdata[0]);
         grads.index_put_({i, 1}, ySdata[1]);
         grads.index_put_({i, 2}, ySdata[2]);
     }
     //We have to convert to torch tensors
     


     // Save the grads needed for backward
     ctx->save_for_backward({grads});
     delete data;

     return {output};

    }

    // Backward pass: from grad_output to grad_input
    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext *ctx,
        std::vector<at::Tensor>& grad_output
    ) {
        // Retrieve saved data
        auto grads = ctx->get_saved_variables();
        return grads;
    }
};


