#include <petscts.h>
#include <petsctao.h>
#include <cassert>
#define W 0.01
#define umin 1.0
#define umax 1.0
#define eps 0.1

/*
 * The method uses a pseudo transient method to
 */
typedef struct
{
  TS tsInner; /* TS for the inner ODE system */
  Vec X;      /* solution vector for [p1, p2, p3, x1, x2, x3] */
  PetscInt stage; /* stage of the pseudo transient method */
  PetscScalar x1f, x2f, x3f, p1f, p2f, p3f; /* final states of the inner system */
  PetscScalar p10, p20, p30, x10, x20, x30; /* initial states of the inner system */
  TSTrajectory tj;
} AppCtx;

/*
  The time is replaced by a state variable x3
  where x3=log(t+\epsilon) and x3(0)=log(\epsilon) which corresponds to t=0
  Note that x3 can move into negative territory
  The time derivative of x3 is 1/(t+\epsilon)
  so dot{x3}=1/(t+\epsilon) but t+\epsilon = exp(x3) so dot{x3}=1/exp(x3)=exp(-x3)
  The Hamiltonian is given by
  H = p1*x2 + p2*(ustar*ustar*((1 - x1^2)*x2 - x1))) + p3*exp(-x3) + 1
  Evaluate ustar = umax or umin depending on the sign of p2*((1 - x1^2)*x2-x1).
  If p2*((1 - x1^2)*x2-1) < 0 => umax minimizes the Hamiltonian
  If p2*((1 - x1^2)*x2-1) >= 0 => umin minimizes the Hamiltonian
*/
static inline PetscReal ComputeUstar(const PetscReal p1,
                                     const PetscReal p2,
                                     const PetscReal p3,
                                     const PetscReal x1,
                                     const PetscReal x2,
                                     const PetscReal x3)
{
  PetscReal ustar;
  // H = p1*x2 + p2*(ustar*ustar*((1 - x1^2)*x2 - x1)) + p3*exp(-x3) + 1
  PetscReal factor = p2 * ((1.0 - x1 * x1) * x2-x1);
  if (factor < 0)
  {
    ustar = umax;
  }
  if (factor >= 0)
  {
    ustar = umin;
  }
  //printf("ustar=%g\n", (double)ustar);
  return ustar;
}

/* -------------------------------------------------------------------
   RHSFunction_Inner Explicit term:
   Defines the 6D ODE:
     X = [p1, p2, p3, x1, x2, x3]

   dot(p1) = -p2
   dot(p2) = p1
   dot(p3) = 0
   dot(x1) = x2
   dot(x2) = 0.0
   dot(x3) = 0.0
   ------------------------------------------------------------------- */
static PetscErrorCode RHSFunction_Inner(TS ts,
                                        PetscReal t,
                                        Vec X,
                                        Vec Xdot,
                                        void *ctx)
{
  PetscFunctionBeginUser;
  const PetscReal *x;
  PetscReal *xdot;

  PetscCall(VecGetArrayRead(X, &x)); /* x[0]=p1, x[1]=p2, x[2]=p3, x[3]=x1, x[4]=x2, x[5]=x3 */
  PetscCall(VecGetArray(Xdot, &xdot));
  PetscReal p1 = x[0];
  PetscReal p2 = x[1];
  PetscReal p3 = x[2];

  PetscReal x1 = x[3];
  PetscReal x2 = x[4];
  PetscReal x3 = x[5];
  // printf("Input in inner dynamics p1=%g, p2=%g, p3=%g, x1=%g, x2=%g, x3=%g\n", (double)p1, (double)p2, (double)p3, (double)x1, (double)x2, (double)x3);

  /* Evaluate ustar based on sign condition */
  AppCtx *actx = (AppCtx *)ctx;
  PetscReal ustar = ComputeUstar(p1, p2, p3, x1, x2, x3);
  // printf("ustar=%g\n", (double)ustar);
  // H = p1*x2 + p2*(ustar*ustar*((1 - x1^2)*x2 - x1)) + p3*exp(-x3) + 1
  /* dot(p1) */
  xdot[0] = 0.0;
  /* dot(p2) */
  xdot[1] = p1;
  /* dot(p3) = 0 */
  xdot[2] = 0.0;
  /* dot(x1) */
  xdot[3] = x2;
  /* dot(x2) */
  xdot[4] = 0.0;
  /* dot(x3) = 0 */
  xdot[5] = 0.0;

  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArray(Xdot, &xdot));
  // Print out the values of the terminal states
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------
   RHSFunction_Inner Implicit form:
   Defines the 6D ODE:
     X = [p1, p2, p3, x1, x2, x3]

   dot(p1) = -2*x1*p2*x2*u*u
   dot(p2) = p2*ustar*ustar*(1.0 - x1*x1)
   dot(p3) = 0
   dot(x1) = 0
   dot(x2) = ustar*ustar*((1.0 - x1*x1)*x2 - x1)
   dot(x3) = 0.0
   ------------------------------------------------------------------- */
static PetscErrorCode IRHSFunction_Inner(TS ts,
                                         PetscReal t,
                                         Vec U,
                                         Vec Udot,
                                         Vec F,
                                         void *ctx)
{
  PetscFunctionBeginUser;
  const PetscScalar *u, *udot;
  PetscScalar *f;

  /* Extract state variables */
  PetscCall(VecGetArrayRead(U, &u)); /* x[0]=p1, x[1]=p2, x[2]=p3, x[3]=x1, x[4]=x2, x[5]=x3 */
  PetscCall(VecGetArrayRead(Udot, &udot));
  PetscCall(VecGetArray(F, &f));

  PetscReal p1 = u[0];
  PetscReal p2 = u[1];
  PetscReal p3 = u[2];
  PetscReal x1 = u[3];
  PetscReal x2 = u[4];
  PetscReal x3 = u[5];

  /* Compute control input ustar */
  AppCtx *actx = (AppCtx *)ctx;
  PetscReal ustar = ComputeUstar(p1, p2, p3, x1, x2, x3);
  //H = p1*x2 + p2*(ustar*ustar*((1 - x1^2)*x2 - x1)) + p3*exp(-x3) + 1

  /* Compute residuals: f_i = udot[i] - g_i(x) */
  f[0] = udot[0] + 2.0 * x1 * p2 * x2 * ustar * ustar-p2*ustar*ustar;
  f[1] = udot[1] - p1-p2 * ustar * ustar * (1.0 - x1 * x1);
  f[2] = udot[2] + p3*exp(-x3); // No implicit dynamics for p3
  f[3] = udot[3]; // No implicit dynamics for x1
  f[4] = udot[4] - ustar * ustar * ((1.0 - x1 * x1) * x2 - x1);
  f[5] = udot[5] - exp(-x3); // No implicit dynamics for x3

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(Udot, &udot));
  PetscCall(VecRestoreArray(F, &f));

  PetscFunctionReturn(0);
}



// Create a Jacobian function for the inner Jacobian only
/*
    snes  - the SNES context
    x     - the current iterate at which to evaluate the Jacobian
    A,B   - the matrices to be filled with the Jacobian entries
    ctx   - (optional) user context
*/
// PetscErrorCode IJacobian(TS ts, PetscReal t, Vec U, Vec Udot, PetscReal shift, Mat A, Mat B, void *ctx)
static PetscErrorCode InnerJacobianFunction(TS ts,
                                            PetscReal t,
                                            Vec U,
                                            Vec Udot,
                                            PetscReal shift,
                                            Mat A,
                                            Mat B,
                                            void *ctx)
{
  PetscFunctionBeginUser;
  const PetscScalar *xx;
  PetscScalar ustar, J[6][6] = {{0.0}}; // Fix: Initialize matrix

  PetscCall(VecGetArrayRead(U, &xx));

  PetscScalar p1 = xx[0];
  PetscScalar p2 = xx[1];
  PetscScalar p3 = xx[2];
  PetscScalar x1 = xx[3];
  PetscScalar x2 = xx[4];
  PetscScalar x3 = xx[5];

  AppCtx *actx = (AppCtx *)ctx;
  ustar = ComputeUstar(p1, p2, p3, x1, x2, x3);

  /* Compute Jacobian entries */
  //2.0 * x1 * p2 * x2 * ustar * ustar-p2*ustar*ustar
  J[0][1] = +2 * x1 * x2 * ustar * ustar-ustar*ustar;
  J[0][3] = +2 * p2 * x2 * ustar * ustar;
  J[0][4] = +2 * x1 * p2 * ustar * ustar;

  //- p2 * ustar * ustar * (1.0 - x1 * x1)
  J[1][1] = -ustar * ustar * (1.0 - x1 * x1);
  J[1][3] = +2.0 * p2 * ustar * ustar * x1;

  //p3*exp(-x3)
  J[2][2] = exp(-x3);
  J[2][5] = -p3 * exp(-x3);

  //- ustar * ustar * ((1.0 - x1 * x1) * x2 - x1)
  J[4][3] = +2.0 * ustar * ustar * x1 * x2+ustar*ustar;
  J[4][4] = -ustar * ustar * (1.0 - x1 * x1);
  
  //- exp(-x3)
  J[5][5] = exp(-x3);

  /* Apply shift term */
  for (int i = 0; i < 6; i++)
    J[i][i] += shift;

  /* Define index array for rows/columns */
  PetscInt idx[6] = {0, 1, 2, 3, 4, 5};

  /* Set Jacobian values (fix segmentation fault) */
  PetscCall(MatSetValues(B, 6, idx, 6, idx, &J[0][0], INSERT_VALUES));

  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------
   RHSFunction_Outer:
   We have Y = [p10, p20, p30, x3f]  in R^3.

   Steps:
     1) Extract p10, p20, tf from Y.
     2) Set the initial conditions of the inner system:
        p1(0)=p10, p2(0)=p20, p3(0)=-1,
        x1(0)=2, x2(0)=0, x3(0)=0.
     3) Solve the inner ODE on t in [0, tf].
     4) Evaluate mismatch:
          g0 = p1(tf)          (we want =0)
          g1 = x2(tf)          (we want =0)
          g2 = H   (we want =0)
     5) dY/dtau = - g(Y).
   ------------------------------------------------------------------- */
PetscErrorCode IRHSFunction_Outer(TS ts, PetscReal t, Vec U, Vec Udot, Vec F, void *ctx)
{
  PetscFunctionBeginUser; /* Required at start of all PETSc functions */
  AppCtx *user = (AppCtx *)ctx;
  const PetscReal *u, *udot;


  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U, &u));
  PetscReal p10 = u[0];
  PetscReal p20 = u[1]; 
  PetscReal p30 = u[2];
  PetscReal x3f = u[3];
  PetscInt stage = user->stage;
  //x3 = log(t+\epsilon) => t+\epsilon = exp(x3) => t = exp(x3) - \epsilon
  PetscReal tf = exp(x3f)-eps; 
  if (tf <= 0)
  {
    tf = eps;
  }
  PetscScalar       *f;
  
  /* 1) Access array data for U, Udot, and F */
  PetscCall(VecGetArrayRead(Udot, &udot));
  PetscCall(VecGetArray(F,        &f));
  PetscPrintf(PETSC_COMM_WORLD, 
              "Input p10=%g, p20=%g, p30=%g, x3f=%g, tf=%g\n",  
              (double)p10, 
              (double)p20,
              (double)p30, 
              (double)x3f,
              (double)tf);
  PetscScalar x10 = user->x10;
  PetscScalar x20 = user->x20;

  /* 1) Set the initial conditions in user->X (dimension 6). */
  PetscCall(VecSet(user->X, 0.0)); /* zero out everything first. */
  /* p1(0)=p10 => X[0], p2(0)=p20 => X[1], p3(0)=-1 => X[2], x1(0)=2 => X[3], x2(0)=0 => X[4], x3(0)=0 => X[5]. */
  PetscCall(VecSetValue(user->X, 0, p10, INSERT_VALUES));
  PetscCall(VecSetValue(user->X, 1, p20, INSERT_VALUES));
  PetscCall(VecSetValue(user->X, 2, p30, INSERT_VALUES));
  PetscCall(VecSetValue(user->X, 3, x10, INSERT_VALUES)); // x10
  PetscCall(VecSetValue(user->X, 4, x20, INSERT_VALUES)); // x20
  PetscCall(VecSetValue(user->X, 5, log(eps), INSERT_VALUES)); // x30 log(t+\epsilon)
  PetscCall(VecAssemblyBegin(user->X));
  PetscCall(VecAssemblyEnd(user->X));

  /* 2) Integrate the inner system from t=0..tf. */
  PetscCall(TSSetTime(user->tsInner, 0.0));
  PetscCall(TSSetStepNumber(user->tsInner, 0));
  // set the tolerance
  PetscReal rtol = 1e-8;
  PetscReal atol = 1e-8;
  PetscCall(TSSetTolerances(user->tsInner, atol, NULL, rtol, NULL));
  PetscCall(TSSetMaxTime(user->tsInner, tf));

  PetscCall(TSSolve(user->tsInner, user->X));

  /* 3) Evaluate the mismatch at final time:
         g1 = x2(tf) = X[4],
         g2 = x3(tf) - tf = X[5] - tf.
  */
  PetscReal g[4]; // Our job here is to fit the terminal state of the state space only
  {
    const PetscReal *x;
    PetscCall(VecGetArrayRead(user->X, &x));
    PetscReal p1 = x[0];
    PetscReal p2 = x[1];
    PetscReal p3 = x[2];
    PetscReal x1 = x[3];
    PetscReal x2 = x[4];
    PetscReal x3 = x[5];
    PetscReal ustar = ComputeUstar(p1, p2, p3, x1, x2, x3);
    PetscReal H = p1 * x2 + p2 * (ustar * ustar * ((1.0 - x1 * x1) * x2 - x1)) + p3*exp(-x3) + 1.0;
    // Print out the values

    g[0] = (p2);
    g[1] = (p3);
    g[2] = (x1);
    g[3] = (H);

    PetscPrintf(PETSC_COMM_WORLD, "Inner ODE solved at t=tf:%g\n"
                                  "  p1(tf)=%g, p2(tf)=%g, p3(tf)=%g,\n"
                                  "  x1(tf)=%g, x2(tf)=%g, x3(tf)=%g.\n",
                (double)tf,
                (double)x[0], (double)x[1], (double)x[2],
                (double)x[3], (double)x[4], (double)x[5]);
    PetscPrintf(PETSC_COMM_WORLD, "g[0]=%g ", (double)g[0]);
    PetscPrintf(PETSC_COMM_WORLD, "g[1]=%g ", (double)g[1]);
    PetscPrintf(PETSC_COMM_WORLD, "g[2]=%g ", (double)g[2]);
    PetscPrintf(PETSC_COMM_WORLD, "g[3]=%g ", (double)g[3]);

    PetscCall(VecRestoreArrayRead(user->X, &x));
  }

  /* 4) dY/dtau = - g => ydot = [p2(tf), p3(tf), x1(tf), H]. */
  // We are not looking for a unique solution, we are looking for a solution that satisfies the terminal state
  //Compute the residuals wrt the constraints applying the constraints in stages
  if (stage ==1)
  {
    f[0] = udot[0] - g[2];
    f[1] = udot[1] - g[2];
    f[2] = udot[2] - g[2];
    f[3] = udot[3] - g[2];
  }
  else if (stage == 2) {
    f[0] = udot[0] -g[0];
    f[1] = udot[1] -g[0];
    f[2] = udot[2] -g[0];
    f[3] = udot[3] -g[0];

  }
  else if (stage==3) {
    f[0] = udot[0] - g[2]-g[3]-g[0];
    f[1] = udot[1] - g[2]-g[3]-g[0];
    f[2] = udot[2] - g[2]-g[3]-g[0];
    f[3] = udot[3] - g[2]-g[3]-g[0];
  }
  else {
    f[0] = udot[0] - g[0]-g[1]-g[2]-g[3];
    f[1] = udot[1] - g[1]-g[2]-g[3]-g[0];
    f[2] = udot[2] - g[2]-g[3]-g[0]-g[1];
    f[3] = udot[3] - g[3]-g[0]-g[1]-g[2];
  }
  
  
  
  /* 3) Restore vectors */
  PetscCall(VecRestoreArrayRead(U,    &u));
  PetscCall(VecRestoreArrayRead(Udot, &udot));
  PetscCall(VecRestoreArray(F,        &f));

  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------
   RHSFunction_PMP Implicit form:
   Defines the 3D ODE for the costates:
     X = [p1, p2, p3]

   dot(p1) = -2*x1*p2*x2*u*u
   dot(p2) = p2*ustar*ustar*(1.0 - x1*x1)
   dot(p3) = 0
   ------------------------------------------------------------------- */
   static PetscErrorCode IRHSFunction_Costate(TS ts,
    PetscReal t,
    Vec U,
    Vec Udot,
    Vec F,
    void *ctx)
{
  PetscFunctionBeginUser;
  const PetscScalar *u, *udot;
  PetscScalar *f;

  /* Extract state variables */
  PetscCall(VecGetArrayRead(U, &u)); /* x[0]=p1, x[1]=p2, x[2]=p3, x[3]=x1, x[4]=x2, x[5]=x3 */
  PetscCall(VecGetArrayRead(Udot, &udot));
  PetscCall(VecGetArray(F, &f));
  AppCtx *actx = (AppCtx *)ctx;


  PetscReal p1 = u[0];
  PetscReal p2 = u[1];
  PetscReal p3 = u[2];
  /* Create vector for interpolated solution */
  Vec U_interp;
  PetscCall(VecDuplicate(actx->X, &U_interp));
  
  //We need to get the x values at this time
  PetscCall(TSTrajectoryGetVecs(actx->tj, actx->tsInner, PETSC_DECIDE, &t, U_interp, NULL));
  //Extract elements from U_interp
  const PetscScalar *U_interp_array;
  PetscCall(VecGetArrayRead(U_interp, &U_interp_array));
  //We get the x values from U_interp
  PetscScalar x1 = U_interp_array[3];
  PetscScalar x2 = U_interp_array[4];
  PetscScalar x3 = U_interp_array[5];

  PetscReal ustar = ComputeUstar(p1, p2, p3, x1, x2, x3);
  //H = p1*x2 + p2*(ustar*ustar*((1 - x1^2)*x2 - x1)) + p3*exp(-x3) + 1

  /* Compute residuals: f_i = udot[i] - g_i(x) */
  f[0] = udot[0] -2.0 * x1 * p2 * x2 * ustar * ustar+p2*ustar*ustar;
  f[1] = udot[1] + p1+p2 * ustar * ustar * (1.0 - x1 * x1);
  f[2] = udot[2] - p3*exp(-x3); // No implicit dynamics for p3
  //In this case the state space is not propagated
  f[3] = udot[3]; // No implicit dynamics for x1
  f[4] = udot[4];
  f[5] = udot[5]; // No implicit dynamics for x3

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(Udot, &udot));
  PetscCall(VecRestoreArray(F, &f));

  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------
   RHSFunction_State Implicit form:
   Defines the 3D ODE for the costates:
     X = [x1, x2, x3]

   dot(p1) = -2*x1*p2*x2*u*u
   dot(p2) = p2*ustar*ustar*(1.0 - x1*x1)
   dot(p3) = 0
   ------------------------------------------------------------------- */
   static PetscErrorCode IRHSFunction_Inner_State(TS ts,
    PetscReal t,
    Vec U,
    Vec Udot,
    Vec F,
    void *ctx)
{
  PetscFunctionBeginUser;
  const PetscScalar *u, *udot;
  PetscScalar *f;

  /* Extract state variables */
  PetscCall(VecGetArrayRead(U, &u)); /* x[0]=p1, x[1]=p2, x[2]=p3, x[3]=x1, x[4]=x2, x[5]=x3 */
  PetscCall(VecGetArrayRead(Udot, &udot));
  PetscCall(VecGetArray(F, &f));
  AppCtx *actx = (AppCtx *)ctx;


  PetscReal p1 = u[0];
  PetscReal p2 = u[1];
  PetscReal p3 = u[2];
  /* Create vector for interpolated solution */
  Vec U_interp;
  PetscCall(VecDuplicate(actx->X, &U_interp));
  
  //We need to get the x values at this time
  PetscCall(TSTrajectoryGetVecs(actx->tj, actx->tsInner, PETSC_DECIDE, &t, U_interp, NULL));
  //Extract elements from U_interp
  const PetscScalar *U_interp_array;
  PetscCall(VecGetArrayRead(U_interp, &U_interp_array));
  //We get the x values from U_interp
  PetscScalar x1 = U_interp_array[3];
  PetscScalar x2 = U_interp_array[4];
  PetscScalar x3 = U_interp_array[5];

  PetscReal ustar = ComputeUstar(p1, p2, p3, x1, x2, x3);
  //H = p1*x2 + p2*(ustar*ustar*((1 - x1^2)*x2 - x1)) + p3*exp(-x3) + 1

  /* Compute residuals: f_i = udot[i] - g_i(x) */
  f[0] = udot[0];
  f[1] = udot[1];
  f[2] = udot[2]; // No implicit dynamics for p3
  //In this case the state space is not propagated
  f[3] = udot[3]-x2; // No implicit dynamics for x1
  f[4] = udot[4]-ustar * ustar * ((1.0 - x1 * x1) * x2 - x1);
  f[5] = udot[5]-exp(x3); // No implicit dynamics for x3

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(Udot, &udot));
  PetscCall(VecRestoreArray(F, &f));

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------
   RHSFunction_Outer_State Implicit form:
   Defines the 1D ODE for minimum time to be used in a pseudo transient method:
   We need to find the terminal x3f that sets the Hamiltonian to zero
   assuming the costates are known and assuming the initial state space is known
   here x3 = log(t+\epsilon) => t = exp(x3) - \epsilon
   X = [x3f] 


   ------------------------------------------------------------------- */
   static PetscErrorCode IRHSFunction_Outer_State(TS ts,
    PetscReal t,
    Vec U,
    Vec Udot,
    Vec F,
    void *ctx)
{
  PetscFunctionBeginUser;
  const PetscScalar *u, *udot;
  PetscScalar *f;

  /* Extract state variables */
  PetscCall(VecGetArrayRead(U, &u)); /* x[0]=p1, x[1]=p2, x[2]=p3, x[3]=x1, x[4]=x2, x[5]=x3 */
  PetscCall(VecGetArrayRead(Udot, &udot));
  PetscCall(VecGetArray(F, &f));
  AppCtx *actx = (AppCtx *)ctx;


  PetscReal p1 = u[0];
  PetscReal p2 = u[1];
  PetscReal p3 = u[2];
  /* Create vector for interpolated solution */
  Vec U_interp;
  PetscCall(VecDuplicate(actx->X, &U_interp));
  
  //We need to get the x values at this time
  PetscCall(TSTrajectoryGetVecs(actx->tj, actx->tsInner, PETSC_DECIDE, &t, U_interp, NULL));
  //Extract elements from U_interp
  const PetscScalar *U_interp_array;
  PetscCall(VecGetArrayRead(U_interp, &U_interp_array));
  //We get the x values from U_interp
  PetscScalar x1 = U_interp_array[3];
  PetscScalar x2 = U_interp_array[4];
  PetscScalar x3 = U_interp_array[5];

  PetscReal ustar = ComputeUstar(p1, p2, p3, x1, x2, x3);
  //H = p1*x2 + p2*(ustar*ustar*((1 - x1^2)*x2 - x1)) + p3*exp(-x3) + 1

  /* Compute residuals: f_i = udot[i] - g_i(x) */
  f[0] = udot[0];
  f[1] = udot[1];
  f[2] = udot[2]; // No implicit dynamics for p3
  //In this case the state space is not propagated
  f[3] = udot[3]-x2; // No implicit dynamics for x1
  f[4] = udot[4]-ustar * ustar * ((1.0 - x1 * x1) * x2 - x1);
  f[5] = udot[5]-exp(x3); // No implicit dynamics for x3

  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArrayRead(Udot, &udot));
  PetscCall(VecRestoreArray(F, &f));

  PetscFunctionReturn(0);
}








/* -------------------------------------------------------------------
   main():
     1) Create AppCtx, parse options.
     2) Build inner TS & 6D vector.
     3) Build outer TS & 3D vector Y.
     4) Solve in pseudo-time.
     5) Print results, cleanup.
   ------------------------------------------------------------------- */
int main(int argc, char **argv)
{
  PetscInitialize(&argc, &argv, NULL, NULL);

  /* Create the application context. */
  AppCtx user;
  Mat A;
  PetscInt n = 6;
  user.x10 = 1.0; user.x20 = 0.0;

  /* 1) Build the INNER system (dimension 6). */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &user.tsInner));
  PetscCall(TSSetProblemType(user.tsInner, TS_NONLINEAR));
  /* You can choose a TS type, e.g. TSRK, TSBEULER, TSTHETA, etc. */
  PetscCall(TSSetType(user.tsInner, TSARKIMEX));
  // set the order of the method
  PetscCall(TSARKIMEXSetType(user.tsInner, "5"));
  // Set the tolerances for the inner solver
  PetscReal atol = 1e-8, rtol = 1e-6;
  PetscOptionsGetReal(NULL, NULL, "-atol", &atol, NULL);
  PetscOptionsGetReal(NULL, NULL, "-rtol", &rtol, NULL);

  /* Create X in R^6 for [p1, p2, p3, x1, x2, x3]. */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &user.X));
  PetscCall(VecSetSizes(user.X, PETSC_DECIDE, 6));
  PetscCall(VecSetFromOptions(user.X));

  /* Set the Explicit RHS for the inner system. */
  PetscCall(TSSetRHSFunction(user.tsInner, NULL, RHSFunction_Inner, &user));
  // Set the implicit RHS function
  PetscCall(TSSetIFunction(user.tsInner, NULL, IRHSFunction_Inner, &user));
  /* Create the A Jacobian matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(TSSetIJacobian(user.tsInner, A, A, InnerJacobianFunction, &user));

  /* Optionally set time-step parameters for the inner solver. */
  PetscReal dt_inner = 0.01;
  PetscOptionsGetReal(NULL, NULL, "-dt_inner", &dt_inner, NULL);
  PetscCall(TSSetTimeStep(user.tsInner, dt_inner));
  PetscCall(TSSetMaxSteps(user.tsInner, 1000000));
  PetscCall(TSSetMaxSNESFailures(user.tsInner, 100000));

  /* 2) Build the OUTER system in R^3 for [p10, p20, tf]. */
  TS tsOuter;
  PetscCall(TSCreate(PETSC_COMM_WORLD, &tsOuter));
  PetscCall(TSSetProblemType(tsOuter, TS_NONLINEAR));

  PetscCall(TSSetType(tsOuter, TSBDF));  /* or TSTHETA, etc. */
  PetscCall(TSSetMaxSNESFailures(tsOuter, 10000));
  //set the tolerances to be quite small
  user.stage = 1;


  Vec Y;
  PetscCall(VecCreate(PETSC_COMM_WORLD, &Y));
  PetscCall(VecSetSizes(Y, PETSC_DECIDE, 4)); // This is the outer ODE with four unknowns.
  PetscCall(VecSetFromOptions(Y));

  /* Initial guess for Y = [p10, p20, p30, x3f]. We allow them to be set by command line. */
  PetscReal p10_init = 1.0, p20_init = 1.0, p30_init=1.0, x3f_init = 0.1;
  PetscOptionsGetReal(NULL, NULL, "-p10_init", &p10_init, NULL);
  PetscOptionsGetReal(NULL, NULL, "-p20_init", &p20_init, NULL);
  PetscOptionsGetReal(NULL, NULL, "-p30_init", &p30_init, NULL);
  PetscOptionsGetReal(NULL, NULL, "-x3f_init", &x3f_init, NULL);

  /* Put them into the Y vector. */
  PetscCall(VecSetValue(Y, 0, p10_init, INSERT_VALUES));
  PetscCall(VecSetValue(Y, 1, p20_init, INSERT_VALUES));
  PetscCall(VecSetValue(Y, 2, p30_init, INSERT_VALUES));
  PetscCall(VecSetValue(Y, 3, x3f_init, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(Y));
  PetscCall(VecAssemblyEnd(Y));

  /* Set the outer RHS function. */
  PetscCall(TSSetIFunction(tsOuter, NULL, IRHSFunction_Outer, &user));

  /* Set pseudo-time range: 0..tau_max. */
  PetscReal tau_max = 100.0;
  PetscOptionsGetReal(NULL, NULL, "-tau_max", &tau_max, NULL);
  PetscCall(TSSetTime(tsOuter, 0.1));
  PetscCall(TSSetMaxTime(tsOuter, tau_max));

  /* Outer time-step size. */
  PetscReal dt_outer = 0.1;
  PetscOptionsGetReal(NULL, NULL, "-dt_outer", &dt_outer, NULL);
  PetscCall(TSSetTimeStep(tsOuter, dt_outer));
  PetscCall(TSSetMaxSteps(tsOuter, 1000000));

  /* 3) Solve the outer system in pseudo-time. */
  PetscCall(TSSolve(tsOuter, Y));

  /* 4) Extract the final [p10, p20, tf]. */
  const PetscReal *yfinal;
  PetscCall(VecGetArrayRead(Y, &yfinal));
  PetscReal p10_final  = yfinal[0];
  PetscReal p20_final  = yfinal[1];
  PetscReal p30_final  = yfinal[2];
  PetscReal x3f_final  = yfinal[3];
  PetscPrintf(PETSC_COMM_WORLD, 
              "Final p10=%g, p20=%g, p30=%g, x3f=%g\n", 
              (double)p10_final, 
              (double)p20_final, 
              (double)p30_final,
              (double)x3f_final);
  PetscCall(VecRestoreArrayRead(Y, &yfinal));
  // At this stage we have a solution that is feasible in state space only
  // We need to calculate the BVP problem with the terminal costate conditions
  // to get the full PMP solution to the problem
  // We will use SNES with NGMRES to solve the BVP problem

  /* Print out the results. */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "\nPseudo-transient solve finished.\n"
                        "  p10_final = %g, p20_final = %g, p30_final=%g, x3f_final=%g\n",
                        (double)p10_final, 
                        (double)p20_final, 
                        (double)p30_final,
                        (double)x3f_final));
  //END OF PSEUDO TRANSIENT SOLUTION
  //This simply gives us a feasible trajectory.  Now we need to find the optimal solution
  //Using this as a guess we can solve the PMP problem





  //Since we have a first solution we need to re run the inner dynamics with trajectory interpolation
  PetscReal p10 = p10_final;
  PetscReal p20 = p20_final; 
  PetscReal p30 = p30_final;
  PetscReal x3f = x3f_final;
  //x3 = log(t+\epsilon) => t+\epsilon = exp(x3) => t = exp(x3) - \epsilon
  PetscReal tf = exp(x3f)-eps; 
  

  /* 1) Set the initial conditions in user->X (dimension 6). */
  PetscCall(VecSet(user.X, 0.0)); /* zero out everything first. */
  /* p1(0)=p10 => X[0], p2(0)=p20 => X[1], p3(0)=-1 => X[2], x1(0)=2 => X[3], x2(0)=0 => X[4], x3(0)=0 => X[5]. */
  PetscCall(VecSetValue(user.X, 0, p10, INSERT_VALUES));
  PetscCall(VecSetValue(user.X, 1, p20, INSERT_VALUES));
  PetscCall(VecSetValue(user.X, 2, p30, INSERT_VALUES));
  PetscCall(VecSetValue(user.X, 3, user.x10, INSERT_VALUES)); // x10
  PetscCall(VecSetValue(user.X, 4, user.x20, INSERT_VALUES)); // x20
  PetscCall(VecSetValue(user.X, 5, log(eps), INSERT_VALUES)); // x30 log(t+\epsilon)
  PetscCall(VecAssemblyBegin(user.X));
  PetscCall(VecAssemblyEnd(user.X));

  /* 2) Integrate the inner system from t=0..tf. */
  PetscCall(TSSetSaveTrajectory(user.tsInner));
  PetscCall(TSSetTime(user.tsInner, 0.0));
  PetscCall(TSSetStepNumber(user.tsInner, 0));
  // set the tolerance
  PetscCall(TSSetTolerances(user.tsInner, atol, NULL, rtol, NULL));
  PetscCall(TSSetMaxTime(user.tsInner, tf));

  PetscCall(TSSolve(user.tsInner, user.X));
  /* Set the TS to save its trajectory */
  PetscCall(TSTrajectorySetUp(user.tj, user.tsInner));
  PetscCall(TSGetTrajectory(user.tsInner, &user.tj));




  //Now run the PMP backwards since we have a guess for the terminal state space
  //Locally copy the user.X vector
  const PetscScalar *U_interp_array;
  PetscCall(VecGetArrayRead(user.X, &U_interp_array));
  //We get the x values from U_interp
  user.p1f = U_interp_array[0];
  user.p2f = U_interp_array[1];
  user.p3f = U_interp_array[2];
  user.x1f = U_interp_array[3];
  user.x2f = U_interp_array[4];
  user.x3f = U_interp_array[5];
  //Create an implicit solver for the PMP
  TS tsPMP;
  PetscCall(TSSetSaveTrajectory(user.tsInner));

  PetscCall(TSCreate(PETSC_COMM_WORLD, &tsPMP));
  PetscCall(TSSetProblemType(tsPMP, TS_NONLINEAR));
  PetscCall(TSSetType(tsPMP, TSBDF));  /* or TSTHETA, etc. */
  PetscCall(TSSetMaxSNESFailures(tsPMP, 10000));
  //set the tolerances to be quite small
  PetscCall(TSSetTolerances(tsPMP, atol, NULL, rtol, NULL));
  //Set the initial conditions for the PMP
  PetscCall(VecSet(user.X, 0.0)); /* zero out everything first. */
  /* Run this backwards*/
  PetscCall(VecSetValue(user.X, 0, user.p1f, INSERT_VALUES));
  PetscCall(VecSetValue(user.X, 1, 0.0, INSERT_VALUES));
  PetscCall(VecSetValue(user.X, 2, 0.0, INSERT_VALUES));
  PetscCall(VecSetValue(user.X, 3, user.x1f, INSERT_VALUES)); // x10
  PetscCall(VecSetValue(user.X, 4, user.x2f, INSERT_VALUES)); // x20
  PetscCall(VecSetValue(user.X, 5, user.x3f, INSERT_VALUES)); // x30 log(t+\epsilon)
  tf = exp(user.x3f)-eps;  //The final time is contained in the x3f
  PetscCall(VecAssemblyBegin(user.X));
  PetscCall(VecAssemblyEnd(user.X));
  PetscCall(TSSetTime(tsPMP, tf));
  PetscCall(TSSetMaxTime(tsPMP, 0.0));
  PetscCall(TSSetTimeStep(tsPMP, dt_inner));
  PetscCall(TSSetMaxSteps(tsPMP, 1000000));
  PetscCall(TSSetIFunction(tsPMP, NULL, IRHSFunction_Costate, &user));
  PetscCall(TSSolve(tsPMP, user.X));
  PetscCall(TSTrajectorySetUp(user.tj, tsPMP));
  PetscCall(TSGetTrajectory(tsPMP, &user.tj));

  //Print out the final values
  PetscCall(VecGetArrayRead(user.X, &U_interp_array));
  //We get the x values from U_interp
  PetscReal p1f = U_interp_array[0];
  PetscReal p2f = U_interp_array[1];
  PetscReal p3f = U_interp_array[2];
  PetscReal x1f = U_interp_array[3];
  PetscReal x2f = U_interp_array[4];
  x3f = U_interp_array[5];
  PetscPrintf(PETSC_COMM_WORLD, "Final after PMP reverse p1=%g, p2=%g, p3=%g, x1=%g, x2=%g, x3=%g\n", (double)p1f, (double)p2f, (double)p3f, (double)x1f, (double)x2f, (double)x3f);
  
  
  
  
  
  
  

  // Now we need to solver for the terminal costate conditions
  // We will use the SNES solver to solve the BVP problem
  // We will use multiple shooting with SNESVINEWTONRSLS that
  // can accomodate constraints in the variables
  // We will use a N segments for which there are 6*N variables
  // plus 6 initial conditiosn and 6 terminal conditions
  // We have 6*N+12 variables for which we have 6*N equations
  // and 3 unknown initial conditions and 3 unknown terminal conditions
  // For which we have 3 known initial conditions and 3 final conditions
  // PetscInt Nseg = 2; //Number of segments
  // SNES           snes;
  // Mat            J; //This is the Jacobian is a fake Jacobian needed by Petsc solver
  // PetscErrorCode ierr;
  // SNESLineSearch linesearch;
  // Vec            x, r;
  /* Create 1D Vec for solution x */
  // ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
  // ierr = VecSetSizes(x, PETSC_DECIDE, 3); CHKERRQ(ierr);
  // ierr = VecSetFromOptions(x); CHKERRQ(ierr);

  // ierr = SNESCreate(PETSC_COMM_WORLD, &snes); CHKERRQ(ierr);
  // Set the bounds for the variables
  // Vec L, U;
  // ierr = VecDuplicate(x, &L); CHKERRQ(ierr);
  // ierr = VecSet(L, PETSC_NINFINITY); CHKERRQ(ierr);
  // Time should be positive
  // ierr = VecSetValue(L, 2, 0.0, INSERT_VALUES); CHKERRQ(ierr);
  // ierr = VecDuplicate(x, &U); CHKERRQ(ierr);
  // ierr = VecSet(U, PETSC_INFINITY); CHKERRQ(ierr);
  // ierr = SNESVISetVariableBounds(snes, L, U); CHKERRQ(ierr);
  // ierr = SNESSetType(snes, SNESNGMRES); CHKERRQ(ierr);
  /* Set initial guess */
  // ierr = VecSetValue(x, 0, p10_final, INSERT_VALUES); CHKERRQ(ierr);
  // ierr = VecSetValue(x, 1, p20_final, INSERT_VALUES); CHKERRQ(ierr);
  // ierr = VecSetValue(x, 2, tf_final, INSERT_VALUES); CHKERRQ(ierr);
  // ierr = VecAssemblyBegin(x); CHKERRQ(ierr);
  // ierr = VecAssemblyEnd(x);   CHKERRQ(ierr);

  /* Optional residual Vec (same size as x) */
  // ierr = VecDuplicate(x, &r); CHKERRQ(ierr);

  /* Create 1x1 matrix for the Jacobian */
  // ierr = MatCreate(PETSC_COMM_WORLD, &J); CHKERRQ(ierr);
  // ierr = MatSetSizes(J, 1, 1, 1, 1); CHKERRQ(ierr);
  // ierr = MatSetFromOptions(J); CHKERRQ(ierr);
  // ierr = MatSetUp(J); CHKERRQ(ierr);

  /* Create the top-level SNES and set it to composite mode */

  /* Set function and Jacobian on the top-level SNES */
  // ierr = SNESSetFunction(snes, r, VDPPMPResidualFunction, &user); CHKERRQ(ierr);
  /*
     Get the internal line search context from the SNES,
     specify we want to use an L2 line search (i.e., SNESLINESEARCHL2),
     and optionally set other parameters.
  */
  // ierr = SNESGetLineSearch(snes, &linesearch); CHKERRQ(ierr);
  /* Set the type of the line search (can be SNESLINESEARCHBT, SNESLINESEARCHBASIC, etc.) */
  // ierr = SNESLineSearchSetType(linesearch, SNESLINESEARCHL2); CHKERRQ(ierr);

  // ierr = SNESSetJacobian(snes, J, J, VDPJacobianFunction, &user); CHKERRQ(ierr);
  /* Optional: set a damping factor or other parameters directly in code. */
  /*
     For example, if we want a damping factor = 0.5, we can do:
       SNESLineSearchSetDamping(linesearch, 0.5);
     or read from user-provided command line with SNESLineSearchSetFromOptions().
  */
  // ierr = SNESLineSearchSetDamping(linesearch, 0.5); CHKERRQ(ierr);

  /*
     Honor any additional user-provided command-line options
     (like -snes_ngmres_m 30, -snes_ngmres_linesearch_type l2, etc.)
   */
  // ierr = SNESLineSearchSetFromOptions(linesearch); CHKERRQ(ierr);

  /* Optionally set from command line, e.g. -snes_monitor */
  // ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

  /* Solve */
  // ierr = SNESSolve(snes, NULL, x); CHKERRQ(ierr);

  /* Cleanup */
  PetscCall(VecDestroy(&Y));
  PetscCall(TSDestroy(&tsOuter));
  PetscCall(VecDestroy(&user.X));
  PetscCall(TSDestroy(&user.tsInner));
  // ierr = SNESDestroy(&snes); CHKERRQ(ierr);
  // ierr = MatDestroy(&J);     CHKERRQ(ierr);
  // ierr = VecDestroy(&x);     CHKERRQ(ierr);
  // ierr = VecDestroy(&r);     CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}