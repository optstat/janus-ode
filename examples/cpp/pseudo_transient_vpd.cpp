#include <petscts.h>
#include <petsctao.h>
#include <stdio.h>

/* 
 * We use a pseudo-transient method to drive the outer variables [p10, p20, tf].
 * The inner system is 6D: [p1, p2, p3, x1, x2, x3].
 */

typedef struct {
  TS          tsInner;  /* TS for the inner ODE system */
  Vec         X;        /* 6D solution vector for [p1, p2, p3, x1, x2, x3] */

  PetscScalar W;        /* weight used in the "smoothed" control formula      */
  PetscScalar umin, umax;
  PetscScalar x10, x20; /* initial positions x1(0), x2(0) */
  PetscScalar p30;      /* (often = -1.0 if final time is free and cost = x3) */

  /* We'll store the 'outer' unknowns in the struct, but we also keep them in the Y vector. */
  PetscScalar p10; 
  PetscScalar p20; 
  PetscScalar tf;

} AppCtx;


/* -------------------------------------------------------------------
   ComputeUstar(): a smoothed / saturated control
     u = -p2*(1 - x1^2)*x2 / W   (then clamp between umin, umax)
   ------------------------------------------------------------------- */
static inline PetscReal ComputeUstar(PetscReal p1,
                                     PetscReal p2,
                                     PetscReal p3,
                                     PetscReal x1,
                                     PetscReal x2,
                                     PetscReal x3,
                                     void *ctx)
{
  AppCtx     *actx = (AppCtx*) ctx;
  PetscReal   u    = -p2*(1.0 - x1*x1)*x2 / (actx->W);
  if (u < actx->umin) u = actx->umin;
  if (u > actx->umax) u = actx->umax;
  return u;
}


/* -------------------------------------------------------------------
   RHSFunction_Inner:
   X = [p1, p2, p3, x1, x2, x3].

   dot(p1) = -2*x1*p2*x2*u - p2
   dot(p2) =  p1 + p2*u*(1 - x1^2)
   dot(p3) =  0
   dot(x1) =  x2
   dot(x2) =  u*(1 - x1^2)*x2 - x1
   dot(x3) =  1
   ------------------------------------------------------------------- */
static PetscErrorCode RHSFunction_Inner(TS ts, PetscReal t, Vec X, Vec Xdot, void* ctx)
{
  PetscFunctionBeginUser;

  const PetscReal *x;
  PetscReal       *xdot;
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArray(Xdot, &xdot));

  PetscReal p1 = x[0];
  PetscReal p2 = x[1];
  PetscReal p3 = x[2];
  PetscReal x1 = x[3];
  PetscReal x2 = x[4];
  PetscReal x3 = x[5];

  AppCtx     *actx = (AppCtx*)ctx;
  PetscReal   u    = ComputeUstar(p1,p2,p3, x1,x2,x3, actx);

  xdot[0] = -2.0*x1*p2*x2*u - p2;
  xdot[1] =  p1 + p2*u*(1.0 - x1*x1);
  xdot[2] =  0.0;
  xdot[3] =  x2;
  xdot[4] =  u*(1.0 - x1*x1)*x2 - x1;
  xdot[5] =  1.0;

  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArray(Xdot, &xdot));
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------
   RHSJacobianP:
   Partial derivative of f(t,x,p) wrt the parameters p.
   We want d(f)/d(x(0)). For an n-by-m system, this is an n x m matrix.
   In PETSc, we pass it back as a Mat with dimension [n x m].
   ------------------------------------------------------------------- */
   static PetscErrorCode RHSJacobianP_Inner(TS ts, PetscReal t, Vec X, Mat A, void *ctx)
   {
     AppCtx            *user = (AppCtx*)ctx;
     const PetscScalar *x;
     PetscCall(VecGetArrayRead(X, &x));
   
     PetscInt row[2] = {0,1}, col[2] = {0,1}; /* for 2D example: 2 parameters => col size=2 */
     PetscScalar J[4];
   
     /* d f1 / d p1 */ J[0] = -x[0];
     /* d f1 / d p2 */ J[1] =  x[1];
     /* d f2 / d p1 */ J[2] =  x[0];
     /* d f2 / d p2 */ J[3] = -x[1];
   
     /* Insert into the [2 x 2] matrix A. */
     PetscCall(MatSetValues(A, 2, row, 2, col, J, INSERT_VALUES));
     PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
     PetscCall(MatAssemblyEnd(A,   MAT_FINAL_ASSEMBLY));
   
     PetscCall(VecRestoreArrayRead(X, &x));
     return 0;
   }


/* -------------------------------------------------------------------
   RHSFunction_Outer:
   Y = [p10, p20, tf].
   in pseudo-time tau:

     dY/dtau = -g( Y ),

   where
     g0 = p1(tf)  -> we want p1(tf)=0
     g1 = p2(tf)  -> we want p2(tf)=0
     g2 = x1(tf)  -> we want x1(tf)=0

   This is just an example system. 
   ------------------------------------------------------------------- */
static PetscErrorCode RHSFunction_Outer(TS ts, PetscReal tau, Vec Y, Vec Ydot, void* ctx)
{
  PetscFunctionBeginUser;
  AppCtx         *user = (AppCtx*) ctx;

  const PetscReal *y;
  PetscReal       *ydot;
  PetscCall(VecGetArrayRead(Y, &y));

  user->p10 = y[0];
  user->p20 = y[1];
  /* Square y[2] to keep tf >= 0: */
  user->tf  = (y[2] < 0.0) ? 0.0 : y[2]*y[2];

  PetscCall(VecRestoreArrayRead(Y, &y));

  PetscReal tf = user->tf;
  PetscPrintf(PETSC_COMM_WORLD, "Outer step: p10=%g, p20=%g, tf=%g\n",
              (double)user->p10, (double)user->p20, (double)tf);

  /* 1) Set initial conditions in user->X. */
  PetscCall(VecSet(user->X, 0.0));
  /* p1(0)=p10 => X[0], p2(0)=p20 => X[1], p3(0)= -1 => X[2],
     x1(0)= x10 => X[3], x2(0)= x20 => X[4], x3(0)= 0 => X[5]. */
  PetscCall(VecSetValue(user->X, 0, user->p10, INSERT_VALUES));
  PetscCall(VecSetValue(user->X, 1, user->p20, INSERT_VALUES));
  PetscCall(VecSetValue(user->X, 2, user->p30, INSERT_VALUES));
  PetscCall(VecSetValue(user->X, 3, user->x10, INSERT_VALUES));
  PetscCall(VecSetValue(user->X, 4, user->x20, INSERT_VALUES));
  PetscCall(VecSetValue(user->X, 5, 0.0,       INSERT_VALUES));
  PetscCall(VecAssemblyBegin(user->X));
  PetscCall(VecAssemblyEnd(user->X));

  /* 2) Integrate the inner system from t=0..tf. */
  PetscCall(TSSetTime(user->tsInner, 0.0));
  PetscCall(TSSetStepNumber(user->tsInner, 0));
  PetscCall(TSSetMaxTime(user->tsInner, tf));
  PetscCall(TSSolve(user->tsInner, user->X));

  /* 3) Evaluate mismatch at final time. */
  PetscReal g[3] = {0.0, 0.0, 0.0};
  {
    const PetscReal *x;
    PetscCall(VecGetArrayRead(user->X, &x));
    PetscReal p1f = x[0];
    PetscReal p2f = x[1];
    PetscReal p3f = x[2];
    PetscReal x1f = x[3];
    PetscReal x2f = x[4];
    PetscReal x3f = x[5];

    /* We want p1(tf)=0, p2(tf)=0, x1(tf)=0, etc. */
    g[0] = p1f;
    g[1] = p2f;
    g[2] = x1f;

    /* Debug print */
    PetscPrintf(PETSC_COMM_WORLD,
      "  Inner solve at tf=%g => p1=%g, p2=%g, p3=%g, x1=%g, x2=%g, x3=%g\n",
      (double)tf,(double)p1f,(double)p2f,(double)p3f,(double)x1f,(double)x2f,(double)x3f);

    PetscCall(VecRestoreArrayRead(user->X, &x));
  }

  /* 4) dY/dtau = -g => ydot = [-p1f, -p2f, -x1f]. */
  PetscCall(VecGetArray(Ydot, &ydot));
  ydot[0] = -g[0];
  ydot[1] = -g[1];
  ydot[2] = -g[2];
  PetscCall(VecRestoreArray(Ydot, &ydot));

  PetscFunctionReturn(0);
}


int main(int argc, char** argv)
{
  PetscInitialize(&argc, &argv, NULL, NULL);

  /* Create the application context and set default values */
  AppCtx user;
  user.W     = 0.001;
  user.umin  = 1.0;
  user.umax  = 10.0;
  user.x10   = 2.0;
  user.x20   = 2.0;
  user.p30   = -1.0; /* typical for free-final-time with cost = x3(tf) */

  /* Outer unknowns: initial guess */
  user.p10   = 1.0;
  user.p20   = 1.0;
  user.tf    = 1.0;

  /* Parse from command line if desired: e.g. -p10_init 3.0, etc. */
  PetscOptionsGetReal(NULL,NULL,"-p10_init",&user.p10, NULL);
  PetscOptionsGetReal(NULL,NULL,"-p20_init",&user.p20, NULL);
  PetscOptionsGetReal(NULL,NULL,"-tf_init", &user.tf,  NULL);

  /* 1) Build the INNER system (dimension 6). */
  TSCreate(PETSC_COMM_WORLD, &user.tsInner);
  TSSetProblemType(user.tsInner, TS_NONLINEAR);

  /* Optionally set a default TS type in code. The user can override it on cmd line. */
  TSSetType(user.tsInner, TSARKIMEX);

  /* Let the user control the inner TS with “-inner_ts_*” command-line options */
  TSSetOptionsPrefix(user.tsInner, "inner_");
  TSSetFromOptions(user.tsInner);

  /* Example: set tolerances or read from the command line directly, if needed */
  PetscReal atol = 1e-8, rtol = 1e-6;
  PetscOptionsGetReal(NULL, NULL, "-atol", &atol, NULL);
  PetscOptionsGetReal(NULL, NULL, "-rtol", &rtol, NULL);
  /* ... If you want, you can do TSSetTolerances(user.tsInner, rtol, NULL, atol, NULL); */

  /* Create X in R^6 for [p1, p2, p3, x1, x2, x3]. */
  VecCreate(PETSC_COMM_WORLD, &user.X);
  VecSetSizes(user.X, PETSC_DECIDE, 6);
  VecSetFromOptions(user.X);

  /* Set the RHS for the inner system. */
  TSSetRHSFunction(user.tsInner, NULL, RHSFunction_Inner, &user);

  /* Set the jacobian for the inner system*/
    /* 4) Provide partial derivative wrt parameters p. */
  /*    A is an [n x m] matrix. PETSc will create it internally if not provided. */
  PetscCall(TSSetRHSJacobianP(user.tsInner, NULL, RHSJacobianP_Inner, &user));

  

  /* Optionally set time-step parameters for the inner solver. */
  PetscReal dt_inner = 0.001;
  PetscOptionsGetReal(NULL, NULL, "-dt_inner", &dt_inner, NULL);
  TSSetTimeStep(user.tsInner, dt_inner);
  TSSetMaxSteps(user.tsInner, 1000000);
  TSSetMaxSNESFailures(user.tsInner, 100000);


  /* 2) Build the OUTER system in R^3 for [p10, p20, tf]. */
  TS tsOuter;
  TSCreate(PETSC_COMM_WORLD, &tsOuter);
  TSSetProblemType(tsOuter, TS_NONLINEAR);
  /* Optionally set a default solver type (the user can override). */
  TSSetType(tsOuter, TSARKIMEX);

  /* Let user control outer TS via “-outer_ts_*” command-line options */
  TSSetOptionsPrefix(tsOuter, "outer_");
  TSSetFromOptions(tsOuter);

  TSSetMaxSNESFailures(tsOuter, 100000);

  /* Create Y in R^3 for the outer problem. */
  Vec Y;
  VecCreate(PETSC_COMM_WORLD, &Y);
  VecSetSizes(Y, PETSC_DECIDE, 3);
  VecSetFromOptions(Y);

  /* Fill Y with initial guesses [p10, p20, sqrt(tf)] because we do tf = y[2]^2 */
  /* or just store the direct value and let the code do the squaring. */
  PetscCall(VecSetValue(Y, 0, user.p10, INSERT_VALUES));
  PetscCall(VecSetValue(Y, 1, user.p20, INSERT_VALUES));
  PetscCall(VecSetValue(Y, 2, user.tf,  INSERT_VALUES));
  VecAssemblyBegin(Y);
  VecAssemblyEnd(Y);

  /* Set the outer RHS function. */
  TSSetRHSFunction(tsOuter, NULL, RHSFunction_Outer, &user);

  /* Set pseudo-time range for the outer solver. */
  PetscReal tau_max = 10.0;
  PetscOptionsGetReal(NULL,NULL,"-tau_max",&tau_max,NULL);
  TSSetTime(tsOuter, 0.0);
  TSSetMaxTime(tsOuter, tau_max);

  /* Outer time-step size. */
  PetscReal dt_outer = 0.001;
  PetscOptionsGetReal(NULL,NULL,"-dt_outer",&dt_outer,NULL);
  TSSetTimeStep(tsOuter, dt_outer);
  TSSetMaxSteps(tsOuter, 1000000);

  /* 3) Solve the outer system in pseudo-time. */
  TSSolve(tsOuter, Y);

  /* 4) Extract the final [p10, p20, tf]. */
  const PetscReal *yfinal;
  VecGetArrayRead(Y, &yfinal);
  PetscReal p10_final = yfinal[0];
  PetscReal p20_final = yfinal[1];
  PetscReal tf_final  = yfinal[2];
  VecRestoreArrayRead(Y, &yfinal);

  PetscPrintf(PETSC_COMM_WORLD, 
              "\nPseudo-transient outer solve finished.\n"
              "  p10_final = %g\n"
              "  p20_final = %g\n"
              "  tf_final  = %g  (recall code does tf = y[2]^2)\n",
              (double)p10_final, (double)p20_final, (double)tf_final);

  /* Cleanup */
  VecDestroy(&Y);
  TSDestroy(&tsOuter);
  VecDestroy(&user.X);
  TSDestroy(&user.tsInner);

  PetscFinalize();
  return 0;
}