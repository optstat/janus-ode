#include <petscts.h>

/* 
   Data structure to hold application-specific information
   (the "user context").
*/
typedef struct {
  PetscReal    mu;       /* Van der Pol parameter */
  PetscReal    x0, v0;   /* Initial conditions for x1(0), x2(0) */
  TS           tsInner;  /* TS for the inner Van der Pol integration */
  Vec          X;        /* Solution vector [x1, x2] for inner system */
} AppCtx;

/* -------------------------------------------------------------------
   Inner system: Van der Pol ODE
     x1' = x2
     x2' = mu * (1 - x1^2) * x2 - x1

   Called by TS each time it needs f(t,X).
   Ydot = f(t,Y).
   Here, Y = [x1, x2].
   We'll store mu in the user context.
   ------------------------------------------------------------------- */
static PetscErrorCode RHSFunction_Inner(TS ts, PetscReal t, Vec Y, Vec Ydot, void* ctx)
{
  AppCtx         *user = (AppCtx*) ctx;
  const PetscReal mu   = user->mu;
  const PetscReal *y;
  PetscReal       *f;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(Y, &y));
  PetscCall(VecGetArray(Ydot, &f));

  /* y[0] = x1, y[1] = x2 */
  PetscReal x1 = y[0];
  PetscReal x2 = y[1];

  f[0] = x2;
  f[1] = mu * (1.0 - x1*x1) * x2 - x1;

  PetscCall(VecRestoreArrayRead(Y, &y));
  PetscCall(VecRestoreArray(Ydot, &f));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------
   Outer system: 
     dT/dtau = x2( T )

   At each time the outer TS needs f(tau, T), we:
     1) Extract T from Tvec.
     2) Solve the inner Van der Pol from 0..T.
     3) Get x2(T) from the final solution.
     4) Set dT/dtau = x2(T).
   ------------------------------------------------------------------- */
static PetscErrorCode RHSFunction_Outer(TS ts, PetscReal tau, Vec Tvec, Vec Tdot, void* ctx)
{
  AppCtx        *user = (AppCtx*) ctx;
  PetscReal     Tval;
  const PetscReal *tPtr;
  PetscReal     *tdotPtr;

  PetscFunctionBeginUser;
  /* 1) Get T from Tvec */
  PetscCall(VecGetArrayRead(Tvec, &tPtr));
  Tval = tPtr[0];
  PetscCall(VecRestoreArrayRead(Tvec, &tPtr));

  /* If T < 0, handle gracefully (like returning 0.0). */
  if (Tval < 0) {
    PetscCall(VecSetValue(Tdot, 0, 0.0, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(Tdot));
    PetscCall(VecAssemblyEnd(Tdot));
    PetscFunctionReturn(0);
  }

  /* 2) Reset the inner solution to initial conditions x(0)=(x0, v0). */
  PetscCall(VecSetValue(user->X, 0, user->x0, INSERT_VALUES));
  PetscCall(VecSetValue(user->X, 1, user->v0, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(user->X));
  PetscCall(VecAssemblyEnd(user->X));

  /* Reset the inner TS time to 0 and integrate to Tval. */
  PetscCall(TSSetTime(user->tsInner, 0.0));
  PetscCall(TSSetStepNumber(user->tsInner, 0));
  PetscCall(TSSetMaxTime(user->tsInner, Tval));

  /* 3) Solve the inner problem on [0, Tval]. */
  PetscCall(TSSolve(user->tsInner, user->X));

  /* 4) Read x2(Tval) from user->X. */
  {
    const PetscReal *x;
    PetscCall(VecGetArrayRead(user->X, &x));
    PetscReal x2T = x[1];
    PetscCall(VecRestoreArrayRead(user->X, &x));

    /* 5) dT/dtau = x2(Tval) */
    PetscCall(VecGetArray(Tdot, &tdotPtr));
    tdotPtr[0] = x2T;
    PetscCall(VecRestoreArray(Tdot, &tdotPtr));
  }

  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------
   Top-level routine: 
     - Create the PETSc TS for the inner system (Van der Pol).
     - Create the PETSc TS for the outer system (T in pseudo-time).
     - Solve outer system. 
     - Print final T.
   ------------------------------------------------------------------- */
int main(int argc, char** argv)
{
  PetscInitialize(&argc, &argv, NULL, NULL);

  /* 1) Create an application context to hold data. */
  AppCtx user;
  user.mu = 1.5;   /* default value for mu */
  user.x0 = 0.0;    /* initial x1(0) */
  user.v0 = 2.0;    /* initial x2(0) */

  /* Possibly read from command line:
     e.g. -mu 10.0
  */
  PetscOptionsGetReal(NULL,NULL,"-mu",&user.mu,NULL);
  PetscOptionsGetReal(NULL,NULL,"-x0",&user.x0,NULL);
  PetscOptionsGetReal(NULL,NULL,"-v0",&user.v0,NULL);

  /* 2) Create the INNER TS for Van der Pol. */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &user.tsInner));
  PetscCall(TSSetProblemType(user.tsInner, TS_NONLINEAR));
  /* For demonstration, pick a simple TS type, e.g. TSRK. 
     One can also do TSSetType(user.tsInner, TSBEULER) or TSTHETA, etc.
  */
  PetscCall(TSSetType(user.tsInner, TSTHETA));
  /* Create vector [x1, x2]. */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &user.X));
  PetscCall(VecSetSizes(user.X, PETSC_DECIDE, 2));
  PetscCall(VecSetFromOptions(user.X));
  /* Provide the RHS function for the inner system. */
  PetscCall(TSSetRHSFunction(user.tsInner, NULL, RHSFunction_Inner, &user));

  /* Optionally set some time-stepping parameters for the inner TS: 
     e.g. TSSetTimeStep(user.tsInner, 1e-3), TSSetMaxSteps(user.tsInner, 10000), etc.
  */

  /* 3) Create the OUTER TS for T(\tau). */
  TS tsOuter;
  PetscCall(TSCreate(PETSC_COMM_WORLD, &tsOuter));
  PetscCall(TSSetProblemType(tsOuter, TS_NONLINEAR));
  PetscCall(TSSetType(tsOuter, TSTHETA));

  /* Create the outer solution vector Tvec (1D). Let's initialize T0=0. */
  Vec Tvec;
  PetscCall(VecCreate(PETSC_COMM_WORLD, &Tvec));
  PetscCall(VecSetSizes(Tvec, PETSC_DECIDE, 1));
  PetscCall(VecSetFromOptions(Tvec));
  /* Let user override initial T from command line with -T0 <value>. */
  PetscReal T0 = 0.0;
  PetscOptionsGetReal(NULL,NULL,"-T0",&T0,NULL);
  PetscCall(VecSetValue(Tvec, 0, T0, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(Tvec));
  PetscCall(VecAssemblyEnd(Tvec));

  /* Provide the RHS function for the outer system. */
  PetscCall(TSSetRHSFunction(tsOuter, NULL, RHSFunction_Outer, &user));

  /* Set pseudo-time range for the outer system: 0..tau_max. */
  PetscReal tau_max = 50.0; /* default, can override -tau_max <val> */
  PetscOptionsGetReal(NULL,NULL,"-tau_max",&tau_max,NULL);

  PetscCall(TSSetTime(tsOuter, 0.0));
  PetscCall(TSSetMaxTime(tsOuter, tau_max));
  /* Optional: TSSetTimeStep(tsOuter, 0.1), etc. */

  /* 4) Solve the outer system in pseudo-time. */
  PetscCall(TSSolve(tsOuter, Tvec));

  /* 5) Extract the final T. */
  const PetscReal *tarray;
  PetscCall(VecGetArrayRead(Tvec, &tarray));
  PetscReal T_final = tarray[0];
  PetscCall(VecRestoreArrayRead(Tvec, &tarray));

  /* Print result. */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, 
           "Final T from pseudo-transient approach = %g\n", (double)T_final));

  /* 6) (Optional) Check by re-integrating Van der Pol from 0..T_final 
     to see if x2 is indeed near zero. 
     We'll do a quick check here. */

  /* Reset inner initial conditions. */
  PetscCall(VecSetValue(user.X, 0, user.x0, INSERT_VALUES));
  PetscCall(VecSetValue(user.X, 1, user.v0, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(user.X));
  PetscCall(VecAssemblyEnd(user.X));

  PetscCall(TSSetTime(user.tsInner, 0.0));
  PetscCall(TSSetStepNumber(user.tsInner, 0));
  PetscCall(TSSetMaxTime(user.tsInner, T_final));
  PetscCall(TSSolve(user.tsInner, user.X));

  /* Read final x2. */
  {
    const PetscReal *xx;
    PetscCall(VecGetArrayRead(user.X, &xx));
    PetscReal x2_final = xx[1];
    PetscCall(VecRestoreArrayRead(user.X, &xx));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, 
             "After integrating from 0..%g, x2(T) = %g\n", 
             (double)T_final, (double)x2_final));
  }

  /* Cleanup. */
  PetscCall(VecDestroy(&Tvec));
  PetscCall(TSDestroy(&tsOuter));
  PetscCall(TSDestroy(&user.tsInner));
  PetscCall(VecDestroy(&user.X));

  PetscFinalize();
  return 0;
}
