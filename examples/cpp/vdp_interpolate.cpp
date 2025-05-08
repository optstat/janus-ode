#include <petscts.h>

/* 
Van der Pol oscillator:

x1' = x2
x2' = mu * (1 - x1^2) * x2 - x1

We'll store the state as X = (x1, x2). Parameter mu > 0.

We want to solve from t=0 to t=1, 
and then get the solution at t=0.75 using TSInterpolate.
*/

/* Context to hold parameters if needed */
typedef struct {
  PetscReal mu;
} AppCtx;

/* 
IFunction for the implicit ODE form F(t,X,Xdot) = 0 
If using an implicit method in PETSc, we set: 
    IFunction = Xdot - f(X,t) = 0 
Here we show the standard form so TS can call it.
*/
static PetscErrorCode IFunction(TS ts, PetscReal t, Vec X, Vec Xdot, Vec F, void *ctx)
{
  const PetscReal *x, *xdot;
  PetscReal       *f;
  AppCtx          *user = (AppCtx*)ctx;
  PetscReal       mu    = user->mu;

  PetscFunctionBeginUser;
  VecGetArrayRead(X, &x);
  VecGetArrayRead(Xdot, &xdot);
  VecGetArray(F, &f);

  /* 
      x = (x1, x2)
      xdot = (x1dot, x2dot)
      Van der Pol f(x) = ( x2,
                          mu*(1 - x1^2)*x2 - x1 )
      IFunction: F = Xdot - f(x,t) 
  */

  f[0] = xdot[0] - x[1];
  f[1] = xdot[1] - (mu*(1 - x[0]*x[0])*x[1] - x[0]);

  VecRestoreArrayRead(X, &x);
  VecRestoreArrayRead(Xdot, &xdot);
  VecRestoreArray(F, &f);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  TS             ts;             /* time stepper */
  Vec            X;              /* solution vector */
  PetscReal      dt = 0.01;
  PetscMPIInt    size;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Example is uniprocessor only");

  user.mu = 25.0; /* Example mu parameter, can override via -mu <value> */

  /* Create vector for the 2D state */
  ierr = VecCreate(PETSC_COMM_WORLD, &X);CHKERRQ(ierr);
  ierr = VecSetSizes(X, PETSC_DECIDE, 2);CHKERRQ(ierr);
  ierr = VecSetFromOptions(X);CHKERRQ(ierr);

  /* Set initial condition: (x1(0), x2(0)) */
  {
    PetscScalar *x;
    ierr = VecGetArray(X, &x);CHKERRQ(ierr);
    x[0] = 2.0; /* x1(0) */
    x[1] = 0.0; /* x2(0) */
    ierr = VecRestoreArray(X, &x);CHKERRQ(ierr);
  }

  /* Create TS and set type (e.g., BDF or Rosenbrock) */
  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts, TS_NONLINEAR);CHKERRQ(ierr);

  /* We'll use the IFunction approach for implicit solves */
  ierr = TSSetIFunction(ts, NULL, IFunction, &user);CHKERRQ(ierr);

  /* Set some basic time-step and range options */
  ierr = TSSetTime(ts, 0.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts, dt);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts, 10.0);CHKERRQ(ierr);

  /* Set the TS to save its trajectory */
  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);

  /* Let user pick method at runtime, e.g., -ts_type rosw, -ts_type bdf, etc. 
      or we can fix it in code:
      TSSetType(ts, TSBDF);
  */

  /* We'll allow adaptive stepping by default. 
      You can control adaptivity via command-line opts: -ts_adapt_type xxx
  */

  /* If we want to attempt interpolation, we must ensure 
      TS methods that implement TSInterpolate. BDF, Theta, some Rosenbrock do.
  */

  /* Set from command line to allow user customizations, e.g., -ts_type bdf */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* Provide the initial condition to TS (the "current" solution at t=0) */
  ierr = TSSetSolution(ts, X);CHKERRQ(ierr);

  /* Solve ODE from t=0 to t=1 */
  ierr = TSSolve(ts, X);CHKERRQ(ierr);

  /* Retrieve the TSTrajectory object */
  TSTrajectory tj;
  ierr = TSGetTrajectory(ts, &tj);CHKERRQ(ierr);

  /* Desired interpolation time */
  PetscReal t_interp = 7.0;

  /* Create vector for interpolated solution */
  Vec U_interp;
  ierr = VecDuplicate(X, &U_interp);CHKERRQ(ierr);

  /* Interpolate solution at t_interp */
  ierr = TSTrajectoryGetVecs(tj, ts, PETSC_DECIDE, &t_interp, U_interp, NULL);CHKERRQ(ierr);

  /* Print the interpolated solution */
  PetscPrintf(PETSC_COMM_WORLD, "\nInterpolated solution at t=%g:\n", (double)t_interp);
  VecView(U_interp, PETSC_VIEWER_STDOUT_WORLD);

  /* Cleanup */
  ierr = VecDestroy(&U_interp);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
