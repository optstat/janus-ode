#include <petscts.h>

/* A simple app context for mu. */
typedef struct {
  PetscScalar mu;
} AppCtx;

/* A shell context for the Jacobian-vector product. */
typedef struct {
  PetscScalar mu;
  Vec         X; /* Will hold the current solution. */
} ShellCtx;

/* The standard RHS function: f(x,y) = (y, mu(1-x^2)*y - x). */
static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec X, Vec F, void *ctx)
{
  const PetscScalar *x;
  PetscScalar       *f;
  AppCtx            *user = (AppCtx*)ctx;
  PetscScalar       mu    = user->mu;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArray(F, &f));

  f[0] = x[1];
  f[1] = mu*(1.0 - x[0]*x[0])*x[1] - x[0];

  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(0);
}

/* This is our shell MatMult: y = J(X)*v, where J depends on the current X. */
static PetscErrorCode MatMult_Shell(Mat M, Vec v, Vec w)
{
  ShellCtx          *shell;
  const PetscScalar *xarr;
  PetscScalar       x, y;
  const PetscScalar *varr;
  PetscScalar       vx, vy;
  PetscScalar       *warr;
  PetscScalar       mu;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(M, &shell));
  mu = shell->mu;

  /* Read the current solution X */
  PetscCall(VecGetArrayRead(shell->X, &xarr));
  x = xarr[0];
  y = xarr[1];
  PetscCall(VecRestoreArrayRead(shell->X, &xarr));

  /* Read the input vector v */
  PetscCall(VecGetArrayRead(v, &varr));
  vx = varr[0];
  vy = varr[1];
  PetscCall(VecRestoreArrayRead(v, &varr));

  /* Compute w = J(X)*v. J is 2x2. */
  PetscCall(VecGetArray(w, &warr));
  warr[0] = vy;
  warr[1] = (-1.0 - 2.0*mu*x*y)*vx + mu*(1.0 - x*x)*vy;
  PetscCall(VecRestoreArray(w, &warr));

  PetscFunctionReturn(0);
}

/* Jacobian "setup" routine for TS. We just store X in the shell context. */
static PetscErrorCode MyRHSJacobian(TS ts, PetscReal t, Vec X, Mat A, Mat P, void *ctx)
{
  ShellCtx *shell = (ShellCtx*)ctx;

  PetscFunctionBeginUser;
  /* Just copy X into shell->X so our MatMult can see it. */
  PetscCall(VecCopy(X, shell->X));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  TS        ts;
  Vec       X;
  PetscInt  n = 2;
  PetscReal t0=0.0, tf=10.0;
  PetscScalar   *x_ptr;
  AppCtx    user;
  ShellCtx  shell;
  Mat       Jshell;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

  /* Default mu=100, or read from -mu. */
  user.mu = 1.0;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-mu", &user.mu, NULL));

  /* Create the solution vector X. */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &X));
  PetscCall(VecSetSizes(X, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(X));

  /* Initial condition x(0)=2, y(0)=0. */
  PetscCall(VecGetArray(X, &x_ptr));
  x_ptr[0] = 2.0;
  x_ptr[1] = 0.0;
  PetscCall(VecRestoreArray(X, &x_ptr));

  /* Create TS. */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetType(ts, TSBEULER));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));

  /* Set the RHS function. */
  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, &user));

  /* Create a shell context. */
  shell.mu = user.mu;
  PetscCall(VecDuplicate(X, &shell.X)); /* We'll store the current solution in shell.X. */
  PetscCall(VecCopy(X, shell.X));       /* Initialize it. */

  /* Create the shell matrix Jshell. It's 2x2 in global dimension. */
  PetscCall(MatCreateShell(PETSC_COMM_WORLD, n, n, PETSC_DETERMINE, PETSC_DETERMINE, &shell, &Jshell));

  /*
    Provide our custom MatMult to the shell matrix.
    We do not store anything in the matrix itself.
  */
  PetscCall(MatShellSetOperation(Jshell, MATOP_MULT, (void(*)(void))MatMult_Shell));

  /* 
    Finally, tell TS that Jshell is our Jacobian. We pass MyRHSJacobian, which updates shell->X.
    We'll use the same matrix for 'A' and 'P' (i.e. no separate preconditioner).
  */
  PetscCall(TSSetRHSJacobian(ts, Jshell, Jshell, MyRHSJacobian, &shell));

  /* Time-stepping options. */
  PetscCall(TSSetTime(ts, t0));
  PetscCall(TSSetMaxTime(ts, tf));
  PetscCall(TSSetTimeStep(ts, 0.01));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));

  /* Solve */
  PetscCall(TSSolve(ts, X));

  /* Report results. */
  {
    PetscInt steps;
    PetscCall(TSGetStepNumber(ts, &steps));
    PetscPrintf(PETSC_COMM_WORLD,"Van der Pol solution with user-provided J*v:\n");
    PetscPrintf(PETSC_COMM_WORLD,"  final time = %g\n", (double)tf);
    const PetscScalar *xfinal;
    PetscCall(VecGetArrayRead(X, &xfinal));
    PetscPrintf(PETSC_COMM_WORLD,"  x(final) = [%g, %g]\n",(double)xfinal[0],(double)xfinal[1]);
    PetscCall(VecRestoreArrayRead(X, &xfinal));
    PetscPrintf(PETSC_COMM_WORLD,"  steps taken = %d\n",(int)steps);
  }

  /* Cleanup */
  PetscCall(MatDestroy(&Jshell));
  PetscCall(VecDestroy(&shell.X));
  PetscCall(VecDestroy(&X));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}
