#include <petscts.h>

/* A simple context to store the parameter mu. */
typedef struct {
  PetscScalar mu;
} AppCtx;

/* ------------------------------------------------------------------
   RHSFunction:  f(X) = ( y, mu(1 - x^2)*y - x ).
   This is dX/dt, used by PETSc in "RHS" form: dot(X) = f(X).
   ------------------------------------------------------------------ */
static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec X, Vec F, void *ctx)
{
  const PetscScalar *x;
  PetscScalar       *f;
  AppCtx            *user = (AppCtx*)ctx;
  PetscScalar       mu    = user->mu;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X, &x)); /* x[0] = x, x[1] = y */
  PetscCall(VecGetArray(F, &f));

  f[0] = x[1];
  f[1] = mu*(1.0 - x[0]*x[0])*x[1] - x[0];

  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArray(F, &f));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------
   RHSJacobian:  J(X) = dF/dX, the 2x2 matrix:
                  [   0,            1            ]
                  [ -1 - 2mu*x*y,   mu(1 - x^2)  ]
   We form it explicitly and insert into "Mat P".
   ------------------------------------------------------------------ */
static PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec X, Mat A, Mat P, void *ctx)
{
  AppCtx            *user = (AppCtx*)ctx;
  const PetscScalar *xarr;
  PetscScalar       mu = user->mu;
  PetscScalar       x, y;
  PetscScalar       J[4];     /* will store the 2x2 Jacobian entries */
  PetscInt          row[2], col[2];

  PetscFunctionBeginUser;
  /* Access current state X. */
  PetscCall(VecGetArrayRead(X, &xarr));
  x = xarr[0];
  y = xarr[1];
  PetscCall(VecRestoreArrayRead(X, &xarr));

  /* Fill the 2x2 Jacobian for f(x,y). */
  /* df1/dx = 0, df1/dy = 1  */
  /* df2/dx = -1 - 2 mu x y,  df2/dy = mu (1 - x^2). */
  J[0] = 0.0;              /* (0,0) */
  J[1] = 1.0;              /* (0,1) */
  J[2] = -1.0 - 2.0*mu*x*y;/* (1,0) */
  J[3] = mu*(1.0 - x*x);   /* (1,1) */

  /* The pattern is row-major if we do:
       J[0] -> (row=0,col=0)
       J[1] -> (row=0,col=1)
       J[2] -> (row=1,col=0)
       J[3] -> (row=1,col=1)
  */

  /* Insert the values into P (and A). It's a tiny dense submatrix. */
  row[0] = 0;  row[1] = 1;
  col[0] = 0;  col[1] = 1;

  PetscCall(MatZeroEntries(P)); /* For a small 2x2, we can just zero it out first. */
  PetscCall(MatSetValues(P, 2, row, 2, col, J, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));

  /* Typically we do the same with A if it's a different matrix. Here A == P. */
  if (A != P) {
    PetscCall(MatZeroEntries(A));
    PetscCall(MatSetValues(A, 2, row, 2, col, J, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  }

  PetscFunctionReturn(0);
}

int main(int argc, char**argv)
{
  TS             ts;     /* time integrator */
  Vec            X;      /* solution vector */
  Mat            J;      /* Jacobian matrix */
  AppCtx         user;   /* user-defined context */
  PetscInt       n = 2;  /* dimension: (x,y) */
  PetscScalar   *x_ptr;
  PetscReal      t0=0.0, tf=100.0;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

  /* Parse command-line for the mu parameter (default mu=1.0). */
  user.mu = 100.0;
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-mu", &user.mu, NULL));

  /* Create the solution vector X, dimension 2. */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &X));
  PetscCall(VecSetSizes(X, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(X));

  /* Set initial conditions: x(0)=2, y(0)=0. */
  PetscCall(VecGetArray(X, &x_ptr));
  x_ptr[0] = 2.0; /* x */
  x_ptr[1] = 0.0; /* y */
  PetscCall(VecRestoreArray(X, &x_ptr));

  /* Create the TS solver and choose Backward Euler (an implicit method). */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetType(ts, TSBEULER));  
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR); /* We have a nonlinear ODE. */);

  /* Provide the ODE function dot(X) = RHS(X). */
  PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, &user));

  /* Create a 2x2 Jacobian matrix. We'll form it explicitly. */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &J));
  PetscCall(MatSetSizes(J, n, n, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetType(J, MATAIJ)); 
    /* Could also do MATDENSE for a 2x2, or MATSBAIJ. MATAIJ is typical for general sparse. */
  PetscCall(MatSetUp(J));

  /* Provide the Jacobian to TS. TS will call RHSJacobian() to fill it. */
  PetscCall(TSSetRHSJacobian(ts, J, J, RHSJacobian, &user));

  /* Set time domain and some TS options. */
  PetscCall(TSSetTime(ts, t0));
  PetscCall(TSSetMaxTime(ts, tf));
  PetscCall(TSSetTimeStep(ts, 0.01));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));

  /* Solve the ODE system. */
  PetscCall(TSSolve(ts, X));

  /* Report final solution and time steps used. */
  {
    PetscInt steps;
    PetscCall(TSGetStepNumber(ts, &steps));
    PetscPrintf(PETSC_COMM_WORLD,"Van der Pol solution with Backward Euler:\n");
    PetscPrintf(PETSC_COMM_WORLD,"  final time = %g\n",(double)tf);

    const PetscScalar *xfinal;
    PetscCall(VecGetArrayRead(X, &xfinal));
    PetscPrintf(PETSC_COMM_WORLD,"  x(final) = [%g, %g]\n",(double)xfinal[0],(double)xfinal[1]);
    PetscCall(VecRestoreArrayRead(X, &xfinal));

    PetscPrintf(PETSC_COMM_WORLD,"  steps taken = %d\n",(int)steps);
  }

  /* Clean up. */
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&X));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}
