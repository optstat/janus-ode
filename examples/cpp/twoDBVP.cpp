#include <petscts.h>
#include <petscsnes.h>


/**
 * This example demonstrates how to use PETSc to solve a boundary value problem (BVP)
 * using the shooting method. The problem is to find the initial condition y(0) such that
 * the solution x(t) satisfies x(1) = 1. The system of ODEs is given by:
 *   x'(t) = y(t)
 *   y'(t) = -x(t)
 * with initial conditions x(0) = 0 and y(0) = eta (unknown).
 * The shooting method involves solving the initial value problem (IVP) for different
 * values of eta until the boundary condition is satisfied.
 * The code uses PETSc's TS (time-stepping) and SNES (nonlinear solver) modules.
 */

typedef struct {
  // No PDE parameters needed here, but we store final x(1) for residual
  PetscScalar finalX;   // this will be set after forward solve
} AppCtx;

PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  PetscErrorCode ierr;
  const PetscScalar *u;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U, &u);CHKERRQ(ierr);
  ierr = VecGetArray(F, &f);CHKERRQ(ierr);

  // u[0] = x(t), u[1] = y(t)
  // f[0] = x'(t) = y(t)
  // f[1] = y'(t) = -x(t)
  f[0] = u[1];
  f[1] = -u[0];

  ierr = VecRestoreArrayRead(U, &u);CHKERRQ(ierr);
  ierr = VecRestoreArray(F, &f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/**
 * ShootingSolve solves the initial value problem (IVP) for the ODE system
 * given the initial condition y(0) = eta. It uses PETSc's TS module to perform
 * the time-stepping and solve the ODEs. The final value of x(1) is returned
 * through the finalX pointer.
 */
PetscErrorCode ShootingSolve(PetscScalar eta, PetscScalar *finalX, AppCtx *user)
{
  PetscErrorCode ierr;
  TS             ts;
  Vec            U;     // solution vector [x,y]
  PetscScalar    *u0;
  PetscReal      t0=0.0, tf=1.0;

  PetscFunctionBeginUser;
  // Create TS and vector
  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts, TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts, TSRK);CHKERRQ(ierr); // e.g. a Runge-Kutta method
  // Or TSEULER, TSCN, etc.

  // 2-component solution
  ierr = VecCreate(PETSC_COMM_WORLD,&U);CHKERRQ(ierr);
  ierr = VecSetSizes(U, PETSC_DECIDE, 2);CHKERRQ(ierr);
  ierr = VecSetFromOptions(U);CHKERRQ(ierr);

  // Set initial conditions: x(0)=0, y(0)=eta
  ierr = VecGetArray(U,&u0);CHKERRQ(ierr);
  u0[0] = 0.0;    // x(0)
  u0[1] = eta;    // y(0) = unknown guess
  ierr = VecRestoreArray(U,&u0);CHKERRQ(ierr);

  // Set function for ODE
  ierr = TSSetRHSFunction(ts, NULL, RHSFunction, user);CHKERRQ(ierr);

  // Set times
  ierr = TSSetTime(ts, t0);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts, tf);CHKERRQ(ierr);

  // (Optional) set time step or tolerances
  // e.g. TSSetTimeStep(ts, 0.01);

  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  // Solve forward
  ierr = TSSolve(ts, U);CHKERRQ(ierr);

  // Get final x(1)
  const PetscScalar *ufinal;
  ierr = VecGetArrayRead(U,&ufinal);CHKERRQ(ierr);
  // [x(1), y(1)]
  *finalX = ufinal[0];
  ierr = VecRestoreArrayRead(U,&ufinal);CHKERRQ(ierr);

  // Cleanup
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


PetscErrorCode ShootResidual(SNES snes, Vec guess, Vec R, void *ctx)
{
  AppCtx         *user = (AppCtx*)ctx;
  PetscErrorCode ierr;
  const PetscScalar *g;
  PetscScalar       *r;

  PetscFunctionBeginUser;
  // guess is [eta]
  ierr = VecGetArrayRead(guess, &g);CHKERRQ(ierr);
  // R is the residual [Phi(eta)]
  ierr = VecGetArray(R, &r);CHKERRQ(ierr);

  PetscScalar eta = g[0];
  PetscScalar finalX;
  // Solve forward IVP
  ierr = ShootingSolve(eta, &finalX, user);CHKERRQ(ierr);

  // Residual = finalX - 1
  r[0] = finalX - 1.0;

  ierr = VecRestoreArrayRead(guess, &g);CHKERRQ(ierr);
  ierr = VecRestoreArray(R, &r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  SNES           snes;      // we'll use SNES to do the 1D root find
  Vec            guess;     // 1D unknown [eta]
  AppCtx         user;
  PetscScalar    *arr;

  ierr = PetscInitialize(&argc,&argv,NULL,NULL); if (ierr) return ierr;

  // 1. Create SNES
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);

  // 2. Create 1D vector for the unknown (eta)
  ierr = VecCreate(PETSC_COMM_WORLD, &guess);CHKERRQ(ierr);
  ierr = VecSetSizes(guess, PETSC_DECIDE, 1);CHKERRQ(ierr);
  ierr = VecSetFromOptions(guess);CHKERRQ(ierr);

  // 3. Set an initial guess for eta
  ierr = VecGetArray(guess, &arr);CHKERRQ(ierr);
  arr[0] = 100.0;  // a guess: y(0)=0.5, for example
  ierr = VecRestoreArray(guess, &arr);CHKERRQ(ierr);

  // 4. Tell SNES the residual function (ShootResidual)
  ierr = SNESSetFunction(snes, NULL, ShootResidual, &user);CHKERRQ(ierr);

  // 5. We can use a default Jacobian-free approach for 1D or a FD derivative
  // e.g. SNESSetType(snes, SNESNEWTONLS) or from command line -snes_mf

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  // 6. Solve for the root
  ierr = SNESSolve(snes, NULL, guess);CHKERRQ(ierr);

  // 7. Final solution for eta
  const PetscScalar *finalEta;
  ierr = VecGetArrayRead(guess, &finalEta);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Optimal y(0) = %g\n",(double)finalEta[0]);
  ierr = VecRestoreArrayRead(guess, &finalEta);CHKERRQ(ierr);

  // Cleanup
  ierr = VecDestroy(&guess);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}