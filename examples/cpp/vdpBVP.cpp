#include <petscts.h>
#include <petscsnes.h>

/* 
   Data structure for Van der Pol.
   We store the parameter mu, and possibly anything else we need.
*/
typedef struct {
  PetscScalar mu;       /* the Van der Pol parameter */
} VDPCtx;

/* ----------------------------------------------------------------------
   VanDerPolRHSFunction - defines the ODE right-hand side f(U) for
   U = [x, y]:
     x'(t) = y
     y'(t) = mu * (1 - x^2) * y - x
   So if U = (x, y), then F = (y, mu*(1-x^2)*y - x).

   TS calls this to evaluate F(t,U).
   We assume:
     U[0] = x(t),
     U[1] = y(t).
 ---------------------------------------------------------------------- */
PetscErrorCode VanDerPolRHSFunction(TS ts, PetscReal t, Vec U, Vec F, void *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *u;
  PetscScalar       *f;
  VDPCtx            *user = (VDPCtx*)ctx;
  PetscScalar       x, y, mu;

  PetscFunctionBeginUser;
  mu = user->mu; /* get the parameter */

  ierr = VecGetArrayRead(U, &u);CHKERRQ(ierr);
  ierr = VecGetArray(F, &f);CHKERRQ(ierr);

  x = u[0]; 
  y = u[1];
  /* The ODE: x'(t)= y,  y'(t)= mu * (1 - x^2) * y - x */
  f[0] = y;
  f[1] = mu*(1.0 - x*x)*y - x;

  ierr = VecRestoreArrayRead(U, &u);CHKERRQ(ierr);
  ierr = VecRestoreArray(F, &f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------
   VanDerPolSolve - Given an initial guess for y(0)=eta, integrate 
   the Van der Pol system from t=0 to t=1 with x(0)=0, y(0)=eta.
   Then return final x(1) in *finalX.

   We'll use a basic time integrator (TSRK or TSEULER, etc.) 
   so we can see how the solution evolves. 
 ---------------------------------------------------------------------- */
PetscErrorCode VanDerPolSolve(PetscScalar eta, PetscScalar *finalX, VDPCtx *user)
{
  PetscErrorCode ierr;
  TS             ts;
  Vec            U;         /* solution vector: [x(t), y(t)] */
  PetscScalar    *uarray;
  PetscReal      t0=0.0, tf=1.0;  /* integration interval */
  
  PetscFunctionBeginUser;
  /* Create the TS object */
  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  /* We'll solve a general nonlinear ODE system */
  ierr = TSSetProblemType(ts, TS_NONLINEAR);CHKERRQ(ierr);

  /* 
     Choose a time-stepping method. 
     TSRK, TSEULER, TSCN, etc. 
     We'll pick a Runge-Kutta method for demonstration:
  */
  ierr = TSSetType(ts, TSBDF);CHKERRQ(ierr);

  /* Create the solution vector U */
  ierr = VecCreate(PETSC_COMM_WORLD, &U);CHKERRQ(ierr);
  ierr = VecSetSizes(U, PETSC_DECIDE, 2);CHKERRQ(ierr);
  ierr = VecSetFromOptions(U);CHKERRQ(ierr);

  /* Set initial conditions: x(0)=0, y(0)=eta */
  ierr = VecGetArray(U, &uarray);CHKERRQ(ierr);
  uarray[0] = 2.0;      /* x(0) */
  uarray[1] = eta;      /* y(0) = guess */
  ierr = VecRestoreArray(U, &uarray);CHKERRQ(ierr);

  /* Provide the RHSFunction for the ODE. 
     We do NOT need a Jacobian for basic time stepping, or we can use FD. */
  ierr = TSSetRHSFunction(ts, NULL, VanDerPolRHSFunction, user);CHKERRQ(ierr);

  /* Set up the time range */
  ierr = TSSetTime(ts, t0);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts, tf);CHKERRQ(ierr);
  /* We can set a step size or let PETSc adapt. 
     For a BVP shooting, we often want to ensure stable steps. 
     E.g.: 
        TSSetTimeStep(ts, 1e-3);
     or let user pick from command line 
  */

  /* 
     Let command-line options override everything. 
     This allows e.g. -ts_dt 1e-3, -ts_monitor, etc.
  */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* Integrate forward in time from t=0 to t=1 */
  ierr = TSSolve(ts, U);CHKERRQ(ierr);

  /* Now extract the final x(1) from the solution U. U = [x(1), y(1)] */
  {
    const PetscScalar *ufinal;
    ierr = VecGetArrayRead(U, &ufinal);CHKERRQ(ierr);
    *finalX = ufinal[0];
    ierr = VecRestoreArrayRead(U, &ufinal);CHKERRQ(ierr);
  }

  /* Clean up */
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------
   ShootResidual - SNES residual callback for the scalar "shooting parameter"
   We store it in a 1D vector guess = [eta].

   Then we integrate forward with that eta, measure 
       R(eta) = x(1) - 2
   We want R(eta)=0 => x(1)=2.
 ---------------------------------------------------------------------- */
PetscErrorCode ShootResidual(SNES snes, Vec guess, Vec R, void *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *g;
  PetscScalar       *r;
  VDPCtx            *user = (VDPCtx*)ctx;
  PetscScalar       eta, finalX;

  PetscFunctionBeginUser;
  /* guess = [eta], a single scalar. */
  ierr = VecGetArrayRead(guess, &g);CHKERRQ(ierr);
  eta = g[0];
  ierr = VecRestoreArrayRead(guess, &g);CHKERRQ(ierr);

  /* Integrate forward with that eta. We get final x(1). */
  ierr = VanDerPolSolve(eta, &finalX, user);CHKERRQ(ierr);

  /* 
     The boundary condition we want is x(1)=2 => finalX - 2=0. 
     So the SNES residual is R(eta)= finalX - 2. 
  */
  ierr = VecGetArray(R, &r);CHKERRQ(ierr);
  r[0] = finalX - 2.0;
  ierr = VecRestoreArray(R, &r);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------
   main - orchestrates the shooting approach:

   1. We'll treat y(0) as an unknown. 
   2. We'll create SNES with dimension=1. 
   3. SNESSetFunction(ShootResidual,...).
   4. We find the root of R(eta)=0. 
 ---------------------------------------------------------------------- */
int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  SNES           snes;      /* SNES for the 1D root solve in "eta" */
  Vec            guess;     /* 1D unknown: [eta] */
  VDPCtx         user;      /* store mu, etc. */
  PetscScalar    *arr;

  ierr = PetscInitialize(&argc,&argv,NULL,NULL); if (ierr) return ierr;

  /* 
     We'll let mu default to 1.0 unless user overrides -mu <value>.
     We'll parse it from command line. 
  */
  user.mu = 10000.0; 
  ierr = PetscOptionsGetReal(NULL,NULL,"-mu",&user.mu,NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Van der Pol BVP with mu=%g\n", (double)user.mu);

  /* 1. Create the SNES for root finding ( dimension=1 ) */
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);

  /* 2. Create the 1D vector for the unknown "eta" */
  ierr = VecCreate(PETSC_COMM_WORLD, &guess);CHKERRQ(ierr);
  ierr = VecSetSizes(guess, PETSC_DECIDE, 1);CHKERRQ(ierr);
  ierr = VecSetFromOptions(guess);CHKERRQ(ierr);

  /* 3. Initial guess for y(0). We can guess 0 or 1 or any real. */
  ierr = VecGetArray(guess, &arr);CHKERRQ(ierr);
  arr[0] = 100.0; /* try y(0)=0 as an initial guess, for example */
  ierr = VecRestoreArray(guess, &arr);CHKERRQ(ierr);

  /* 4. Set the SNES residual function */
  ierr = SNESSetFunction(snes,NULL,ShootResidual,&user);CHKERRQ(ierr);

  /* 
     We can rely on -snes_mf or a numeric derivative for the 1D problem, 
     or do a custom derivative. 
     For now, let's just use finite-difference via 
     -snes_fd or -snes_mf from the command line. 
  */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* 5. Solve for the root => "eta^*" */
  ierr = SNESSolve(snes,NULL,guess);CHKERRQ(ierr);

  /* Print the final solution for y(0) */
  {
    const PetscScalar *gfinal;
    ierr = VecGetArrayRead(guess,&gfinal);CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"Final solution: y(0) = %g\n",(double)gfinal[0]);
    ierr = VecRestoreArrayRead(guess,&gfinal);CHKERRQ(ierr);
  }

  /* Cleanup */
  ierr = VecDestroy(&guess);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}