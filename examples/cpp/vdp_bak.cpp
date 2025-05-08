#include <petscts.h>
#include <petscsnes.h>

/* 
   Relaxation method for solving a BVP problem from the VDP ODE
   of the form M* Udot = F(U) with a mass matrix M.
   This method uses shooting to solve the BVP.
*/
typedef struct {
  PetscScalar mu;       /* the Van der Pol parameter */
} VDPCtx;

/* 
   VanDerPolIFunction - defines G(t,U,Udot) = M Udot - F(U) = 0.
   
   Here, M = [[1, 0],[0, 2]] for demonstration:
     G0 = 1 * xdot - (y),
     G1 = 2 * ydot - (mu*(1 - x^2)*y - x).

   We assume:
     U = [ x, y ],
     Udot = [ xdot, ydot ].

   TS calls this for implicit ODE/DAE integration.
*/
PetscErrorCode VanDerPolIFunction(TS ts, PetscReal t,
                                  Vec U, Vec Udot, Vec G, 
                                  void *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *u,*udot;
  PetscScalar       *g;
  VDPCtx            *user = (VDPCtx*)ctx;
  PetscScalar       x, y, xdot, ydot, mu;

  PetscFunctionBeginUser;
  mu = user->mu; /* get the parameter */

  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecGetArray(G,&g);CHKERRQ(ierr);

  x     = u[0];
  y     = u[1];
  xdot  = udot[0];
  ydot  = udot[1];

  /* M*Udot - F(U) = 0 */
  /* M = diag(1,2). So: 
       G0 = 1*xdot - ( y ),
       G1 = 2*ydot - ( mu*(1 - x^2)*y - x ).
  */
  g[0] = xdot - y;
  g[1] = 2.0*ydot - (mu*(1.0 - x*x)*y - x);

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecRestoreArray(G,&g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------
   VanDerPolSolve - same shooting-time procedure:
   Integrate from t=0 to t=1 with x(0)=2, y(0)=eta.
   Return final x(1) in *finalX.
 ---------------------------------------------------------------------- */
PetscErrorCode VanDerPolSolve(PetscScalar eta, 
                              PetscScalar *finalX, 
                              VDPCtx *user)   
{
  PetscErrorCode ierr;
  TS             ts;
  Vec            U;         /* solution vector: [x(t), y(t)] */
  PetscScalar    *uarray;
  PetscReal      t0=0.0, tf=1.0;  /* integration interval */
  
  PetscFunctionBeginUser;
  /* Create the TS object */
  ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts, TS_NONLINEAR);CHKERRQ(ierr);

  /* 
     Choose a time-stepping method. 
     We'll pick BDF for demonstration (implicit).
  */
  ierr = TSSetType(ts, TSBDF);CHKERRQ(ierr);

  /* Create the solution vector U */
  ierr = VecCreate(PETSC_COMM_WORLD,&U);CHKERRQ(ierr);
  ierr = VecSetSizes(U,PETSC_DECIDE,2);CHKERRQ(ierr);
  ierr = VecSetFromOptions(U);CHKERRQ(ierr);

  /* Set initial conditions: x(0)=2, y(0)=eta */
  ierr = VecGetArray(U, &uarray);CHKERRQ(ierr);
  uarray[0] = 2.0;  /* x(0) */
  uarray[1] = eta;  /* y(0) */
  ierr = VecRestoreArray(U, &uarray);CHKERRQ(ierr);

  /* Provide the IFunction for the DAE/ODE:
       G(t,U,Udot) = M Udot - F(U) = 0
     We do NOT need IJacobian if we allow FD approximations. */
  ierr = TSSetIFunction(ts, NULL, VanDerPolIFunction, user);CHKERRQ(ierr);

  /* Set up the time range */
  ierr = TSSetTime(ts,t0);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,tf);CHKERRQ(ierr);

  /* Optionally set a step size or let PETSc adapt. */
  /* TSSetTimeStep(ts, 1e-3); */

  /* Let command-line options override everything. */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* Integrate forward in time from t=0 to t=1 */
  ierr = TSSolve(ts, U);CHKERRQ(ierr);

  /* Extract x(1) from the solution U = [x(1), y(1)] */
  {
    const PetscScalar *ufinal;
    ierr = VecGetArrayRead(U,&ufinal);CHKERRQ(ierr);
    *finalX = ufinal[0];
    ierr = VecRestoreArrayRead(U,&ufinal);CHKERRQ(ierr);
  }

  /* Clean up */
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------
   ShootResidual - same as before: R(eta) = x(1)-2
 ---------------------------------------------------------------------- */
PetscErrorCode ShootResidual(SNES snes, Vec guess, Vec R, void *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *g;
  PetscScalar       *r;
  VDPCtx            *user = (VDPCtx*)ctx;
  PetscScalar       eta, finalX;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(guess, &g);CHKERRQ(ierr);
  eta = g[0];
  ierr = VecRestoreArrayRead(guess, &g);CHKERRQ(ierr);

  /* Solve the ODE (with mass matrix) forward */
  ierr = VanDerPolSolve(eta, &finalX, user);CHKERRQ(ierr);

  /* Boundary condition x(1) = 2 => residual = finalX - 2 */
  ierr = VecGetArray(R, &r);CHKERRQ(ierr);
  r[0] = finalX - 2.0;
  ierr = VecRestoreArray(R, &r);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------
   main - same shooting orchestrator
 ---------------------------------------------------------------------- */
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  SNES           snes;      /* SNES for root solve in "eta" */
  Vec            guess;     /* 1D unknown: [eta] */
  VDPCtx         user;      /* store mu, etc. */
  PetscScalar    *arr;
 
  ierr = PetscInitialize(&argc,&argv,NULL,NULL); if (ierr) return ierr;
 
  /* Let mu default to 10000 unless user overrides -mu <value> */
  user.mu = 100.0;
  ierr = PetscOptionsGetReal(NULL,NULL,"-mu",&user.mu,NULL);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Van der Pol BVP with mu=%g\n",(double)user.mu);
 
  /* 1. Create the SNES (dimension=1) */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
 
  /* 2. Create the 1D vector for unknown "eta" = y(0) */
  ierr = VecCreate(PETSC_COMM_WORLD,&guess);CHKERRQ(ierr);
  ierr = VecSetSizes(guess,PETSC_DECIDE,1);CHKERRQ(ierr);
  ierr = VecSetFromOptions(guess);CHKERRQ(ierr);
 
  /* 3. Initial guess for y(0) */
  ierr = VecGetArray(guess, &arr);CHKERRQ(ierr);
  arr[0] = 100.0; /* e.g., guess y(0) = 100 */
  ierr = VecRestoreArray(guess, &arr);CHKERRQ(ierr);
 
  /* 4. Set the SNES residual: R(eta)= x(1)-2 */
  ierr = SNESSetFunction(snes,NULL,ShootResidual,&user);CHKERRQ(ierr);
 
  /* We'll rely on -snes_fd or -snes_mf for derivative if needed */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
 
  /* 5. Solve for the root => "eta^*" */
  ierr = SNESSolve(snes,NULL,guess);CHKERRQ(ierr);
 
  /* Print final solution for y(0) */
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