#include <petscts.h>

/* 
  Context for Van der Pol parameters, if needed (e.g., mu).
*/
typedef struct {
  PetscReal mu; /* The mu parameter in Van der Pol */
} VDPContext;

/* 
   IFunction for the Van der Pol system in implicit form:

   F(t, Y, Ydot) = Ydot - f(t,Y) = 0.
   
   Here, Y = (y1, y2).
   So:
     F1 = Ydot1 - y2
     F2 = Ydot2 - mu * (1 - y1^2) * y2 + y1
*/
PetscErrorCode IFunctionVDP(TS ts, PetscReal t, Vec Y, Vec Ydot, Vec F, void *ctx)
{
  const PetscScalar *y, *ydot;
  PetscScalar       *f;
  VDPContext        *user = (VDPContext*)ctx;
  PetscReal         mu    = user->mu;

  PetscFunctionBeginUser;

  /* Get array access to Y, Ydot, F */
  VecGetArrayRead(Y,&y);
  VecGetArrayRead(Ydot,&ydot);
  VecGetArray(F,&f);

  /* y1, y2 */
  PetscReal y1 = y[0];
  PetscReal y2 = y[1];

  /* ydot1, ydot2 */
  PetscReal ydot1 = ydot[0];
  PetscReal ydot2 = ydot[1];

  /* Fill F = Ydot - f(t,Y) */
  f[0] = ydot1 - y2;
  f[1] = ydot2 - mu*(1.0 - y1*y1)*y2 + y1;

  VecRestoreArrayRead(Y,&y);
  VecRestoreArrayRead(Ydot,&ydot);
  VecRestoreArray(F,&f);

  PetscFunctionReturn(0);
}

/*
   (Optional) IJacobian for the Van der Pol system.

   J = dF/dY + shift * dF/dYdot
   But since F = Ydot - f(Y), we have:
     dF/dYdot = I (the 2x2 identity)
     dF/dY    = -df/dY
   So Jacobian = shift*I - df/dY

   df/dY = 
       df1/dy1 =  0,         df1/dy2 = -1
       df2/dy1 = derivative of [-mu*(1-y1^2)*y2 + y1] w.r.t y1
                = -mu[-2*y1*y2] + 1 = 1 + 2*mu*y1*y2
       df2/dy2 = -mu*(1-y1^2)
*/
PetscErrorCode IJacobianVDP(TS ts, PetscReal t, Vec Y, Vec Ydot, PetscReal shift,
                            Mat A, Mat B, void *ctx)
{
  const PetscScalar *y;
  VDPContext        *user = (VDPContext*)ctx;
  PetscReal         mu    = user->mu;

  PetscFunctionBeginUser;
  VecGetArrayRead(Y,&y);

  PetscReal y1 = y[0];
  PetscReal y2 = y[1];

  /* Jacobian entries. 
     We want to set:  A = shift*I - df/dY.
     That is a 2x2 matrix:
         [ shift - dF1/dy1,    - dF1/dy2
           - dF2/dy1,          shift - dF2/dy2 ]
     But recall F1 = ydot1 - y2, F2 = ydot2 - mu(1-y1^2)*y2 + y1
     => df1/dy1 = 0, df1/dy2 = -1
        df2/dy1 = d/dy1 of [-mu(1-y1^2)*y2 + y1] = +1 + 2*mu*y1*y2
        df2/dy2 = -mu(1-y1^2)
  */

  PetscReal df1dy1 = 0.0;
  PetscReal df1dy2 = -1.0;

  PetscReal df2dy1 = 1.0 + 2.0*mu*y1*y2; 
  PetscReal df2dy2 = -mu*(1.0 - y1*y1);

  /* So final matrix entries (since J = shift*I - dF/dY): */
  PetscReal J11 = shift - df1dy1; /* = shift - 0 = shift */
  PetscReal J12 =        - df1dy2; /* = +1 */
  PetscReal J21 =        - df2dy1; /* = - (1 + 2 mu y1 y2) */
  PetscReal J22 = shift - df2dy2;  /* = shift - [ -mu(1-y1^2) ] = shift + mu(1-y1^2) */

  MatSetValue(B,0,0,J11,INSERT_VALUES);
  MatSetValue(B,0,1,J12,INSERT_VALUES);
  MatSetValue(B,1,0,J21,INSERT_VALUES);
  MatSetValue(B,1,1,J22,INSERT_VALUES);

  VecRestoreArrayRead(Y,&y);

  MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);

  if (A != B) {
    MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
  }

  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;       /* time-stepper */
  Vec            Y;        /* solution vector */
  Mat            A;        /* Jacobian matrix */
  PetscScalar    *y_ptr;
  PetscReal      t0 = 0.0, tf=1.0;
  PetscInt       n=2;      /* 2D system */
  PetscErrorCode ierr;
  VDPContext     user;     /* context for Van der Pol parameter */

  ierr = PetscInitialize(&argc,&argv,NULL,NULL);CHKERRQ(ierr);

  /* Set a default mu, can be overridden via -mu <value> */
  user.mu = 100.0;
  tf = 3*user.mu; /* Set final time to 3*mu */
  ierr = PetscOptionsGetReal(NULL,NULL,"-mu",&user.mu,NULL);CHKERRQ(ierr);

  /* Create the solution vector and set initial conditions */
  ierr = VecCreate(PETSC_COMM_WORLD,&Y);CHKERRQ(ierr);
  ierr = VecSetSizes(Y,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(Y);CHKERRQ(ierr);

  /* Typical initial conditions for Van der Pol */
  ierr = VecGetArray(Y,&y_ptr);CHKERRQ(ierr);
  y_ptr[0] = 2.0;   /* y1(0) */
  y_ptr[1] = 0.0;   /* y2(0) */
  ierr = VecRestoreArray(Y,&y_ptr);CHKERRQ(ierr);

  /* Create the TS (Time-Stepping) solver and set type to BDF (implicit) */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  //set the order
  

  /* Set the time domain and initial time */
  ierr = TSSetTime(ts, t0);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts, tf);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts, 0.001);CHKERRQ(ierr);

  /* Tell the TS we will provide an implicit form F(t,Y,Ydot)=0 */
  ierr = TSSetIFunction(ts,NULL,IFunctionVDP,&user);CHKERRQ(ierr);

  /* Create a matrix for the Jacobian and set IJacobian function */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = TSSetIJacobian(ts,A,A,IJacobianVDP,&user);CHKERRQ(ierr);

  /* ----------------------------------------------------------------
     Now specify GMRES + LU preconditioning for the *linear* solves.
     We can do so in code or via the command line. Shown here in code:
  ---------------------------------------------------------------- */
  {
    SNES snes; /* the nonlinear solver inside TS */
    KSP  ksp;  /* the linear solver inside SNES */
    PC   pc;   /* the preconditioner */

    ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);

    /* Use GMRES as the Krylov method */
    ierr = KSPSetType(ksp,KSPGMRES);CHKERRQ(ierr);

    /* Use LU as the preconditioner */
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    //ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
    //use cholesky as preconditioner
    ierr = PCSetType(pc,PCCHOLESKY);CHKERRQ(ierr);

    /* Optionally pick a specific LU solver package: e.g., MUMPS, SuperLU, etc. */
    /* ierr = PCFactorSetMatSolverType(pc,MATSOLVERMUMPS);CHKERRQ(ierr); */
  }

  /* Allow command line options to override these settings if desired */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* Solve the system */
  ierr = TSSolve(ts,Y);CHKERRQ(ierr);

  /* Print final solution */
  PetscReal t_final;
  ierr = TSGetSolveTime(ts,&t_final);CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Final time reached %g\n",(double)t_final);

  VecGetArray(Y,&y_ptr);
  PetscPrintf(PETSC_COMM_WORLD,"Solution at final time: y1=%g, y2=%g\n",
              (double)y_ptr[0], (double)y_ptr[1]);
  VecRestoreArray(Y,&y_ptr);

  /* Clean up */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&Y);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}
