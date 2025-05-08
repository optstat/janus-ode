#include <petscts.h>

/* ------------------------------------------------------------------------
   Van der Pol oscillator:

     x1'(t) = x2
     x2'(t) = mu*(1 - x1^2)*x2 - x1

   We'll solve it implicitly. We do a matrix-free approach for J(x):
     J(x)*v  =>  df/dx * v

   We'll do forward sensitivities (w.r.t. x1(0), x2(0)) using the older
   TSForwardSetSensitivities() from PETSc 3.15–3.22.

   ------------------------------------------------------------------------ */

typedef struct {
  PetscReal  mu;       // parameter in Van der Pol
  PetscReal  alpha;    // "shift" factor for I - alpha*J
  Vec        Xwork;    // store current solution x(t) so MatMult can read it
} AppCtx;

/* ---- f(t,x) = RHSFunction:  f(x) = ( x2,  mu(1-x1^2)x2 - x1 ) ---- */
static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
{
  AppCtx            *user = (AppCtx*)ctx;
  const PetscScalar *x;
  PetscScalar       *f;
  const PetscReal    mu = user->mu;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X,&x));
  PetscCall(VecGetArray(F,&f));

  f[0] = x[1];
  f[1] = mu*(1.0 - x[0]*x[0])*x[1] - x[0];

  PetscCall(VecRestoreArrayRead(X,&x));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------
   MatMult for (I - alpha*J) or just J — whichever style you prefer:

   We'll define: y = (I - alpha * J(x)) * v

   J(x) = [[0,                 1],
           [-2 mu x1 x2 - 1,   mu(1 - x1^2)]]

   =>   (I - alpha*J) * v = v - alpha*(J*v)

   We'll store the "alpha" and "x" in the AppCtx each time IJacobian is called.
   Then the solver can do Newton steps in a purely matrix‐free manner.
   ------------------------------------------------------------------------ */
static PetscErrorCode ShellMatMult(Mat A, Vec v, Vec y)
{
  AppCtx            *user;
  const PetscScalar *x,*vv;
  PetscScalar       *yy;
  PetscReal          mu, alpha;
  PetscReal          x1,x2;

  PetscFunctionBeginUser;
  /* Grab the context and read current alpha, x(t), etc. */
  PetscCall(MatShellGetContext(A,(void**)&user));
  mu    = user->mu;
  alpha = user->alpha;

  /* Current solution x(t) is in user->Xwork */
  PetscCall(VecGetArrayRead(user->Xwork,&x));
  x1 = x[0];
  x2 = x[1];
  PetscCall(VecRestoreArrayRead(user->Xwork,&x));

  /* v => vv, then compute (I - alpha J)*v. */
  PetscCall(VecGetArrayRead(v,&vv));
  PetscCall(VecGetArray(y,&yy));

  /* First J(x)*v:  Jv = [ vv[1];
   *                     (-2 mu x1 x2 - 1)*vv[0] + mu(1 - x1^2)*vv[1] ]
   */
  PetscScalar Jv0 = vv[1];
  PetscScalar Jv1 = (-2.0*mu*x1*x2 - 1.0)*vv[0] + mu*(1.0 - x1*x1)*vv[1];

  /* Then (I - alpha*J)*v = v - alpha*(Jv) */
  yy[0] = vv[0] - alpha * Jv0;
  yy[1] = vv[1] - alpha * Jv1;

  PetscCall(VecRestoreArrayRead(v,&vv));
  PetscCall(VecRestoreArray(y,&yy));

  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------
   IJacobianShell: This is called by TS to form the operator (I - alpha*J).
   But since we're matrix-free, we don't do MatSetValues().

   We just store alpha and the current X in user->Xwork. Then ShellMatMult()
   can do (I - alpha*J)*v.
   ------------------------------------------------------------------------ */
static PetscErrorCode IJacobianShell(TS ts, PetscReal t, Vec X, Vec Xdot,
                                     PetscReal alpha, Mat A, Mat B, void *ctx)
{
  AppCtx *user = (AppCtx*)ctx;

  PetscFunctionBeginUser;
  /* store the current solution in user->Xwork */
  PetscCall(VecCopy(X, user->Xwork));
  /* store alpha => this will be used inside ShellMatMult() */
  user->alpha = alpha;

  /* That’s all: do NOT call MatSetValues() on a MAT_SHELL! */
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------
   main()
   ------------------------------------------------------------------------ */
int main(int argc,char **argv)
{
  TS        ts;
  Vec       X;       // state vector (x1, x2)
  Mat       J;       // shell matrix for (I - alpha * df/dx)
  Mat       S;       // forward-sensitivity matrix
  AppCtx    *user;
  PetscInt  steps;
  PetscReal dt       = 0.0001;
  PetscReal tmax     = 1.0;   // shorter run for example
  PetscReal mu       = 1.0;
  PetscScalar x0[2]  = {2.0, 0.0};  // initial conditions
  PetscInt  numParams = 2;
  PetscInt  n         = 2;   // system dimension

  PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));

  /* Handle some options from command line */
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Van der Pol options","");
  PetscCall(PetscOptionsReal("-mu","Stiffness parameter","",mu,&mu,NULL));
  PetscCall(PetscOptionsReal("-ts_dt","Time step","",dt,&dt,NULL));
  PetscCall(PetscOptionsReal("-ts_max_time","Final time","",tmax,&tmax,NULL));
  PetscOptionsEnd();

  /* Create user context */
  PetscCall(PetscNew(&user));
  user->mu    = mu;
  user->alpha = 1.0;

  /* user->Xwork will hold the "current" state so ShellMatMult sees it */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&user->Xwork));
  PetscCall(VecSetSizes(user->Xwork,PETSC_DECIDE,n));
  PetscCall(VecSetFromOptions(user->Xwork));
  PetscCall(VecSet(user->Xwork,0.0));

  /* Create solution vector X, set initial condition. */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&X));
  PetscCall(VecSetSizes(X,PETSC_DECIDE,n));
  PetscCall(VecSetFromOptions(X));
  {
    PetscInt ix[2] = {0,1};
    PetscCall(VecSetValues(X,2,ix,x0,INSERT_VALUES));
    PetscCall(VecAssemblyBegin(X));
    PetscCall(VecAssemblyEnd(X));
  }

  /* Create TS, set the ODE function f(x). */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetRHSFunction(ts,NULL,RHSFunction,(void*)user));
  
  /* Create a shell matrix for (I - alpha J). We'll define MatMult = ShellMatMult. */
  PetscCall(MatCreateShell(PETSC_COMM_WORLD,n,n,n,n,(void*)user,&J));
  PetscCall(MatShellSetOperation(J,MATOP_MULT,(void(*)(void))ShellMatMult));
  
  /* set the IJacobian to be that shell matrix, in a purely matrix-free sense */
  PetscCall(TSSetIJacobian(ts,J,J,IJacobianShell,(void*)user));

  /* Make it an implicit integrator. Let's pick ARKIMEX for demonstration. */
  PetscCall(TSSetType(ts, TSARKIMEX));
  PetscCall(TSSetTimeStep(ts,dt));
  PetscCall(TSSetMaxTime(ts,tmax));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetFromOptions(ts));

  /* Provide the initial solution X to TS. */
  PetscCall(TSSetSolution(ts,X));

  /* Forward-sensitivity matrix S (size n x numParams = 2x2).
     Each column is d x(t)/d x_i(0).  */
  {
    PetscCall(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,numParams,NULL,&S));
    PetscCall(MatSetUp(S));

    /* At t=0,  d x(0)/d x(0) = I => S(0,0)=1, S(1,1)=1 */
    PetscCall(MatZeroEntries(S));
    PetscCall(MatSetValue(S,0,0,1.0,INSERT_VALUES)); /* d x1(0)/d x1(0)=1 */
    PetscCall(MatSetValue(S,1,1,1.0,INSERT_VALUES)); /* d x2(0)/d x2(0)=1 */
    PetscCall(MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(S,  MAT_FINAL_ASSEMBLY));
  }

  /* Old PETSc 3.22 call to carry S(t) along with the state integration. */
  PetscCall(TSForwardSetSensitivities(ts,numParams,S));

  /* Solve for x(t) and also integrate forward the columns of S(t). */
  PetscCall(TSSolve(ts,X));
  PetscCall(TSGetStepNumber(ts,&steps));

  /* Print final solution, final sensitivities. */
  {
    PetscReal tf;
    PetscCall(TSGetTime(ts,&tf));
    PetscPrintf(PETSC_COMM_WORLD,"Final time = %g\n",(double)tf);

    {
      const PetscScalar *xfinal;
      PetscCall(VecGetArrayRead(X,&xfinal));
      PetscPrintf(PETSC_COMM_WORLD,"  x(final) = [%g, %g]\n",
                  (double)xfinal[0], (double)xfinal[1]);
      PetscCall(VecRestoreArrayRead(X,&xfinal));
    }

    {
      PetscInt    row[2] = {0, 1};
      PetscInt    col0[1] = {0}, col1[1] = {1};
      PetscScalar vals[2];

      /* d x(tf)/d x1(0) => column 0 */
      PetscCall(MatGetValues(S,2,row,1,col0,vals));
      PetscPrintf(PETSC_COMM_WORLD,
          "  Sensitivity wrt x1(0): [%g, %g]\n",
          (double)vals[0], (double)vals[1]);

      /* d x(tf)/d x2(0) => column 1 */
      PetscCall(MatGetValues(S,2,row,1,col1,vals));
      PetscPrintf(PETSC_COMM_WORLD,
          "  Sensitivity wrt x2(0): [%g, %g]\n",
          (double)vals[0], (double)vals[1]);
    }
  }

  /* Cleanup */
  PetscCall(MatDestroy(&S));
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&user->Xwork));
  PetscCall(PetscFree(user));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
}
