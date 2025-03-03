#ifndef PETSC_VDP_JV_EXAMPLE_HPP
#define PETSC_VDP_JV_EXAMPLE_HPP
#include "../../src/cpp/janus_ode_common.hpp"
#include <petscts.h>
#include <iostream>
#include <vector>
//-------------------------------------------
// Application context
//-------------------------------------------
typedef struct {
  PetscReal   mu;  // Van der Pol parameter
  // These get updated each time IJacobian is called:
  PetscReal   a;   // "coefficient" from TS, typically 1/dt in implicit form
  PetscScalar x;   // current x(t)
  PetscScalar y;   // current y(t)
} AppCtx;


// First define your MonitorCtx struct
typedef struct {
  PetscInt  maxCount;

  PetscReal dt;    // desired output interval
  PetscInt   maxSteps;  
  PetscInt   count;     
  PetscReal *t;
  PetscReal *x;
  PetscReal *y;
  PetscReal  nextOutputTime; // only if you need it for fixed-interval storage
} MonitorCtx;


//-------------------------------------------
// IFunction:  F(t,U,Udot) = Udot - f(U) = 0
//-------------------------------------------
// U = (x, y), so
//   x'(t) = y
//   y'(t) = mu*(1 - x^2)*y - x
//
// F1 = x'(t) - y
// F2 = y'(t) - [mu(1-x^2)*y - x]
//
PetscErrorCode IFunction(TS ts, PetscReal t, Vec U, Vec Udot, Vec F, void *ctx)
{
  AppCtx               *user = (AppCtx*)ctx;
  const PetscScalar    *u, *udot;
  PetscScalar          *f;

  PetscFunctionBeginUser;
  VecGetArrayRead(U,    &u);
  VecGetArrayRead(Udot, &udot);
  VecGetArray(F, &f);

  // F1 = x'(t) - y
  f[0] = udot[0] - u[1];

  // F2 = y'(t) - [ mu(1 - x^2)*y - x ]
  //     = udot[1] - mu(1 - x^2)*y + x
  f[1] = udot[1] - (user->mu*(1.0 - u[0]*u[0])*u[1] - u[0]);

  VecRestoreArrayRead(Udot, &udot);
  VecRestoreArrayRead(U,    &u);
  VecRestoreArray(F, &f);
  PetscFunctionReturn(0);
}


//-------------------------------------------
// The matrix-free Jv: 
//   J = aI - dF/dU  = aI - d/dU [ Udot - f(U) ]
//
// We store 'a', 'x', 'y', 'mu' in the user context.
// Then compute  Jv = [ a         -1        ] [ v0 ]
//                [ (2mu*x*y+1)  a + mu(...) ] [ v1 ]
//
//-------------------------------------------
PetscErrorCode VdpJvFunction(Mat A, Vec v, Vec Jv)
{
  AppCtx               *user;
  PetscScalar          *jv;
  const PetscScalar    *vv;
  PetscReal            a, mu, x, y;

  PetscFunctionBeginUser;
  // 1) Access the shell context
  MatShellGetContext(A, (void**)&user);
  a  = user->a;
  mu = user->mu;
  x  = user->x;
  y  = user->y;

  // 2) Read input vector v
  VecGetArrayRead(v, &vv);
  VecGetArray(Jv, &jv);

  // 3) Jv = (aI - df/dU)*v
  //    df1/dx = 0 => J(0,0)= a
  //    df1/dy = 1 => J(0,1)= -1
  //    df2/dx = d/dx [ mu(1-x^2)*y - x ] = 2 mu x y + 1 => with minus => -(2 mu x y + 1)
  //    df2/dy = mu(1 - x^2) => with minus => -mu(1 - x^2)
  //
  //    So J = [ a,                  -1      ]
  //            [ (2mu*x*y + 1),  a + mu(...) ]
  //
  jv[0] = a*vv[0] - 1.0 * vv[1];
  jv[1] = (2.0*mu*x*y + 1.0)*vv[0] 
            + (a + mu*(1.0 - x*x))*vv[1];

  // 4) Restore
  VecRestoreArrayRead(v, &vv);
  VecRestoreArray(Jv, &jv);
  PetscFunctionReturn(0);
}


//-------------------------------------------
// IJacobianShell: 
//   We do NOT assemble the operator A (it's a shell).
//   But we MUST fill the preconditioner matrix P.
//
//   A:   the shell matrix for Jv
//   P:   a real 2x2 matrix we factor for preconditioning
//-------------------------------------------
PetscErrorCode IJacobianShell(TS ts, PetscReal t, Vec U, Vec Udot,
                              PetscReal a, Mat A, Mat P, void *ctx)
{
  AppCtx            *user = (AppCtx*)ctx;
  const PetscScalar *u;
  PetscScalar       vals[4];
  PetscInt          row[2] = {0, 1}, col[2] = {0, 1};

  PetscFunctionBeginUser;
  // 1) Extract current x,y
  VecGetArrayRead(U, &u);
  user->x = u[0];
  user->y = u[1];
  VecRestoreArrayRead(U, &u);

  // 2) Store the 'a' in user context
  user->a = a;

  // 3) Fill the preconditioner matrix P with the same 2x2 values we use in Jv
  //    J = aI - df/dU. 
  //    So explicitly:
  //      J(0,0) = a, 
  //      J(0,1) = -1,
  //      J(1,0) = 2 mu x y + 1,
  //      J(1,1) = a + mu(1 - x^2).
  vals[0] = a;                          // (row=0,col=0)
  vals[1] = -1.0;                       // (row=0,col=1)
  vals[2] = 2.0*user->mu*user->x*user->y + 1.0; // (row=1,col=0)
  vals[3] = a + user->mu*(1.0 - user->x*user->x); // (row=1,col=1)

  MatSetValues(P, 2, row, 2, col, vals, INSERT_VALUES);
  MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(P,   MAT_FINAL_ASSEMBLY);

  // Also assemble A (shell) for safety, even though there's no real data
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A,   MAT_FINAL_ASSEMBLY);

  PetscFunctionReturn(0);
}


PetscErrorCode CustomMonitor(TS ts, PetscInt step, PetscReal time, Vec U, void *ctx)
{
  MonitorCtx          *mctx = (MonitorCtx*)ctx;
  const PetscScalar   *sol;

  PetscFunctionBeginUser;

  // 1) Check if we are repeating the same time
  if (mctx->count > 0) {
    PetscReal lastTime = mctx->t[mctx->count - 1];
    if (PetscAbsReal(time - lastTime) < 1e-14) {
      // This time is effectively the same as the last stored one
      // => skip it
      PetscFunctionReturn(0);
    }
  }

  // 2) Actually store
  PetscCall(VecGetArrayRead(U,&sol));
  if (mctx->count < mctx->maxSteps) {
    mctx->t[mctx->count] = time;
    mctx->x[mctx->count] = PetscRealPart(sol[0]);
    mctx->y[mctx->count] = PetscRealPart(sol[1]);
    mctx->count++;
  }
  PetscCall(VecRestoreArrayRead(U,&sol));

  PetscFunctionReturn(0);
}


int calcTraj(int argc, 
             char** argv,
            std::vector<double> &tvals,
            std::vector<double> &xvals,
            std::vector<double> &yvals)
{
  PetscCall(PetscInitialize(&argc,&argv,NULL,NULL));
  {
    TS         ts;
    Vec        U;
    Mat        A,P;  
    AppCtx     user;
    PetscScalar u0[2] = {2.0, 0.0};

    // 1) Set up a small context to store solutions
    MonitorCtx mon;
    mon.maxSteps = 10000;
    mon.count    = 0;
    PetscCalloc1(mon.maxSteps,&mon.t);
    PetscCalloc1(mon.maxSteps,&mon.x);
    PetscCalloc1(mon.maxSteps,&mon.y);

    // 2) Create the TS
    PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
    PetscCall(TSSetType(ts,TSBDF));  // or TSROS, TSARKIMEX, etc.

    // 3) Create solution vector (2 unknowns)
    PetscCall(VecCreate(PETSC_COMM_WORLD,&U));
    PetscCall(VecSetSizes(U,PETSC_DECIDE,2));
    PetscCall(VecSetFromOptions(U));
    // Set initial condition
    {
      PetscInt idx[2]={0,1};
      PetscCall(VecSetValues(U,2,idx,u0,INSERT_VALUES));
    }
    PetscCall(VecAssemblyBegin(U));
    PetscCall(VecAssemblyEnd(U));

    // 4) Fill user context
    user.mu = 1.0;  // or 1000 for stiffer
    user.a  = 0.0;
    user.x  = u0[0];
    user.y  = u0[1];

    // 5) Attach IFunction
    PetscCall(TSSetIFunction(ts,NULL,IFunction,&user));

    // 6) Create the shell matrix A and the dense P
    PetscCall(MatCreateShell(PETSC_COMM_WORLD,2,2,2,2,(void*)&user,&A));
    PetscCall(MatShellSetOperation(A,MATOP_MULT,(void(*)(void))VdpJvFunction));
    PetscCall(MatCreateDense(PETSC_COMM_WORLD,2,2,2,2,NULL,&P));

    // 7) Set IJacobian
    PetscCall(TSSetIJacobian(ts,A,P,IJacobianShell,&user));

    // 8) Configure time-step adaptivity if you want
    {
      TSAdapt adapt;
      PetscCall(TSGetAdapt(ts,&adapt));
      PetscCall(TSAdaptSetType(adapt,TSADAPTBASIC));
      PetscReal hmin=1e-6, hmax=0.1;
      PetscCall(TSAdaptSetStepLimits(adapt,hmin,hmax));
    }

    // 9) We will manually break [0,1.0] into chunks of dt=0.01.
    PetscReal time0 = 0.0, Tfinal=1.0, dt=0.01;
    PetscInt  nSteps = (PetscInt)((Tfinal - time0)/dt);

    // Provide a small initial step so it doesn't jump from e.g. 0.0->0.01 in one step
    PetscCall(TSSetTimeStep(ts,dt/10.0));

    // Optionally allow unlimited SNES/step rejections
    PetscCall(TSSetMaxSNESFailures(ts,-1));
    PetscCall(TSSetMaxStepRejections(ts,-1));

    // 10) If we want the custom monitor called at each sub-step:
    PetscCall(TSMonitorSet(ts,CustomMonitor,&mon,NULL));

    // 11) Let user set run-time options
    PetscCall(TSSetFromOptions(ts));

    // 12) Repeatedly call TSSolve in sub-intervals
    PetscCall(TSSetTime(ts,time0));
    tvals.resize(nSteps+1);
    xvals.resize(nSteps+1);
    yvals.resize(nSteps+1);

    for (PetscInt i=1; i<=nSteps; i++) {
      PetscReal targetTime = time0 + i*dt;

      // Force the solver to stop at time = targetTime
      PetscCall(TSSetMaxTime(ts,targetTime));

      // Integrate from the current time up to 'targetTime'
      PetscCall(TSSolve(ts,U));

      // After TSSolve returns, the solution in U is at t=targetTime
      // If you ONLY want data at the final chunk time (not each sub-step),
      // you can store it here yourself. For example:
      const PetscScalar *sol;
      PetscCall(VecGetArrayRead(U,&sol));
      std::cout << "[CHUNK " << i << "] t=" << targetTime 
                << " x=" << PetscRealPart(sol[0]) 
                << " y=" << PetscRealPart(sol[1]) << std::endl;
      tvals[i-1] = targetTime;
      xvals[i-1] = PetscRealPart(sol[0]);
      yvals[i-1] = PetscRealPart(sol[1]);
          
      PetscCall(VecRestoreArrayRead(U,&sol));
    

    }

    // 13) Print final solution (should be at t=1.0)
    {
      const PetscScalar *sol;
      PetscCall(VecGetArrayRead(U,&sol));
      PetscPrintf(PETSC_COMM_WORLD,
                  "Final solution at t=%g: x=%g, y=%g\n",
                  (double)1.0,(double)sol[0],(double)sol[1]);
      PetscCall(VecRestoreArrayRead(U,&sol));
    }

    // 14) Clean up
    PetscCall(MatDestroy(&A));
    PetscCall(MatDestroy(&P));
    PetscCall(VecDestroy(&U));
    PetscCall(TSDestroy(&ts));

    // 15) Dump the data we collected in the monitor
    //     (every internal step across each chunk).
    //     mon.count is how many sub-steps total occurred.
    std::cout << "\nMonitor stored " << mon.count << " steps in total.\n";

    PetscCall(PetscFree(mon.t));
    PetscCall(PetscFree(mon.x));
    PetscCall(PetscFree(mon.y));
  }
  PetscCall(PetscFinalize());


}

#endif