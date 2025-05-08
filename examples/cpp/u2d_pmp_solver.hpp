#ifndef u2d_pmp_solver_hpp
#define u2d_pmp_solver_hpp
#include <petscts.h>
#include <petscsnes.h>

#include <iostream>
/**
 * Universal PMP solver for 2D problem using PETSC
 * This solves a set of augmented ODES of the form [\dot{l} \dot{x}]
 * For the universal 2D minimum time Hamiltonian
 * H = l1*u1 + l2*u2 + l3 + W/2*(u1^2 + u2^2) + 
 *     \sum alpha*(x1-x1_i)^2*exp(-(t-t_i)^2/(2*sigma^2))
 *     +W/2*(u1^2 + u2^2)
 * 
 */

 namespace u2d {
    double sigma = 0.01;
    double alpha = 1.0;
    double gf = 1.0;
    double W = 0.01;

    /* 
       Data structure for u2d problem.
       We have six states: x1, x2, x3, l1, l2, l3
       We have six equations with 
    */
   typedef struct {
      PetscScalar l10; //This is unknown
      PetscScalar l20; //This is unknown
      PetscScalar l30; //This is always -1
      PetscScalar x10; //This will be provided by data
      PetscScalar x20; //This will be provided by data
      PetscScalar x30; //This is always 0
      PetscScalar x1f; //This will be provided by data 
      PetscScalar x2f; //This will be provided by data
      //The third costate is always -1
      PetscScalar x3f; //This is the same as terminal time and is not known
    } U2DCtx;


    //Using PETSC define dymamics 
    //-------------------------------------------
    // IFunction:  F(t,U,Udot) = Udot - f(U) = 0
    //-------------------------------------------
    // U = (x, y), so
    //   l1'(t) = \sum_i alpha*2*(x1-x1_i)*(x2-x2_i)^2*exp(-(t-t_i)^2/(2*sigma^2))
    //   l2'(t) = \sum_i alpha*(x1-x1_i)^2*2*(x2-x2_i)*exp(-(t-t_i)^2/(2*sigma^2))
    //   l3'(t) = 0
    //   x1'(t) = -l1/W
    //   x2'(t) = -l2/W
    //   x3'(t) = 1
    //
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
    
    
    
    /* ----------------------------------------------------------------------
       U2DSolve - Given an initial guess for y(0)=eta, integrate 
       the Van der Pol system from t=0 to t=1 with x(0)=0, y(0)={l10, l20, l30}.
       Then return final x(6) in *finalX.
       This is the outer loop of the inner shooting method.

    
       We'll use a basic time integrator (TSRK or TSEULER, etc.) 
       so we can see how the solution evolves. 
     ---------------------------------------------------------------------- */
    PetscErrorCode U2DSolve(PetscScalar l10, 
                            PetscScalar *l1f,
                            PetscScalar *l2f,
                            PetscScalar *l3f,
                            PetscScalar *x1f,
                            PetscScalar *x2f,
                            PetscScalar *x3f, 
                            U2DCtx *user)
    {
      PetscErrorCode ierr;
      TS             ts;
      Vec            U;         /* solution vector: [l1f(t), l2f(t), l3f(t), x1f(t), x2f(t), x3f(t)] */
      PetscScalar    *uarray;
      PetscReal      t0=0.0, tf=1.0;  /* integration interval */
      
      PetscFunctionBeginUser;
      /* Create the TS object */
      ierr = TSCreate(PETSC_COMM_WORLD, &ts);CHKERRQ(ierr);
      /* We'll solve a general nonlinear ODE system */
      ierr = TSSetProblemType(ts, TS_NONLINEAR);CHKERRQ(ierr);
    
      /* 
         Choose a time-stepping method. 
         Here we set to BDF 
      */
      ierr = TSSetType(ts, TSBDF);CHKERRQ(ierr);
    
      /* Create the solution vector U */
      ierr = VecCreate(PETSC_COMM_WORLD, &U);CHKERRQ(ierr);
      ierr = VecSetSizes(U, PETSC_DECIDE, 2);CHKERRQ(ierr);
      ierr = VecSetFromOptions(U);CHKERRQ(ierr);
    
      /* Set initial conditions: */
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
    

 }

#endif