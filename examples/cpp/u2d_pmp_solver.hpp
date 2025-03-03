#ifndef u2d_pmp_solver_hpp
#define u2d_pmp_solver_hpp
#include <petscts.h>
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


 }

#endif