#include <petsc.h>

/* -------------------------------------------------------------------
   Data structures
   ------------------------------------------------------------------- */
typedef struct
{
    PetscReal p;     /* ODE parameter ( x'(t) = p ) */
    PetscReal alpha; /* The unknown initial condition at t=0.5 for sub-interval 2 */
} AppCtx;

/* -------------------------------------------------------------------
   ODE for sub-intervals: x'(t) = p
   We'll define F(t, x, xdot) = xdot - p = 0.
   ------------------------------------------------------------------- */

/* An explicit RHS function:  dx/dt = p */
static PetscErrorCode RHSFunction_SimpleODE(TS ts, PetscReal t, Vec X, Vec F, void *ctx)
{
    const AppCtx *user = (AppCtx *)ctx;
    const PetscReal p = user->p;
    PetscScalar *farr;

    PetscFunctionBeginUser;
    PetscCall(VecGetArray(F, &farr));
    farr[0] = p; /* dx/dt = p */
    PetscCall(VecRestoreArray(F, &farr));
    PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------
   Helper routine: Solve the ODE x'(t)=p from time t0 to tf,
   with initial condition x(t0)=xIC. Returns solution at tf.
   ------------------------------------------------------------------- */
static PetscErrorCode SolveSubInterval(PetscReal t0, PetscReal tf,
                                       PetscReal xIC, const AppCtx *user,
                                       PetscReal *xOut)
{
    TS ts;
    Vec X;
    PetscReal dt;
    PetscInt steps;

    PetscFunctionBeginUser;
    /* Create a single DoF vector to hold the state */
    PetscCall(VecCreate(PETSC_COMM_SELF, &X));
    PetscCall(VecSetSizes(X, 1, 1));
    PetscCall(VecSetFromOptions(X));

    /* Set initial condition x(t0) = xIC */
    PetscCall(VecSetValue(X, 0, xIC, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(X));
    PetscCall(VecAssemblyEnd(X));

    /* Create TS */
    PetscCall(TSCreate(PETSC_COMM_SELF, &ts));
    PetscCall(TSSetProblemType(ts, TS_NONLINEAR)); /* ODE is nonlinear or linear */
    PetscCall(TSSetType(ts, TSEULER));             /* For simplicity, use Euler */
    PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction_SimpleODE, (void *)user));

    /* Set time interval */
    PetscCall(TSSetTime(ts, t0));
    PetscCall(TSSetMaxTime(ts, tf));
    PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));

    /* Choose a small time step or let PETSc adapt */
    dt = (tf - t0) / 10.0;
    PetscCall(TSSetTimeStep(ts, dt));

    /* Solve forward in sub-interval */
    PetscCall(TSSolve(ts, X));
    PetscCall(TSGetStepNumber(ts, &steps));

    /* Extract final solution x(tf) */
    {
        const PetscScalar *xarr;
        PetscCall(VecGetArrayRead(X, &xarr));
        *xOut = xarr[0];
        PetscCall(VecRestoreArrayRead(X, &xarr));
    }

    /* Cleanup */
    PetscCall(TSDestroy(&ts));
    PetscCall(VecDestroy(&X));
    PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------
   SNES residual: G(X) = [ G1; G2 ], where
   X = [p, alpha].
   G1 = x1(0.5; p) - alpha    (continuity at t=0.5)
   G2 = x2(1;   p, alpha) - 2 (final boundary condition x(1)=2)

   We solve sub-interval 1 from t=0 to 0.5 with x(0)=0.
   Then sub-interval 2 from t=0.5 to 1 with x(0.5)=alpha.
   ------------------------------------------------------------------- */
static PetscErrorCode SNESResidual_MultiShooting(SNES snes, Vec X, Vec F, void *ctx)
{
    AppCtx *user = (AppCtx *)ctx;
    const PetscScalar *xarr;
    PetscScalar *farr;
    PetscReal xMid, xEnd;

    PetscFunctionBeginUser;
    /* Unpack unknowns from the SNES vector X */
    PetscCall(VecGetArrayRead(X, &xarr));
    user->p = xarr[0];
    user->alpha = xarr[1];
    PetscCall(VecRestoreArrayRead(X, &xarr));

    /* 1) Solve sub-interval 1: [0..0.5],  x(0)=0  */
    PetscCall(SolveSubInterval(0.0, 0.5, /*xIC=*/0.0, user, &xMid));

    /* 2) Solve sub-interval 2: [0.5..1], x(0.5)= alpha */
    PetscCall(SolveSubInterval(0.5, 1.0, /*xIC=*/user->alpha, user, &xEnd));

    /* Now form the residual */
    PetscCall(VecGetArray(F, &farr));
    farr[0] = xMid - user->alpha; /* continuity: x1(0.5) - alpha = 0 */
    farr[1] = xEnd - 2.0;         /* boundary:   x2(1) - 2 = 0      */
    PetscCall(VecRestoreArray(F, &farr));

    PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------
   Main
   ------------------------------------------------------------------- */
int main(int argc, char **argv)
{
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

    /* Create the SNES and solution vector X (size 2) as before. */
    SNES snes;
    Vec X, R, Xl, Xu;
    AppCtx user;

    SNESCreate(PETSC_COMM_WORLD, &snes);

    /* X is your 2D solution vector [p, alpha] */
    VecCreate(PETSC_COMM_WORLD, &X);
    VecSetSizes(X, PETSC_DECIDE, 2);
    VecSetFromOptions(X);

    /* Duplicate X for the residual. */
    VecDuplicate(X, &R);

    /* Provide SNES with your explicitly created residual vector R. */
    SNESSetFunction(snes, R, SNESResidual_MultiShooting, (void *)&user);

    /* Now set variable bounds for box constraints. */
    VecDuplicate(X, &Xl);
    VecDuplicate(X, &Xu);

    /* Fill Xl, Xu with your desired lower, upper bounds. */
    PetscInt idx[2] = {0, 1};
    PetscReal lb[2] = {0.0, 0.0}, ub[2] = {PETSC_INFINITY, PETSC_INFINITY};
    VecSetValues(Xl, 2, idx, lb, INSERT_VALUES);
    VecSetValues(Xu, 2, idx, ub, INSERT_VALUES);
    VecAssemblyBegin(Xl);
    VecAssemblyEnd(Xl);
    VecAssemblyBegin(Xu);
    VecAssemblyEnd(Xu);

    SNESVISetVariableBounds(snes, Xl, Xu);
    SNESSetType(snes, SNESVINEWTONRSLS);

    /* Optionally set an initial guess for X. */
    PetscScalar vals[2] = {1.0, 0.5};
    VecSetValues(X, 2, idx, vals, INSERT_VALUES);
    VecAssemblyBegin(X);
    VecAssemblyEnd(X);

    /* Finish setup and solve. */
    SNESSetFromOptions(snes);
    SNESSolve(snes, NULL, X);

    // Print the final solution
    PetscScalar *xarr;
    PetscCall(VecGetArray(X, &xarr));
    PetscPrintf(PETSC_COMM_WORLD, "Final solution: p=%g, alpha=%g\n", (double)xarr[0], (double)xarr[1]);

    /* Cleanup */
    VecDestroy(&R);
    VecDestroy(&Xl);
    VecDestroy(&Xu);
    VecDestroy(&X);
    SNESDestroy(&snes);
}