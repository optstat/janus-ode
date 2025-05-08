#include <petscts.h>
#include <petscsnes.h>

/* Application context for the shooting problem */
typedef struct {
  PetscInt    nseg;         // number of shooting segments
  PetscInt    state_dim;    // dimension of state (ODE system)
  PetscReal  *time_pts;     // array of time points (segment boundaries), length nseg+1
  PetscScalar *initial_vals;  // known initial boundary values (length = state_dim)
  PetscBool  *initial_fixed;  // flags for which components have fixed initial BC
  PetscScalar *final_vals;    // known final boundary values (length = state_dim)
  PetscBool  *final_fixed;    // flags for which components have fixed final BC
  TS         *ts;           // array of TS solvers for each segment (length = nseg)
  Vec        *Y;           // array of state vectors for each segment (length = nseg)
} ShootingCtx;

/* RHS function for the ODE system: y' = f(t, y).
   In our case: 
      y1' = y2
      y2' = -exp(y1)    (from y'' + e^y = 0 -> y2' = -e^{y1})
*/
static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec Y, Vec Ydot, void *ctx) {
  const PetscScalar *y;
  PetscScalar       *f;
  PetscErrorCode     ierr;
  
  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(Y, &y); CHKERRQ(ierr);
  ierr = VecGetArray(Ydot, &f); CHKERRQ(ierr);
  
  f[0] = y[1];                     /* y1' = y2 */
  f[1] = -PetscExpScalar(y[0]);    /* y2' = -exp(y1) */
  
  ierr = VecRestoreArrayRead(Y, &y); CHKERRQ(ierr);
  ierr = VecRestoreArray(Ydot, &f); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* SNES function to compute the nonlinear residual F(X) for the multi-shooting system.
   X contains the current guess for:
     - any free initial components (those not fixed by initial BC),
     - the state at each interior segment boundary.
   The residual F(X) enforces continuity at each interior boundary and satisfaction of final BC. */
static PetscErrorCode FormFunction(SNES snes, Vec X, Vec F, void *user) {
  ShootingCtx        *ctx = (ShootingCtx*)user;
  PetscInt            m = ctx->state_dim;
  PetscInt            nseg = ctx->nseg;
  const PetscScalar  *x;
  PetscScalar        *f;
  PetscErrorCode      ierr;
  
  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X, &x); CHKERRQ(ierr);
  ierr = VecGetArray(F, &f); CHKERRQ(ierr);
  
  // Set initial state for the first segment (segment 0)
  PetscInt idx = 0;  // index in X array
  // Load the initial state vector for segment 0
  PetscScalar *Y0;
  ierr = VecGetArray(ctx->Y[0], &Y0); CHKERRQ(ierr);
  for (PetscInt j = 0; j < m; ++j) {
    if (ctx->initial_fixed[j]) {
      // Component j has a fixed initial boundary value
      Y0[j] = ctx->initial_vals[j];
    } else {
      // Component j initial value is free (an unknown in X)
      Y0[j] = x[idx++];
    }
  }
  ierr = VecRestoreArray(ctx->Y[0], &Y0); CHKERRQ(ierr);
  
  // Integrate each segment and compute continuity residuals
  PetscInt f_idx = 0;  // index in residual F
  for (PetscInt i = 0; i < nseg; ++i) {
    // Integrate segment i from time_pts[i] to time_pts[i+1]
    ierr = TSSetTime(ctx->ts[i], ctx->time_pts[i]); CHKERRQ(ierr);
    ierr = TSSetMaxTime(ctx->ts[i], ctx->time_pts[i+1]); CHKERRQ(ierr);
    ierr = TSSolve(ctx->ts[i], ctx->Y[i]); CHKERRQ(ierr);
    // On return, ctx->Y[i] holds the state at the end of segment i (time_pts[i+1])
    
    const PetscScalar *Y_end;
    ierr = VecGetArrayRead(ctx->Y[i], &Y_end); CHKERRQ(ierr);
    if (i < nseg - 1) {
      // Continuity constraint between segment i and i+1
      // The start of segment (i+1) is an unknown in X; enforce it equals the end of segment i.
      // That gives m residual equations (one for each state component).
      for (PetscInt j = 0; j < m; ++j) {
        // Unknown state for segment (i+1) start:
        PetscScalar Y_next = x[idx++];      // this is y_j at the beginning of segment i+1 (from X)
        f[f_idx++] = Y_end[j] - Y_next;     // residual: end of seg i minus start of seg (i+1) = 0
      }
      // Set the initial state for segment (i+1) from the guess X, for the next integration
      PetscScalar *Y_next_vec;
      ierr = VecGetArray(ctx->Y[i+1], &Y_next_vec); CHKERRQ(ierr);
      for (PetscInt j = 0; j < m; ++j) {
        Y_next_vec[j] = (x[idx - m + j]);  // note: idx was incremented by m, so idx-m gives start of this block
      }
      ierr = VecRestoreArray(ctx->Y[i+1], &Y_next_vec); CHKERRQ(ierr);
    } else {
      // Last segment: enforce final boundary conditions
      for (PetscInt j = 0; j < m; ++j) {
        if (ctx->final_fixed[j]) {
          // For each component with a specified final value, add residual: (end_value - target) = 0
          f[f_idx++] = Y_end[j] - ctx->final_vals[j];
        }
        // If final value is not fixed for component j, we do not add an equation (free end).
      }
    }
    ierr = VecRestoreArrayRead(ctx->Y[i], &Y_end); CHKERRQ(ierr);
  }
  
  ierr = VecRestoreArrayRead(X, &x); CHKERRQ(ierr);
  ierr = VecRestoreArray(F, &f); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscErrorCode ierr;
  ShootingCtx    ctx;
  SNES           snes;
  Vec            X;
  Mat            J;
  
  /* Initialize PETSc and context */
  ierr = PetscInitialize(&argc, &argv, NULL, "Solves a nonlinear BVP with multi-shooting.\n"); if (ierr) return ierr;
  ctx.state_dim = 2;            // two equations (y1 and y2)
  ctx.nseg = 3;                 // number of shooting segments (can be adjusted)
  
  // Allocate arrays in the context
  ierr = PetscMalloc1(ctx.nseg+1, &ctx.time_pts); CHKERRQ(ierr);
  ierr = PetscMalloc1(ctx.state_dim, &ctx.initial_vals); CHKERRQ(ierr);
  ierr = PetscMalloc1(ctx.state_dim, &ctx.initial_fixed); CHKERRQ(ierr);
  ierr = PetscMalloc1(ctx.state_dim, &ctx.final_vals); CHKERRQ(ierr);
  ierr = PetscMalloc1(ctx.state_dim, &ctx.final_fixed); CHKERRQ(ierr);
  ierr = PetscMalloc1(ctx.nseg, &ctx.ts); CHKERRQ(ierr);
  ierr = PetscMalloc1(ctx.nseg, &ctx.Y); CHKERRQ(ierr);
  
  // Define the time partition (uniform segments from 0 to 1)
  PetscReal t0 = 0.0, t1 = 1.0;
  for (PetscInt i = 0; i <= ctx.nseg; ++i) {
    ctx.time_pts[i] = t0 + (t1 - t0) * ((PetscReal)i / ctx.nseg);
  }
  
  // Set boundary conditions:
  // Initial BC: y1(0) = 0 (fixed), y2(0) is free (to be determined)
  ctx.initial_fixed[0] = PETSC_TRUE;   ctx.initial_vals[0] = 0.0;
  ctx.initial_fixed[1] = PETSC_FALSE;  ctx.initial_vals[1] = 0.0;   // (value here is ignored since not fixed)
  // Final BC: y1(1) = 0 (fixed), y2(1) free
  ctx.final_fixed[0]   = PETSC_TRUE;   ctx.final_vals[0]   = 0.0;
  ctx.final_fixed[1]   = PETSC_FALSE;  ctx.final_vals[1]   = 0.0;
  
  // Create TS for each segment and a solution vector for each segment
  for (PetscInt i = 0; i < ctx.nseg; ++i) {
    ierr = TSCreate(PETSC_COMM_WORLD, &ctx.ts[i]); CHKERRQ(ierr);
    ierr = TSSetProblemType(ctx.ts[i], TS_NONLINEAR); CHKERRQ(ierr);
    ierr = TSSetRHSFunction(ctx.ts[i], NULL, RHSFunction, NULL); CHKERRQ(ierr);
    // Choose a time integration method (explicit Runge-Kutta in this case)
    ierr = TSSetType(ctx.ts[i], TSRK); CHKERRQ(ierr);
    // Set an initial time step (for fixed stepping) and ensure we hit the end time exactly
    PetscReal dt = (ctx.time_pts[i+1] - ctx.time_pts[i]) / 100.0;  // 100 steps per segment (arbitrary choice)
    ierr = TSSetTimeStep(ctx.ts[i], dt); CHKERRQ(ierr);
    ierr = TSSetMaxSteps(ctx.ts[i], 10000); CHKERRQ(ierr);  // safety: allow up to 10000 steps
    ierr = TSSetExactFinalTime(ctx.ts[i], TS_EXACTFINALTIME_INTERPOLATE); CHKERRQ(ierr);
    // Allow runtime options to override defaults (e.g., method, step size)
    ierr = TSSetFromOptions(ctx.ts[i]); CHKERRQ(ierr);
    // Create solution vector for this segment (size = state_dim)
    ierr = VecCreate(PETSC_COMM_WORLD, &ctx.Y[i]); CHKERRQ(ierr);
    ierr = VecSetSizes(ctx.Y[i], PETSC_DECIDE, ctx.state_dim); CHKERRQ(ierr);
    ierr = VecSetFromOptions(ctx.Y[i]); CHKERRQ(ierr);
    ierr = VecSetUp(ctx.Y[i]); CHKERRQ(ierr);
  }
  
  // Create SNES solver
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes); CHKERRQ(ierr);
  // Create vector for unknown shooting variables
  // Number of unknowns = free initial components + (nseg-1) * state_dim (interior points)
  PetscInt n_unknown = 0;
  for (PetscInt j = 0; j < ctx.state_dim; ++j) {
    if (!ctx.initial_fixed[j]) n_unknown++;  // each free initial component is an unknown
  }
  n_unknown += (ctx.nseg - 1) * ctx.state_dim;  // each interior boundary contributes full state dimension unknowns
  ierr = VecCreate(PETSC_COMM_WORLD, &X); CHKERRQ(ierr);
  ierr = VecSetSizes(X, PETSC_DECIDE, n_unknown); CHKERRQ(ierr);
  ierr = VecSetFromOptions(X); CHKERRQ(ierr);
  ierr = VecSetUp(X); CHKERRQ(ierr);
  
  // Set SNES function (residual) and context
  ierr = SNESSetFunction(snes, NULL, FormFunction, &ctx); CHKERRQ(ierr);
  
  // Set SNES Jacobian to use finite differences (no explicit Jacobian provided)
  // We create a matrix for SNES to use. Here we use a dense matrix for simplicity.
  ierr = MatCreate(PETSC_COMM_WORLD, &J); CHKERRQ(ierr);
  ierr = MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, n_unknown, n_unknown); CHKERRQ(ierr);
  ierr = MatSetFromOptions(J); CHKERRQ(ierr);
  ierr = MatSetUp(J); CHKERRQ(ierr);
  // Use PETSc's default finite-difference computation for the Jacobian
  ierr = SNESSetJacobian(snes, J, J, SNESComputeJacobianDefault, NULL); CHKERRQ(ierr);
  
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);  // allow command-line options for SNES
  
  // Set an initial guess for the unknown vector X
  PetscScalar *x_guess;
  ierr = VecGetArray(X, &x_guess); CHKERRQ(ierr);
  PetscInt offset = 0;
  // Guess for free initial components: if final value for that component is known, use it; otherwise 0.
  for (PetscInt j = 0; j < ctx.state_dim; ++j) {
    if (!ctx.initial_fixed[j]) {
      if (ctx.final_fixed[j]) 
        x_guess[offset] = ctx.final_vals[j];
      else 
        x_guess[offset] = 0.0;
      offset++;
    }
  }
  // Guess for interior segment states (for each interior boundary point)
  for (PetscInt i = 1; i < ctx.nseg; ++i) {
    PetscReal alpha = (ctx.time_pts[i] - ctx.time_pts[0]) / (ctx.time_pts[ctx.nseg] - ctx.time_pts[0]);
    for (PetscInt j = 0; j < ctx.state_dim; ++j) {
      if (ctx.initial_fixed[j] && ctx.final_fixed[j]) {
        // If the component has specified values at both ends, linearly interpolate
        PetscScalar y0 = ctx.initial_vals[j];
        PetscScalar y1 = ctx.final_vals[j];
        x_guess[offset] = y0 + alpha * (y1 - y0);
      } else if (ctx.initial_fixed[j] && !ctx.final_fixed[j]) {
        // Only initial BC given for this component: use that value as guess throughout
        x_guess[offset] = ctx.initial_vals[j];
      } else if (!ctx.initial_fixed[j] && ctx.final_fixed[j]) {
        // Only final BC given: use final value as guess
        x_guess[offset] = ctx.final_vals[j];
      } else {
        // Free at both ends: guess 0
        x_guess[offset] = 0.0;
      }
      offset++;
    }
  }
  ierr = VecRestoreArray(X, &x_guess); CHKERRQ(ierr);
  
  // Solve the nonlinear system
  ierr = SNESSolve(snes, NULL, X); CHKERRQ(ierr);
  
  // Get and print the solution for the shooting variables
  const PetscScalar *x_sol;
  ierr = VecGetArrayRead(X, &x_sol); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Solution shooting variables:\n");
  offset = 0;
  // Print free initial conditions found
  for (PetscInt j = 0; j < ctx.state_dim; ++j) {
    if (!ctx.initial_fixed[j]) {
      PetscPrintf(PETSC_COMM_WORLD, "  Initial state component %d = %g\n", j, (double)x_sol[offset]);
      offset++;
    }
  }
  // Print interior point states found
  for (PetscInt i = 1; i < ctx.nseg; ++i) {
    PetscPrintf(PETSC_COMM_WORLD, "  State at x = %.3f :", (double)ctx.time_pts[i]);
    for (PetscInt j = 0; j < ctx.state_dim; ++j) {
      PetscPrintf(PETSC_COMM_WORLD, "  %g", (double)x_sol[offset + j]);
    }
    PetscPrintf(PETSC_COMM_WORLD, "\n");
    offset += ctx.state_dim;
  }
  ierr = VecRestoreArrayRead(X, &x_sol); CHKERRQ(ierr);
  
  // (Optional) Verify the solution by integrating all segments with the found values
  // and checking the final boundary.
  // Set initial state for segment 0 using solution values
  PetscInt idx = 0;
  //Expand the dimension of Y0_sol to ctx.state_dim
  PetscScalar *Y0_sol;

  ierr = VecGetArray(ctx.Y[0], &Y0_sol); CHKERRQ(ierr);
  for (PetscInt j = 0; j < ctx.state_dim; ++j) {
    Y0_sol[j] = ctx.initial_fixed[j] ? ctx.initial_vals[j] : x_sol[idx++];
  }
  ierr = VecRestoreArray(ctx.Y[0], &Y0_sol); CHKERRQ(ierr);
  // Integrate segments sequentially, copying end of one to start of next
  for (PetscInt i = 0; i < ctx.nseg; ++i) {
    ierr = TSSetTime(ctx.ts[i], ctx.time_pts[i]); CHKERRQ(ierr);
    ierr = TSSetMaxTime(ctx.ts[i], ctx.time_pts[i+1]); CHKERRQ(ierr);
    ierr = TSSolve(ctx.ts[i], ctx.Y[i]); CHKERRQ(ierr);
    if (i < ctx.nseg - 1) {
      ierr = VecCopy(ctx.Y[i], ctx.Y[i+1]); CHKERRQ(ierr);  // continuity (should be exact now)
    }
  }
  // Now ctx.Y[ctx.nseg-1] holds the solution at x=1
  const PetscScalar *Y_end;
  ierr = VecGetArrayRead(ctx.Y[ctx.nseg-1], &Y_end); CHKERRQ(ierr);
  PetscPrintf(PETSC_COMM_WORLD, "Final y1(1) = %g (target = %g)\n", (double)Y_end[0], (double)ctx.final_vals[0]);
  ierr = VecRestoreArrayRead(ctx.Y[ctx.nseg-1], &Y_end); CHKERRQ(ierr);
  
  // Clean up PETSc objects
  ierr = VecDestroy(&X); CHKERRQ(ierr);
  ierr = SNESDestroy(&snes); CHKERRQ(ierr);
  ierr = MatDestroy(&J); CHKERRQ(ierr);
  for (PetscInt i = 0; i < ctx.nseg; ++i) {
    ierr = TSDestroy(&ctx.ts[i]); CHKERRQ(ierr);
    ierr = VecDestroy(&ctx.Y[i]); CHKERRQ(ierr);
  }
  ierr = PetscFree(ctx.time_pts); CHKERRQ(ierr);
  ierr = PetscFree(ctx.initial_vals); CHKERRQ(ierr);
  ierr = PetscFree(ctx.initial_fixed); CHKERRQ(ierr);
  ierr = PetscFree(ctx.final_vals); CHKERRQ(ierr);
  ierr = PetscFree(ctx.final_fixed); CHKERRQ(ierr);
  ierr = PetscFree(ctx.ts); CHKERRQ(ierr);
  ierr = PetscFree(ctx.Y); CHKERRQ(ierr);
  
  ierr = PetscFinalize(); CHKERRQ(ierr);
  return 0;
}
