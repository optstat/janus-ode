#include <petscsnes.h>

/* We'll solve a simple 1D nonlinear function:
     F(x) = x^3 - 1 = 0,
   which has the real root x = 1.
*/

static PetscErrorCode MyResidualFunction(SNES snes, Vec x, Vec r, void *ctx)
{
  const PetscScalar *xx;
  PetscScalar       *rr;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(x, &xx); CHKERRQ(ierr);
  ierr = VecGetArray(r, &rr); CHKERRQ(ierr);

  rr[0] = xx[0]*xx[0]*xx[0] - 1.0;

  ierr = VecRestoreArrayRead(x, &xx); CHKERRQ(ierr);
  ierr = VecRestoreArray(r, &rr);     CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MyJacobianFunction(SNES snes, Vec x, Mat A, Mat B, void *ctx)
{
  const PetscScalar *xx;
  PetscScalar       v;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(x, &xx); CHKERRQ(ierr);
  v = 3.0 * xx[0]*xx[0];  /* derivative of x^3 - 1 => 3 x^2 */

  /* It's a 1x1 matrix; set [0,0] entry */
  ierr = MatSetValue(B, 0, 0, v, INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(x, &xx); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);   CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);   CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  SNES           snes;
  Vec            x, r;
  Mat            J;
  PetscScalar    initialGuess = -5.0;
  SNESLineSearch linesearch;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL); if (ierr) return ierr;

  /* Create 1D Vec for solution x */
  ierr = VecCreate(PETSC_COMM_WORLD, &x); CHKERRQ(ierr);
  ierr = VecSetSizes(x, PETSC_DECIDE, 1); CHKERRQ(ierr);
  ierr = VecSetFromOptions(x); CHKERRQ(ierr);

  /* Set initial guess */
  ierr = VecSetValue(x, 0, initialGuess, INSERT_VALUES); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(x); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);   CHKERRQ(ierr);

  /* Optional residual Vec (same size as x) */
  ierr = VecDuplicate(x, &r); CHKERRQ(ierr);

  /* Create 1x1 matrix for the Jacobian */
  ierr = MatCreate(PETSC_COMM_WORLD, &J); CHKERRQ(ierr);
  ierr = MatSetSizes(J, 1, 1, 1, 1); CHKERRQ(ierr);
  ierr = MatSetFromOptions(J); CHKERRQ(ierr);
  ierr = MatSetUp(J); CHKERRQ(ierr);


  ierr = SNESCreate(PETSC_COMM_WORLD, &snes); CHKERRQ(ierr);
  ierr = SNESSetType(snes, SNESNGMRES); CHKERRQ(ierr);
  /* Create the top-level SNES and set it to composite mode */

  /* Set function and Jacobian on the top-level SNES */
  ierr = SNESSetFunction(snes, r, MyResidualFunction, NULL); CHKERRQ(ierr);
  
  /*
     Get the internal line search context from the SNES,
     specify we want to use an L2 line search (i.e., SNESLINESEARCHL2),
     and optionally set other parameters.
  */
  ierr = SNESGetLineSearch(snes, &linesearch); CHKERRQ(ierr);
  /* Set the type of the line search (can be SNESLINESEARCHBT, SNESLINESEARCHBASIC, etc.) */
  ierr = SNESLineSearchSetType(linesearch, SNESLINESEARCHL2); CHKERRQ(ierr);

  ierr = SNESSetJacobian(snes, J, J, MyJacobianFunction, NULL); CHKERRQ(ierr);
    /* Optional: set a damping factor or other parameters directly in code. */
  /*
     For example, if we want a damping factor = 0.5, we can do:
       SNESLineSearchSetDamping(linesearch, 0.5);
     or read from user-provided command line with SNESLineSearchSetFromOptions().
  */
  ierr = SNESLineSearchSetDamping(linesearch, 0.5); CHKERRQ(ierr);

  /*
     Honor any additional user-provided command-line options
     (like -snes_ngmres_m 30, -snes_ngmres_linesearch_type l2, etc.)
   */
  ierr = SNESLineSearchSetFromOptions(linesearch); CHKERRQ(ierr);


  /* Optionally set from command line, e.g. -snes_monitor */
  ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);


  /* Solve */
  ierr = SNESSolve(snes, NULL, x); CHKERRQ(ierr);

  /* Print final solution */
  {
    const PetscScalar *xarr;
    ierr = VecGetArrayRead(x, &xarr); CHKERRQ(ierr);
    PetscScalar sol = xarr[0];  // Read first (and only) entry
    ierr = VecRestoreArrayRead(x, &xarr); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD,"Composite SNES solution = %g\n",(double)sol);
  }

  /* Cleanup */
  ierr = SNESDestroy(&snes); CHKERRQ(ierr);
  ierr = VecDestroy(&x);     CHKERRQ(ierr);
  ierr = VecDestroy(&r);     CHKERRQ(ierr);
  ierr = MatDestroy(&J);     CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
