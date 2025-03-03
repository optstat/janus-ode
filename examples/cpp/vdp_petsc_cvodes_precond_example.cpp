/*****************************************************************************
 *
 *   Example: Van der Pol oscillator with CVODES + PETSc bridging
 *
 *   Compile (example command):
 *   mpicc vdp_cvodes_petsc.c -o vdp_cvodes_petsc \
 *       -I${PETSC_DIR}/include -I${PETSC_DIR}/${PETSC_ARCH}/include \
 *       -I${SUNDIALS_DIR}/include \
 *       -L${PETSC_DIR}/${PETSC_ARCH}/lib -lpetsc \
 *       -L${SUNDIALS_DIR}/lib -lsundials_cvodes -lsundials_nvecserial -lsundials_sunmatrixdense -lsundials_sunlinsolcustom \
 *       -lm
 *
 *   (Adjust paths and libraries as needed; actual linking may vary.)
 *
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* PETSc Headers */
#include <petscsys.h>
#include <petscksp.h>
#include <petscmat.h>

/* SUNDIALS / CVODES Headers */
#include <cvodes/cvodes.h>                 /* main CVODES header */
#include <nvector/nvector_serial.h>        /* serial N_Vector */
#include <sundials/sundials_types.h>       /* realtype, etc. */
#include <sundials/sundials_math.h>
#include <sunlinsol/sunlinsol_spgmr.h>     /* if needed, or custom. We do custom below. */

/* -- Toy system: Van der Pol -- */
typedef struct {
  PetscReal mu;
} *UserData;

/* --------------------------------------------------------------------------
 *   Van der Pol RHS: f(t, y) 
 *     y' = f(t,y)
 *   y[0] = y1, y[1] = y2
 * -------------------------------------------------------------------------- */
static int f(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data)
{
  UserData udata = (UserData)user_data;
  PetscReal mu = udata->mu;

  sunrealtype y1 = NV_Ith_S(y, 0);
  sunrealtype y2 = NV_Ith_S(y, 1);

  NV_Ith_S(ydot, 0) = y2;
  NV_Ith_S(ydot, 1) = mu*(1.0 - y1*y1)*y2 - y1;

  return 0;  /* success */
}

/* --------------------------------------------------------------------------
 *  We'll build a custom SUNLinearSolver that calls PETSc 
 *    - A 2x2 matrix (with FD coloring) 
 *    - KSP for the solve
 * -------------------------------------------------------------------------- */

/* Content structure for the PETSc-based SUNLinearSolver */
typedef struct {
  KSP         ksp;          /* the PETSc KSP */
  Mat         J;            /* Jacobian or preconditioning matrix (2x2) */
  MatFDColoring fdcoloring; /* FD coloring for J */
  Vec         xpetsc;       /* PETSc vector for solution */
  Vec         bpetsc;       /* PETSc vector for RHS */
  N_Vector    sundials_y;   /* a temporary reference to current y (for FD) */
  PetscBool   J_initialized;
  int last_flag; 
} *PETScLSContent;

/* Forward declarations of our required solver callbacks */
static int PETScLS_Setup(SUNLinearSolver S, SUNMatrix A);
static int PETScLS_Solve(SUNLinearSolver S, SUNMatrix A, N_Vector x, N_Vector b, sunrealtype tol);
static int PETScLS_Free(SUNLinearSolver S);
static SUNLinearSolver_Type MySolverGetType(SUNLinearSolver S)
{
  // If you consider this a "direct" solver, return SUNLINEARSOLVER_DIRECT
  // If it’s an iterative solver, return SUNLINEARSOLVER_ITERATIVE
  // or SUNLINEARSOLVER_MATRIX_ITERATIVE, etc.
  return SUNLINEARSOLVER_DIRECT;
}
/* Create a custom SUNLinearSolver that delegates to PETSc */
SUNLinearSolver SUNLinSol_PETScCreate(N_Vector y_template, void *user_data)
{
  SUNLinearSolver S;
  PETScLSContent  content;
  PetscErrorCode  ierr;
  MPI_Comm        comm;

  PetscFunctionBeginUser;

  /* 1) Allocate the SUNDIALS LinearSolver object */
  S = (SUNLinearSolver) malloc(sizeof(*S));
  if (!S) return NULL;
  /* Zero-out the ops first */
  memset(S, 0, sizeof(*S));

  /* 2) Allocate our custom content struct */
  content = (PETScLSContent) malloc(sizeof(*content));
  if (!content) { free(S); return NULL; }
  memset(content, 0, sizeof(*content));

  S->content = content;
  /* Fill the ops that SUNDIALS requires */
  S->ops->gettype  = NULL; /* optional */
  S->ops->initialize = NULL; /* we won't implement separate initialize */
  S->ops->setup    = PETScLS_Setup;
  S->ops->solve    = PETScLS_Solve;
  S->ops->free     = PETScLS_Free;
  content->last_flag = 0;

  /* 3) Create PETSc objects */
  /* We'll assume PETSc is already initialized from main(). */
  comm = PETSC_COMM_WORLD;

  /* KSP create */
  ierr = KSPCreate(comm, &content->ksp); CHKERRABORT(comm, ierr);
  ierr = KSPSetType(content->ksp, KSPGMRES); CHKERRABORT(comm, ierr);
  /* You can choose any PC type, e.g. ILU for a 2x2 is silly but let's do it: */
  {
    PC pc;
    ierr = KSPGetPC(content->ksp, &pc); CHKERRABORT(comm, ierr);
    ierr = PCSetType(pc, PCILU); CHKERRABORT(comm, ierr);
  }

  /* Create small 2x2 matrix for J (but let's keep it general in code) */
  PetscInt nloc = 2, N = 2;
  ierr = MatCreate(comm, &content->J); CHKERRABORT(comm, ierr);
  ierr = MatSetSizes(content->J, nloc, nloc, N, N); CHKERRABORT(comm, ierr);
  ierr = MatSetType(content->J, MATAIJ); CHKERRABORT(comm, ierr);
  ierr = MatSetUp(content->J); CHKERRABORT(comm, ierr);

  /* Create vectors for x, b in PETSc that match dimension 2 */
  ierr = VecCreate(comm, &content->xpetsc); CHKERRABORT(comm, ierr);
  ierr = VecSetSizes(content->xpetsc, nloc, N); CHKERRABORT(comm, ierr);
  ierr = VecSetType(content->xpetsc, VECMPI); CHKERRABORT(comm, ierr);

  ierr = VecDuplicate(content->xpetsc, &content->bpetsc); CHKERRABORT(comm, ierr);
  {
    MatColoring  matcoloring;
    ISColoring   iscoloring;
  
    // 1) Create a MatColoring object for the given matrix
    ierr = MatColoringCreate(content->J, &matcoloring); CHKERRABORT(comm, ierr);
  
    // 2) Choose a coloring type, e.g. "SL" (sequential largest-first)
    ierr = MatColoringSetType(matcoloring, MATCOLORINGSL); CHKERRABORT(comm, ierr);
  
    // 3) Optionally allow user to override coloring settings via the command line
    ierr = MatColoringSetFromOptions(matcoloring); CHKERRABORT(comm, ierr);
  
    // 4) Apply the coloring to get an ISColoring
    ierr = MatColoringApply(matcoloring, &iscoloring); CHKERRABORT(comm, ierr);
  
    // 5) We no longer need the MatColoring object itself
    ierr = MatColoringDestroy(&matcoloring); CHKERRABORT(comm, ierr);
  
    // 6) Now create the FD coloring structure using that ISColoring
    ierr = MatFDColoringCreate(content->J, iscoloring, &content->fdcoloring); CHKERRABORT(comm, ierr);
  
    // 7) (Optional) Set the function used by FD coloring
    ierr = MatFDColoringSetFunction(content->fdcoloring, NULL, user_data); CHKERRABORT(comm, ierr);
  
    // 8) Read FD coloring options from command line
    ierr = MatFDColoringSetFromOptions(content->fdcoloring); CHKERRABORT(comm, ierr);
  
    // 9) Final setup
    ierr = MatFDColoringSetUp(content->J, iscoloring, content->fdcoloring); CHKERRABORT(comm, ierr);
  
    // 10) Clean up the ISColoring
    ierr = ISColoringDestroy(&iscoloring); CHKERRABORT(comm, ierr);
  }

  content->J_initialized = PETSC_FALSE;

  /* SUNDIALS internal properties */
  S->ops->gettype = MySolverGetType;
  content->last_flag = 0;

  /* We will store user_data or the N_Vector y if needed. Let's do that externally. */
  content->sundials_y = NULL;

  PetscFunctionReturn(S);
}

/* Setup routine: This is called by CVODES when it wants to (re)assemble or factor the Jacobian */
static int PETScLS_Setup(SUNLinearSolver S, SUNMatrix A)
{
  PETScLSContent content = (PETScLSContent) S->content;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  Vec            x   = content->xpetsc;  
  Mat            J   = content->J;

  /* We'll do FD coloring if we have a current state vector from CVODES */
  /* That means we must have stored the "current" N_Vector y somewhere.   */
  N_Vector y_sun = content->sundials_y;
  if (!y_sun) return 0;  /* If we never got a y yet, skip. */

  comm = PetscObjectComm((PetscObject)J);

  /* 1) Convert y_sun -> x so we can pass x to FD coloring. */
  PetscScalar *xarray;
  sunrealtype *ydata = N_VGetArrayPointer(y_sun);
  ierr = VecGetArray(x, &xarray); CHKERRABORT(comm, ierr);
  xarray[0] = ydata[0];
  xarray[1] = ydata[1];
  ierr = VecRestoreArray(x, &xarray); CHKERRABORT(comm, ierr);

  /* 2) Zero out J, apply FD coloring to approximate the Jacobian. 
        In a real PDE code, you'd do a PDE-based form or partial FD. */
  ierr = MatZeroEntries(J); CHKERRABORT(comm, ierr);

  /* We will define a local function for computing f(t,y). We do it inline here. 
     Because MatFDColoringSetFunction is a bit tricky to pass the time value or user_data. 
     We'll do a mini-lambda capturing user_data. 
  */
  /* However, we need to define the function that FD coloring calls: 
       MatFDColoringFunction(Mat, Vec, Vec, void *ctx)
     Typically, that function is a static function we pass to MatFDColoringSetFunction. 
     For brevity, we'll do a small local function. */

  /* Instead of hooking directly, we call:
     MatFDColoringApply(J, fdcoloring, x)
     which internally calls the function we set. 
     We'll see we set it as a 'NULL, user_data' above, so we do need a global function. 
     But let's do a trick: we can temporarily override the coloring function pointer. 
  */

  /* 2a) We'll forcibly set the function pointer with a manual approach: */
  extern PetscErrorCode MyMatFDColoringFunction(MatFDColoring,PetscErrorCode(*f)(void*,Vec,Vec),void *ctx);
  ierr = MatFDColoringSetFunction(content->fdcoloring, MyMatFDColoringFunction, NULL); /* ctx is user_data */
  
  CHKERRQ(ierr);
  content->fdcoloring->fctx = (void*) y_sun; /* store y in fctx, 
                                                but be aware we also need mu, which we store somewhere else. 
                                                Let's assume we can get it from CVODES. 
                                              */

  /* 2b) Actually apply FD coloring: */
  ierr = MatFDColoringApply(J, content->fdcoloring, x); CHKERRABORT(comm, ierr);

  /* 3) Now we attach J to KSP as the operator (and PC matrix). */
  ierr = KSPSetOperators(content->ksp, J, J); CHKERRABORT(comm, ierr);
  ierr = KSPSetUp(content->ksp); CHKERRABORT(comm, ierr);

  content->J_initialized = PETSC_TRUE;
  return 0; /* success */
}

/* We'll define the actual FD coloring function. 
   The signature for a MatFDColoring function is:
   PetscErrorCode func(void *ctx, Vec x, Vec f)
   We'll interpret ctx as a pointer to the N_Vector (and user data).
*/
PetscErrorCode MyMatFDColoringFunction(void *ctx, Vec x, Vec f)
{
  /* 'ctx' is content->fdcoloring->fctx. We stored an N_Vector in there. */
  /* But we also need mu. Let’s assume we can stash that in an accessible place. 
     One approach is to store a static global pointer. For a 2D system, it's not a big deal. 
  */
  PetscErrorCode ierr;
  MPI_Comm       comm;
  PetscScalar   *farray;
  const PetscScalar *xarray;

  /* We'll assume a global place or we retrieve from CVODES. For simplicity: */
  extern UserData global_udata;  /* We'll define a global pointer to the user data. */

  /* We'll also interpret 'ctx' as the N_Vector y. That might hold the current t as well, but let's skip that. */
  N_Vector y_sun = (N_Vector) ctx;
  /* But we see that 'x' might be the same data. 
     Anyway, we just need to compute f(t,y). We'll pick t from a global. 
     This is a toy example. 
  */

  PetscFunctionBeginUser;
  comm = PetscObjectComm((PetscObject)x);

  /* 1) Get the data from x (which is the same as y, presumably). */
  ierr = VecGetArrayRead(x, &xarray); CHKERRQ(ierr);
  /* 2) Evaluate f(t, x). We'll pick a global t=0 for demonstration. 
        Or you could store it somewhere else. 
  */
  PetscReal y1 = xarray[0];
  PetscReal y2 = xarray[1];
  PetscReal mu = global_udata->mu;  /* from the global user data */

  PetscReal f1 = y2;
  PetscReal f2 = mu*(1.0 - y1*y1)*y2 - y1;

  /* 3) Put that into the PETSc vector f */
  ierr = VecRestoreArrayRead(x, &xarray); CHKERRQ(ierr);
  ierr = VecGetArray(f, &farray); CHKERRQ(ierr);
  farray[0] = f1;
  farray[1] = f2;
  ierr = VecRestoreArray(f, &farray); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* The solve callback: solves J*x = b using KSP */
static int PETScLS_Solve(SUNLinearSolver S, SUNMatrix A,
                         N_Vector x_sun, N_Vector b_sun, realtype tol)
{
  PETScLSContent content = (PETScLSContent) S->content;
  Vec x = content->xpetsc;
  Vec b = content->bpetsc;
  PetscErrorCode ierr;
  MPI_Comm comm;
  PetscScalar *xarray;
  const realtype *bdata;

  if (!content->J_initialized) {
    /* If J wasn't set up yet, do it now? or error out. */
    return 0;
  }
  comm = PetscObjectComm((PetscObject)(content->J));

  /* 1) Copy b_sun -> b (PETSc Vec) */
  bdata = N_VGetArrayPointer(b_sun);
  ierr = VecSetValues(b, 1, &(PetscInt){0}, &bdata[0], INSERT_VALUES); CHKERRABORT(comm, ierr);
  ierr = VecSetValues(b, 1, &(PetscInt){1}, &bdata[1], INSERT_VALUES); CHKERRABORT(comm, ierr);
  ierr = VecAssemblyBegin(b); CHKERRABORT(comm, ierr);
  ierr = VecAssemblyEnd(b); CHKERRABORT(comm, ierr);

  /* 2) Solve */
  ierr = KSPSolve(content->ksp, b, x); CHKERRABORT(comm, ierr);

  /* 3) Copy x -> x_sun */
  ierr = VecGetArray(x, &xarray); CHKERRABORT(comm, ierr);
  NV_Ith_S(x_sun,0) = xarray[0];
  NV_Ith_S(x_sun,1) = xarray[1];
  ierr = VecRestoreArray(x, &xarray); CHKERRABORT(comm, ierr);

  return 0;
}

/* Free callback */
static int PETScLS_Free(SUNLinearSolver S)
{
  if (S == NULL) return 0;
  PETScLSContent content = (PETScLSContent) S->content;
  if (content) {
    if (content->ksp) KSPDestroy(&content->ksp);
    if (content->J)   MatDestroy(&content->J);
    if (content->xpetsc) VecDestroy(&content->xpetsc);
    if (content->bpetsc) VecDestroy(&content->bpetsc);
    if (content->fdcoloring) MatFDColoringDestroy(&content->fdcoloring);
    free(content);
  }
  free(S);
  return 0;
}

/* --------------------------------------------------------------------------
 *  Now we piece it all together in main().
 * -------------------------------------------------------------------------- */

/* We'll define a global to store user data & time, just for demonstration. */
UserData global_udata = NULL;
realtype global_t     = 0.0;  /* if needed */

int main(int argc, char** argv)
{
  PetscInitialize(&argc, &argv, NULL, "Help for VdP + CVODES + PETSc.\n");

  /* 1) Create user data, e.g. mu large => stiff system. */
  UserData udata = (UserData) malloc(sizeof(*udata));
  udata->mu = 100.0;  /* a stiff parameter */
  global_udata = udata;

  /* 2) Create CVODES object */
  void* cvode_mem = CVodeCreate(CV_BDF);  /* implicit BDF method */
  if (!cvode_mem) {
    printf("Error: CVodeCreate failed.\n");
    return -1;
  }

  /* 3) Create the serial N_Vector for y */
  N_Vector y = N_VNew_Serial(2);
  if (!y) {
    printf("Error: N_VNew_Serial failed.\n");
    return -1;
  }
  /* Set initial conditions, e.g. y(0) = 2, y'(0)=0 */
  NV_Ith_S(y,0) = 2.0;  /* y1(0) */
  NV_Ith_S(y,1) = 0.0;  /* y2(0) */

  /* 4) Initialize CVODE to solve y' = f(t,y), t in [0, Tfinal] */
  realtype t0 = 0.0;
  realtype tout = 50.0;
  int flag = CVodeInit(cvode_mem, f, t0, y);
  if (flag != CV_SUCCESS) {
    printf("Error: CVodeInit failed.\n");
    return -1;
  }

  /* Provide user_data so f() can see mu */
  CVodeSetUserData(cvode_mem, (void*)udata);

  /* 5) Set any tolerances (absolute, relative) */
  realtype reltol = 1e-6;
  realtype abstol = 1e-9;
  CVodeSStolerances(cvode_mem, reltol, abstol);

  /* 6) Create our custom PETSc-based SUNLinearSolver */
  SUNLinearSolver LS = SUNLinSol_PETScCreate(y, (void*)udata);
  if (!LS) {
    printf("Error: SUNLinSol_PETScCreate failed.\n");
    return -1;
  }

  /* Attach the linear solver to CVODES. We won't use a separate SUNMatrix. */
  /* For a matrix-based approach, you could do: 
     SUNMatrix A = SUNDenseMatrix(2,2);
     CVodeSetLinearSolver(cvode_mem, LS, A);
     ...
     But here we skip that and do a "matrixless" from SUNDIALS perspective,
     while inside we do have a PETSc matrix. 
  */
  flag = CVodeSetLinearSolver(cvode_mem, LS, NULL);
  if (flag != CVLS_SUCCESS) {
    printf("Error: CVodeSetLinearSolver failed.\n");
    return -1;
  }

  /* We also need to tell CVODES we will provide a "Jac" or at least a setup function.
     In SUNDIALS 6.x, we can do something like:
       CVodeSetJacFn(cvode_mem, MyJacFunction)
     But we are bridging. We'll rely on the solver's setup. 
     Actually, to trigger a call to solver->ops->setup, we can do:
       CVodeSetJacFn(cvode_mem, NULL);
     which means it'll use the built-in difference-quotient or something. 
     But we want our Setup to be called.  Let's do:
  */
  /* In older SUNDIALS, you might do CVSpilsSetJacTimes or something. We'll just do: */
  CVodeSetJacFn(cvode_mem, NULL);

  /* 7) In order for our Setup to see the current y, we hack a solution: 
       We'll update the 'content->sundials_y' each time we do a step. 
       A robust approach is to write a custom "CVodeDQJac" that sets it.
       For a small example, let's do it manually. 
  */
  PETScLSContent content = (PETScLSContent)(LS->content);

  /* 8) Time stepping loop. We'll do a simple loop, calling CVode each time. */
  realtype t = t0;
  while (t < tout) {
    /* keep the solver from going beyond tout */
    realtype tnext = (t + 1.0 > tout) ? tout : t + 1.0;

    /* Store y in the solver so the Setup() can see it. */
    content->sundials_y = y;  

    flag = CVode(cvode_mem, tnext, y, &t, CV_NORMAL);
    if (flag < 0) {
      printf("CVode error flag=%d\n", flag);
      break;
    }

    printf("t=%g  y=(%.4f, %.4f)\n", t, NV_Ith_S(y,0), NV_Ith_S(y,1));

    if (fabs(t - tout) < 1e-14) break;  /* done */
  }

  /* Free everything */
  N_VDestroy(y);
  CVodeFree(&cvode_mem);
  SUNLinSolFree(LS);

  free(udata);
  PetscFinalize();
  return 0;
}