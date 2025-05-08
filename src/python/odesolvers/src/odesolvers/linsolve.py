# radau_tables.py
import math
import torch
from typing import Tuple
import scipy.sparse as sp
from petsc4py import PETSc
from typing import List, Callable


# --------------------------------------------------------------------------- #
# ---- helpers -------------------------------------------------------------- #
def _torch_csr(t: torch.Tensor) -> sp.csr_matrix:
    """Torch → SciPy CSR without extra copies."""
    return sp.csr_matrix(t.detach().cpu().numpy(), copy=False)


def _make_ksp(A: PETSc.Mat,
              *,
              tol: float  = 1e-10,
              maxits: int = 500,
              pc_type: str = "ilu") -> PETSc.KSP:
    ksp = PETSc.KSP().create()
    ksp.setOperators(A)
    ksp.setType("gmres")
    ksp.setTolerances(rtol=tol, max_it=maxits)
    ksp.pc.setType(pc_type)
    ksp.setFromOptions()                 # honour -ksp_* command-line flags
    return ksp

def _torch_to_petsc_aij(t: torch.Tensor) -> PETSc.Mat:
    """
    Convert a dense (n×n) torch tensor (real or complex) into a PETSc AIJ
    matrix via CSR insertion.  Works on all PETSc ≥3.15.
    """
    csr = sp.csr_matrix(t.detach().cpu().numpy(), copy=False)

    A = PETSc.Mat().createAIJ(size=csr.shape, nnz=csr.nnz)
    A.setValuesCSR(csr.indptr, csr.indices, csr.data)
    A.assemble()
    return A


# -- version-independent vector allocation ----------------------------------
def _closure_from_ksp(ksp: PETSc.KSP, proto: torch.Tensor):
    A_mat, _     = ksp.getOperators()      # works on all PETSc versions
    x_vec, b_vec = A_mat.createVecs()

    def solve(b: torch.Tensor):
        b_vec.array[:] = b.detach().cpu().ravel()
        ksp.solve(b_vec, x_vec)
        target_dtype = (torch.complex128 if x_vec.array.dtype.kind == "c"
                        else torch.float64)
        return torch.as_tensor(x_vec.array, dtype=target_dtype).clone()

    return solve

# --------------------------------------------------------------------------- #


def decom_rc_gmres(h      : float,
                   ValP   : torch.Tensor,
                   Mass   : torch.Tensor,
                   Jac    : torch.Tensor,
                   real   : bool = True,
                   *,
                   tol    : float = 1e-10,
                   maxits : int   = 500,
                   pc     : str   = "ilu") -> List[Callable]:
    """
    GMRES analogue of MATLAB’s DecomRC.

    Parameters
    ----------
    h      : current step-size
    ValP   : inverse eigen-values of A   (length = #stages)
              – exactly what `coertv{s}` returns.
    Mass   : mass matrix  (torch, n×n, **CPU**)
    Jac    : Jacobian     (torch, n×n, **CPU**)
    real   : if True,     ValP is [γ, Re α₁, Im α₁, Re α₂, Im α₂, …]
              if False,    ValP already contains the complex λ       (old “complex” branch)
    tol, maxits, pc : KSP tuning knobs.

    Returns
    -------
    solve_fns : list of callables, one per Radau stage:
                ``x = solve_fns[k](b)``  solves (γₖ M – J) x = b
    """

    # 1 · unpack stage abscissae (γₖ)
    if real:
        gammas = [ValP[0]]                           # γ   (real)
        for q in range(1, (len(ValP) - 1) // 2 + 1):
            re = ValP[2*q - 1]
            im = ValP[2*q]
            gammas.append(re + 1j * im)              # αₖ (complex)
    else:                                            # already complex
        gammas = list(ValP)

    # 2 · build a KSP per stage
    solve_fns = []
    Mass_c128 = Mass.to(torch.complex128)
    Jac_c128  = Jac.to(torch.complex128)

    for γ in gammas:
        B_torch = γ * Mass_c128 - Jac_c128
        A_petsc = _torch_to_petsc_aij(B_torch)        # <-- one-liner
        ksp = _make_ksp(A_petsc, tol=tol,
                        maxits=maxits, pc_type=pc)
        solve_fns.append(_closure_from_ksp(ksp, Mass))

    return solve_fns



def build_lu_solvers(h, ValP, Mass, Jac, *, real_layout=True):
    """
    Return a list of closures solve_fns[k](b) that solve
        (γ_k/h * M – J) x = b
    with dense LU in pure PyTorch.
    """
    inv_h  = 1.0 / h
    Mass_c = Mass.to(torch.complex128)
    Jac_c  = Jac.to(torch.complex128)

    if real_layout:
        gammas = [ValP[0]] + [
            ValP[i] + 1j * ValP[i + 1] for i in range(1, len(ValP), 2)
        ]
    else:
        gammas = list(ValP)

    solve_fns = []
    for γ in gammas:
        B = γ * inv_h * Mass_c - Jac_c                # (n,n) complex128
        LU, piv = torch.linalg.lu_factor(B)           # pre-factorise once

        def solve(b, LU=LU, piv=piv):
            # make RHS at least 2-D
            b_col = b.to(torch.complex128).unsqueeze(-1)      # (n,1)
            x_col = torch.linalg.lu_solve(LU, piv, b_col)     # (n,1)
            return x_col.squeeze(-1)                          # back to (n,)

        solve_fns.append(solve)

    return solve_fns
