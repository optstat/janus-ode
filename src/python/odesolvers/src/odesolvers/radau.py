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


def _closure_from_ksp(ksp: PETSc.KSP, proto: torch.Tensor) -> Callable:
    """
    Build `solve(b)` that accepts a 1-D torch tensor and returns x as torch.
    Memory is reused across calls.
    """
    x_vec, b_vec = ksp.getVecs()

    def solve(b: torch.Tensor) -> torch.Tensor:
        b_vec.array[:] = b.detach().cpu().ravel()      # RHS → PETSc Vec
        ksp.solve(b_vec, x_vec)
        return torch.as_tensor(x_vec.array,
                               dtype=proto.dtype).clone().view_as(b)
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
        A_petsc = PETSc.Mat().createAIJ(Mass.shape,
                                        csr=_torch_csr(B_torch))
        A_petsc.assemble()

        ksp = _make_ksp(A_petsc, tol=tol,
                        maxits=maxits, pc_type=pc)
        solve_fns.append(_closure_from_ksp(ksp, Mass))

    return solve_fns



def _vandermonde(c: torch.Tensor, power: int) -> torch.Tensor:
    """Row-wise Vandermonde: V[i,j] = c[i]**j  for j=0…power-1."""
    exps = torch.arange(power, dtype=c.dtype, device=c.device)
    return c.unsqueeze(1).pow(exps)          # (s, power)

def _integral_vandermonde(c: torch.Tensor) -> torch.Tensor:
    """
    Int-Vandermonde: Q[i,j] = c[i]**(j+1)/(j+1)  (∫₀ᶜ τʲ dτ)
    """
    s = c.numel()
    exps = torch.arange(1, s + 1, dtype=c.dtype, device=c.device)
    return c.unsqueeze(1).pow(exps) / exps

def _stage_matrices(c: torch.Tensor) -> Tuple[torch.Tensor,
                                              torch.Tensor,
                                              torch.Tensor]:
    """
    Build A, its eigenvectors T and inverse-eigenvalue diag ValP
    for a given abscissa vector `c` (Radau II-A).
    """
    s  = c.numel()
    CP = _vandermonde(c, s)                 # C power 0…s-1
    CQ = _integral_vandermonde(c)           # ∫ τʲ dτ,  j=0…s-1
    A  = CQ @ torch.linalg.inv(CP)          #   = CQ / CP  (MATLAB notation)

    # eigen-decomposition (complex128 ⇒ best accuracy)
    eigvals, T = torch.linalg.eig(A.to(torch.complex128))  # A·T = T·Λ
    TI         = torch.linalg.inv(T)
    ValP       = 1.0 / eigvals            # diag(inv(Λ))

    return T, TI, ValP


# --------------------------------------------------------------------------- #
def coertv1(dtype=torch.float64, device="cpu"):
    """
    Implicit-Euler (1-stage Radau)  – trivial coefficients.
    """
    C   = torch.tensor([1.0], dtype=dtype, device=device)
    T   = TI = torch.tensor([[1.0]], dtype=dtype, device=device)
    Val = torch.tensor([1.0],  dtype=dtype, device=device)
    Dd  = torch.tensor([-1.0], dtype=dtype, device=device)
    return T, TI, C, Val, Dd


def coertv3(dtype=torch.float64, device="cpu"):
    """
    3-stage Radau II-A  (order 5)
    """
    sq6 = math.sqrt(6.0)
    C   = torch.tensor([(4.0 - sq6)/10.0,
                        (4.0 + sq6)/10.0,
                        1.0], dtype=dtype, device=device)

    # build T, TI, ValP directly from the Butcher matrix
    T, TI, ValP = _stage_matrices(C)

    Dd  = torch.tensor([-(13.0 + 7.0*sq6)/3.0,
                        (-13.0 + 7.0*sq6)/3.0,
                        -1.0/3.0], dtype=dtype, device=device)
    return T, TI, C, ValP, Dd


def coertv5(dtype=torch.float64, device="cpu"):
    """
    5-stage Radau II-A  (order 9)
    """
    C = torch.tensor([0.05710419611451768219312,
                      0.27684301363812382768,
                      0.5835904323689168200567,
                      0.8602401356562194478479,
                      1.0], dtype=dtype, device=device)

    T, TI, ValP = _stage_matrices(C)

    Dd = torch.tensor([-27.78093394406463730479,
                        3.641478498049213152712,
                       -1.252547721169118720491,
                        0.5920031671845428725662,
                       -0.2],
                      dtype=dtype, device=device)
    return T, TI, C, ValP, Dd


def coertv7(dtype=torch.float64, device="cpu"):
    """
    7-stage Radau II-A  (order 13)
    """
    C = torch.tensor([0.02931642715978489197205,
                      0.14807859966848429185,
                      0.3369846902811542990971,
                      0.5586715187715501320814,
                      0.7692338620300545009169,
                      0.9269456713197411148519,
                      1.0], dtype=dtype, device=device)

    T, TI, ValP = _stage_matrices(C)

    Dd = torch.tensor([-54.37443689412861451458,
                        7.000024004259186512041,
                       -2.355661091987557192256,
                        1.132289066106134386384,
                       -0.6468913267673587118673,
                        0.3875333853753523774248,
                       -0.1428571428571428571429],
                      dtype=dtype, device=device)
    return T, TI, C, ValP, Dd