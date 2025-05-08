# radau_tables.py
import math
import torch
from typing import Tuple

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