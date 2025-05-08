# bdf2_torch.py — adaptive A‑stable BDF2 integrator (real‑valued, PyTorch)
"""Second‑order backward differentiation formula with step‑size control.

* **A‑stable**, good for stiff problems like Van‑der‑Pol μ≈[1–1000].
* One Newton loop per step; LU is *reused* inside the loop.
* Error estimate uses the embedded first‑order BDF1 (implicit Euler).
* Designed to be dual‑number friendly: only `f(t,y)` and `jac(t,y)` need dual
  support; the linear solve is pure real Torch.
"""
from __future__ import annotations
from typing import Callable, Optional
import math, torch
Tensor = torch.Tensor

# ---------------- helper ----------------
_norm = lambda v: math.sqrt(torch.mean(v*v).item())
EPS   = torch.finfo(torch.float32).eps

# ---------------- main class ----------------
class BDF2:
    def __init__(self, f:Callable[[float,Tensor],Tensor], jac:Callable[[float,Tensor],Tensor],
                 t0:float, y0:Tensor, t_bound:float,
                 *, rtol=1e-6, atol=1e-9,
                 first_step:Optional[float]=None, max_step:float=math.inf):
        self.f, self.jac = f, jac
        self.t, self.y   = t0, y0.clone()
        self.TB          = t_bound; self.dir = 1. if t_bound>=t0 else -1.
        self.rtol, self.atol, self.max_step = rtol, atol, max_step
        self.n = y0.numel()
        # first step guess
        f0n = torch.norm(f(t0,y0)).item()
        self.h = first_step or 0.3/max(1e-4,f0n)
        self.h = min(self.h, max_step)
        self.J = jac(t0,y0)
        # store y_{n-1} for multistep; initialise with Euler
        self.y_prev = y0.clone()
        self.prev_f = f(t0,y0)

    # -------- Newton solve helper --------
    def _newton(self, h:float, rhs:Tensor)->Tensor:
        I = torch.eye(self.n, dtype=self.y.dtype, device=self.y.device)
        LU = torch.linalg.lu_factor(I - (2.0/3.0)*h*self.J)  # α=2/3 for BDF2
        for _ in range(4):
            dy = torch.linalg.lu_solve(*LU, rhs.unsqueeze(1)).squeeze(1)
            y_new = self.y_pred + dy
            f_new = self.f(self.t+h, y_new)
            rhs   = (2.0/3.0)*h*f_new - (2.0/3.0)*self.y_pred + (1.0/3.0)*self.y_prev - y_new
            if _norm(dy)/(self.atol+ self.rtol*_norm(y_new)) < 1e-7:
                return y_new, f_new
        return y_new, f_new  # return last iterate even if not fully converged

    # -------- single adaptive step --------
    def step(self):
        t, y, h = self.t, self.y, self.h
        if self.dir*(t+h - self.TB) > 0: h = self.TB - t
        if h==0: return False
        # multistep predictor: linear extrapolation y_pred = y + h*(f - prev_f)
        self.y_pred = y + h*(self.f(t,y) - self.prev_f)
        # Newton solve for implicit BDF2
        rhs0 = (2.0/3.0)*h*self.f(t+h, self.y_pred) - (2.0/3.0)*self.y_pred + (1.0/3.0)*self.y_prev - self.y_pred
        y_new, f_new = self._newton(h, rhs0)
        # BDF1 solution for error estimate
        y_euler = (y + h*self.f(t+h, y_new) )  # implicit Euler (one Newton already done)
        err = y_new - y_euler
        err_norm = torch.norm(err / (self.atol + self.rtol*y_new.abs()))/math.sqrt(self.n)
        if err_norm>1:
            self.h *= max(0.4, 0.9*err_norm**(-1/3))
            return self.step()
        # accept
        self.t += h; self.y_prev = y; self.prev_f = self.f(t,y)
        self.y  = y_new; self.J = self.jac(self.t, y_new)
        self.h  = min(self.h*min(5., 0.9*err_norm**(-1/3)), self.max_step)
        return True

# ---------------- smoke test ----------------
if __name__ == "__main__":
    mu=5.0
    def vdp(t,y):
        return torch.stack([y[1], mu*(1-y[0]**2)*y[1]-y[0]])
    def jac_vdp(t,y):
        return torch.tensor([[0.,1.],[-2*mu*y[0]*y[1]-1., mu*(1-y[0]**2)]],dtype=y.dtype)
    y0=torch.tensor([2.,0.])
    s=BDF2(vdp,jac_vdp,0.,y0,20.)
    n=0
    while s.step():
        n+=1
        if s.t>=20.: break
    print("steps",n,"y",s.y)
