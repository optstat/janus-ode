"""
rosebrock_torch_gmres.py – ROS4 with PETSc GMRES
=============================================

Stiff-stable 4-stage Rosenbrock integrator that uses PETSc’s GMRES
(KSP) with an ILU(0) preconditioner instead of dense LU.

You need:  petsc  +  petsc4py  +  torch  +  scipy  +  matplotlib
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List
import time, math, torch
from petsc4py import PETSc



# --- helper ---
def _t(val, *, dtype=torch.float64, device="cpu"):
    if isinstance(val, torch.Tensor):
        return val.to(dtype=dtype, device=device)
    return torch.tensor(val, dtype=dtype, device=device)


# ----------------------------------------------------------- ROS4 coefficients
@dataclass(frozen=True)
class RosenbrockCoeffs:
    # abscissae
    c2: torch.Tensor; c3: torch.Tensor; c4: torch.Tensor
    # A-matrix
    a21: torch.Tensor; a31: torch.Tensor; a32: torch.Tensor
    a41: torch.Tensor; a42: torch.Tensor; a43: torch.Tensor
    a51: torch.Tensor; a52: torch.Tensor; a53: torch.Tensor; a54: torch.Tensor
    # C-matrix
    c21: torch.Tensor; c31: torch.Tensor; c32: torch.Tensor
    c41: torch.Tensor; c42: torch.Tensor; c43: torch.Tensor
    c51: torch.Tensor; c52: torch.Tensor; c53: torch.Tensor; c54: torch.Tensor
    c61: torch.Tensor; c62: torch.Tensor; c63: torch.Tensor; c64: torch.Tensor; c65: torch.Tensor
    # gamma
    gam: torch.Tensor
    # dense/error coeffs
    d1: torch.Tensor; d2: torch.Tensor; d3: torch.Tensor; d4: torch.Tensor
    d21: torch.Tensor; d22: torch.Tensor; d23: torch.Tensor; d24: torch.Tensor; d25: torch.Tensor
    d31: torch.Tensor; d32: torch.Tensor; d33: torch.Tensor; d34: torch.Tensor; d35: torch.Tensor

    @staticmethod
    def default(dtype=torch.float64, device="cpu") -> "RosenbrockCoeffs":
        t = lambda v: _t(v, dtype=dtype, device=device)
        return RosenbrockCoeffs(
            c2=t(0.386), c3=t(0.21), c4=t(0.63),
            a21=t(1.544), a31=t(0.9466785280815826), a32=t(0.2557011698983284),
            a41=t(3.314825187068521), a42=t(2.896124015972201), a43=t(0.9986419139977817),
            a51=t(1.221224509226641), a52=t(6.019134481288629), a53=t(12.53708332932087), a54=t(-0.687886036105895),
            c21=t(-5.6688), c31=t(-2.430093356833875), c32=t(-0.2063599157091915),
            c41=t(-0.1073529058151375), c42=t(-9.594562251023355), c43=t(-20.47028614809616),
            c51=t( 7.496443313967647), c52=t(-10.24680431464352), c53=t(-33.99990352819905), c54=t(11.7089089320616),
            c61=t( 8.083246795921522), c62=t(-7.981132988064893), c63=t(-31.52159432874371),
            c64=t(16.31930543123136),  c65=t(-6.058818238834054),
            gam=t(0.25),
            d1=t(0.25), d2=t(-0.1043), d3=t(0.1035), d4=t(-0.0362),
            d21=t(10.12623508344586), d22=t(-7.487995877610167), d23=t(-34.80091861555747),
            d24=t(-7.992771707568823), d25=t(1.025137723295662),
            d31=t(-0.6762803392801253), d32=t(6.087714651680015), d33=t(16.43084320892478),
            d34=t(24.76722511418386),  d35=t(-6.594389125716872),
        )


# ---------------------------------------------------- adaptive step controller
@dataclass
class Controller:
    dtype: torch.dtype; device: str | torch.device
    reject: bool = False; first_step: bool = True
    errold: torch.Tensor = field(init=False)
    hold:   torch.Tensor = field(init=False)
    hnext:  torch.Tensor = field(init=False)

    def __post_init__(self):
        self.errold = _t(1.0, dtype=self.dtype, device=self.device)
        self.hold   = _t(0.0, dtype=self.dtype, device=self.device)
        self.hnext  = _t(0.0, dtype=self.dtype, device=self.device)

    def success(self, err: torch.Tensor, h: torch.Tensor):
        safe, fac1, fac2 = 0.9, 5.0, 1/6
        fac  = torch.clamp(err.pow(0.25)/safe, fac2, fac1)
        hnew = h/fac
        if err <= 1:
            if not self.first_step:
                facpred = (self.hold/h)*(err*err/self.errold).pow(0.25)/safe
                facpred = torch.clamp(facpred, fac2, fac1)
                hnew    = h/torch.maximum(fac, facpred)
            self.first_step, self.hold = False, h
            self.errold = torch.maximum(_t(0.01, dtype=err.dtype, device=err.device), err)
            if self.reject:  # step was rejected before
                hnew = torch.minimum(hnew, h) if h >= 0 else torch.maximum(hnew, h)
            self.hnext, self.reject = hnew, False
            return True, h
        # reject
        self.reject = True
        return False, hnew


# ------------------------------------------------------------- ROS4 + GMRES --
@dataclass
class RosenbrockSolver:
    rhs: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    y0:  torch.Tensor
    t0:  float = 0.0
    jac: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    atol: float = 1e-6; rtol: float = 1e-3
    coeffs: RosenbrockCoeffs = field(default_factory=RosenbrockCoeffs.default)

    # internal
    t: torch.Tensor = field(init=False); y: torch.Tensor = field(init=False)
    dydt: torch.Tensor = field(init=False); ctrl: Controller = field(init=False)
    _alloced: bool = field(init=False, default=False)

    # PETSc
    A: PETSc.Mat = field(init=False); b_p: PETSc.Vec = field(init=False); x_p: PETSc.Vec = field(init=False)
    ksp: PETSc.KSP = field(init=False)

    # ---------------------------------------------------------------- init ---
    def __post_init__(self):
        if self.jac is None:
            self.jac = self._autograd_jac
        self.t  = _t(self.t0, dtype=self.y0.dtype, device=self.y0.device)
        self.y  = self.y0.clone()
        self.dydt = self.rhs(self.t, self.y)
        self.ctrl = Controller(dtype=self.y.dtype, device=self.y.device)

    # ------------------------------------------------- allocate work + PETSc --
    def _alloc(self):
        n, d, dev = self.y.numel(), self.y.dtype, self.y.device
        Z = lambda: torch.empty(n, dtype=d, device=dev)
        self.dfdy = torch.empty((n, n), dtype=d, device=dev)
        self.k1, self.k2, self.k3, self.k4, self.k5 = Z(), Z(), Z(), Z(), Z()
        self.yerr, self.yout = Z(), Z()
        # PETSc linear system
        self.A = PETSc.Mat().createAIJ([n, n]); self.A.setUp()
        self.b_p, self.x_p = PETSc.Vec().createSeq(n), PETSc.Vec().createSeq(n)
        self.ksp = PETSc.KSP().create(); self.ksp.setOperators(self.A)
        self.ksp.setType("gmres"); self.ksp.getPC().setType("ilu")
        self.ksp.setTolerances(rtol=1e-10, atol=1e-14, max_it=n*4)
        self.ksp.setFromOptions()
        self._alloced = True

    # ------------------------------------------------- PETSc data helpers -----
    def _mat_from_torch(self, M: torch.Tensor):
        a = M.detach().cpu().numpy(); n = a.shape[0]
        self.A.zeroEntries()
        self.A.setValues(range(n), range(n), a, addv=False)
        self.A.assemblyBegin(); self.A.assemblyEnd()

    def _vec_from_torch(self, src: torch.Tensor, dst: PETSc.Vec):
        dst.setArray(src.detach().cpu().numpy())

    # --- PETSc solve result ---
    def _solve(self, rhs):
        self._vec_from_torch(rhs, self.b_p)
        self.ksp.solve(self.b_p, self.x_p)
        # respect PETSc’s scalar type
        return torch.from_numpy(self.x_p.getArray()).to(device=rhs.device, dtype=rhs.dtype).clone()

    # ------------------------------------------------- autograd Jacobian ------
    def _autograd_jac(self, t: torch.Tensor, y: torch.Tensor):
        y_req = y.detach().requires_grad_(True)
        return torch.autograd.functional.jacobian(lambda z: self.rhs(t, z), y_req)

    # ------------------------------------------------------------- one step ---
    def _ros_step(self, h: torch.Tensor):
        c = self.coeffs
        A_mat = -self.dfdy.clone()
        A_mat.diagonal().add_(1.0/(c.gam*h))
        self._mat_from_torch(A_mat)

        # k1
        self.k1[:] = self._solve(self.dydt)
        # k2
        y2 = self.y + c.a21*self.k1
        f2 = self.rhs(self.t + c.c2*h, y2)
        self.k2[:] = self._solve(f2 + c.c21*self.k1/h)
        # k3
        y3 = self.y + c.a31*self.k1 + c.a32*self.k2
        f3 = self.rhs(self.t + c.c3*h, y3)
        self.k3[:] = self._solve(f3 + (c.c31*self.k1 + c.c32*self.k2)/h)
        # k4
        y4 = self.y + c.a41*self.k1 + c.a42*self.k2 + c.a43*self.k3
        f4 = self.rhs(self.t + c.c4*h, y4)
        self.k4[:] = self._solve(f4 + (c.c41*self.k1 + c.c42*self.k2 + c.c43*self.k3)/h)
        # k5 + error
        y5 = self.y + c.a51*self.k1 + c.a52*self.k2 + c.a53*self.k3 + c.a54*self.k4
        f5 = self.rhs(self.t + h, y5)
        self.k5[:] = self._solve(f5 + (c.c51*self.k1 + c.c52*self.k2 + c.c53*self.k3 + c.c54*self.k4)/h)
        ytemp = y5 + self.k5
        f6 = self.rhs(self.t + h, ytemp)
        rhs6 = f6 + (c.c61*self.k1 + c.c62*self.k2 + c.c63*self.k3 + c.c64*self.k4 + c.c65*self.k5)/h
        self.yerr[:] = self._solve(rhs6)
        self.yout[:] = ytemp + self.yerr

    # ------------------------------------------------ error norm -------------
    def _error(self):
        sk = self.atol + self.rtol*torch.maximum(torch.abs(self.y), torch.abs(self.yout))
        return torch.sqrt(torch.mean((self.yerr/sk)**2))

    # ------------------------------------------------ public step ------------
    def step(self, h_try: float):
        if not self._alloced: self._alloc()
        h = _t(h_try, dtype=self.y.dtype, device=self.y.device)
        self.dfdy[:] = self.jac(self.t, self.y)

        while True:
            self._ros_step(h)
            ok, h_new = self.ctrl.success(self._error(), h)
            if ok: break
            h = h_new
            if torch.abs(h) <= torch.abs(self.t)*torch.finfo(self.y.dtype).eps:
                raise RuntimeError("Step size underflow")

        self.dydt = self.rhs(self.t + h, self.yout)
        self.t += h; self.y[:] = self.yout
        return self.y, self.t, self.ctrl.hnext

    # ------------------------------------------------ integrate helper -------
    def solve(self, t_end: float, h0: float) -> List[Tuple[float, torch.Tensor]]:
        h = h0; out = [(float(self.t), self.y.clone())]
        while float(self.t) < t_end:
            if float(self.t + h) > t_end: h = t_end - float(self.t)
            y, t, h = self.step(h)
            out.append((float(t), y.clone()))
        return out


# ------------------------------------------------------------- demo / timing -
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.integrate import solve_ivp
    from textwrap import dedent

    torch.set_default_dtype(torch.float64)
    torch.set_printoptions(precision=17)   # show 17 significant digits
    device = "cpu"
    mu = 1000.0          # change to 10 or 100 for stiff timing
    ft = 3*mu
    # RHS + Jacobian ----------------------------------------------------------
    def vdp(t, y):
        x, xd = y
        return [xd, mu*(1 - x**2)*xd - x]

    def vdp_jac(t, y):
        x, xd = y
        return [[0.0, 1.0],
                [-2*mu*x*xd - 1, mu*(1 - x**2)]]

    def vdp_torch(t: torch.Tensor, y: torch.Tensor):
        return torch.tensor(vdp(float(t), y.cpu().tolist()), dtype=y.dtype, device=y.device)

    def jac_torch(t: torch.Tensor, y: torch.Tensor):
        return torch.tensor(vdp_jac(float(t), y.cpu().tolist()), dtype=y.dtype, device=y.device)

    # PyTorch ROS4 + GMRES ----------------------------------------------------
    y0 = torch.tensor([2.0, 0.0], device=device)
    ros = RosenbrockSolver(vdp_torch, y0, jac=jac_torch, atol=1e-7, rtol=1e-6)

    t0 = time.perf_counter()
    ros_traj = ros.solve(ft, 0.1)
    t_ros = time.perf_counter() - t0
    ros_final_t, ros_final_y = ros_traj[-1]
    print(f"ROS4  final @ t={ros_final_t:.2f}:  y = {ros_final_y}")
    # SciPy Radau -------------------------------------------------------------
    t0 = time.perf_counter()
    sol = solve_ivp(vdp, (0, ft), [2, 0], method="Radau", jac=vdp_jac,
                    atol=1e-7, rtol=1e-6)
    t_rad = time.perf_counter() - t0
    # after SciPy Radau integrate
    rad_final_t = sol.t[-1]
    rad_final_y = sol.y[:, -1]
    print(f"Radau final @ t={rad_final_t:.2f}:  y = {rad_final_y}")
    print(dedent(f"""
        ---- timing summary (μ={mu}) ----
        ROS4 + GMRES : {t_ros:.4f} s | steps = {len(ros_traj)-1}
        SciPy Radau  : {t_rad:.4f} s | steps = {len(sol.t)-1}
    """))

    # plot --------------------------------------------------------------------
    #plt.plot([t for t, _ in ros_traj], [y[0] for _, y in ros_traj], label="ROS4 x")
    #plt.plot(sol.t, sol.y[0], label="Radau x", ls="--", alpha=0.6)
    #plt.xlabel("t"); plt.legend(); plt.title("Van der Pol – ROS4 GMRES vs Radau")
    #plt.savefig("rosenbrock_gmres.png", dpi=300)
