"""
stepper_ross.py – Pythonic ROS4 (Rosenbrock) integrator
=======================================================

• Stiff-stable 4-stage Rosenbrock method (aka ROS4 / StepperRoss in
  Numerical Recipes) implemented with PyTorch tensors.
• Dataclasses tidy up coefficients, controller, and solver state.
• Optional autograd Jacobian fallback (hand JAC is still recommended).
• Convenience `solve()` wrapper and a runnable Van-der-Pol demo.

Requires: PyTorch ≥ 1.9 and matplotlib for the demo plot.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, List
import torch, math, matplotlib.pyplot as plt
from petsc4py import PETSc

# -----------------------------------------------------------------------------#
# utility: scalar tensor helper                                                #
# -----------------------------------------------------------------------------#
def _t(
    val: float,
    *,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    return torch.tensor(val, dtype=dtype, device=device)


# -----------------------------------------------------------------------------#
# Rosenbrock (ROS4) coefficients                                               #
# -----------------------------------------------------------------------------#
@dataclass(frozen=True)
class RosenbrockCoeffs:
    # stage abscissae
    c2: torch.Tensor
    c3: torch.Tensor
    c4: torch.Tensor

    # tableau entries
    a21: torch.Tensor
    a31: torch.Tensor
    a32: torch.Tensor
    a41: torch.Tensor
    a42: torch.Tensor
    a43: torch.Tensor
    a51: torch.Tensor
    a52: torch.Tensor
    a53: torch.Tensor
    a54: torch.Tensor

    # k-combination coefficients
    c21: torch.Tensor
    c31: torch.Tensor
    c32: torch.Tensor
    c41: torch.Tensor
    c42: torch.Tensor
    c43: torch.Tensor
    c51: torch.Tensor
    c52: torch.Tensor
    c53: torch.Tensor
    c54: torch.Tensor
    c61: torch.Tensor
    c62: torch.Tensor
    c63: torch.Tensor
    c64: torch.Tensor
    c65: torch.Tensor

    # diagonal shift
    gam: torch.Tensor

    # error / dense-output coeffs
    d1: torch.Tensor
    d2: torch.Tensor
    d3: torch.Tensor
    d4: torch.Tensor
    d21: torch.Tensor
    d22: torch.Tensor
    d23: torch.Tensor
    d24: torch.Tensor
    d25: torch.Tensor
    d31: torch.Tensor
    d32: torch.Tensor
    d33: torch.Tensor
    d34: torch.Tensor
    d35: torch.Tensor

    # factory with all the NR values
    @staticmethod
    def default(
        dtype: torch.dtype = torch.float64, device: str | torch.device = "cpu"
    ) -> "RosenbrockCoeffs":
        t = lambda v: _t(v, dtype=dtype, device=device)
        return RosenbrockCoeffs(
            c2=t(0.386),
            c3=t(0.21),
            c4=t(0.63),
            a21=t(1.544),
            a31=t(0.9466785280815826),
            a32=t(0.2557011698983284),
            a41=t(3.314825187068521),
            a42=t(2.896124015972201),
            a43=t(0.9986419139977817),
            a51=t(1.221224509226641),
            a52=t(6.019134481288629),
            a53=t(12.53708332932087),
            a54=t(-0.687886036105895),
            c21=t(-5.6688),
            c31=t(-2.430093356833875),
            c32=t(-0.2063599157091915),
            c41=t(-0.1073529058151375),
            c42=t(-9.594562251023355),
            c43=t(-20.47028614809616),
            c51=t(7.496443313967647),
            c52=t(-10.24680431464352),
            c53=t(-33.99990352819905),
            c54=t(11.7089089320616),
            c61=t(8.083246795921522),
            c62=t(-7.981132988064893),
            c63=t(-31.52159432874371),
            c64=t(16.31930543123136),
            c65=t(-6.058818238834054),
            gam=t(0.25),
            d1=t(0.25),
            d2=t(-0.1043),
            d3=t(0.1035),
            d4=t(-0.0362),
            d21=t(10.12623508344586),
            d22=t(-7.487995877610167),
            d23=t(-34.80091861555747),
            d24=t(-7.992771707568823),
            d25=t(1.025137723295662),
            d31=t(-0.6762803392801253),
            d32=t(6.087714651680015),
            d33=t(16.43084320892478),
            d34=t(24.76722511418386),
            d35=t(-6.594389125716872),
        )


# -----------------------------------------------------------------------------#
# adaptive step-size controller                                                #
# -----------------------------------------------------------------------------#
@dataclass
class Controller:
    dtype: torch.dtype
    device: str | torch.device
    reject: bool = False
    first_step: bool = True
    errold: torch.Tensor = field(init=False)
    hold: torch.Tensor = field(init=False)
    hnext: torch.Tensor = field(init=False)

    def __post_init__(self):
        self.errold = _t(1.0, dtype=self.dtype, device=self.device)
        self.hold = _t(0.0, dtype=self.dtype, device=self.device)
        self.hnext = _t(0.0, dtype=self.dtype, device=self.device)

    # return (accepted?, new_h)
    def success(self, err: torch.Tensor, h: torch.Tensor) -> Tuple[bool, torch.Tensor]:
        safe, fac1, fac2 = 0.9, 5.0, 1.0 / 6.0
        fac = torch.clamp(err.pow(0.25) / safe, fac2, fac1)
        hnew = h / fac
        if err <= 1.0:  # accept
            if not self.first_step:
                facpred = (self.hold / h) * (err * err / self.errold).pow(0.25) / safe
                facpred = torch.clamp(facpred, fac2, fac1)
                hnew = h / torch.maximum(fac, facpred)
            self.first_step = False
            self.hold = h
            self.errold = torch.maximum(_t(0.01, dtype=err.dtype, device=err.device), err)
            if self.reject:
                hnew = torch.minimum(hnew, h) if h >= 0 else torch.maximum(hnew, h)
            self.hnext = hnew
            self.reject = False
            return True, h
        # reject
        self.reject = True
        return False, hnew


# -----------------------------------------------------------------------------#
# main solver class                                                            #
# -----------------------------------------------------------------------------#
@dataclass
class RosenbrockSolver:
    rhs: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    y0: torch.Tensor
    t0: float = 0.0
    jac: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    atol: float = 1e-6
    rtol: float = 1e-3
    coeffs: RosenbrockCoeffs = field(default_factory=RosenbrockCoeffs.default)

    # internal state (initialised in __post_init__)
    t: torch.Tensor = field(init=False)
    y: torch.Tensor = field(init=False)
    dydt: torch.Tensor = field(init=False)
    ctrl: Controller = field(init=False)

    # workspaces allocated lazily
    _allocated: bool = field(init=False, default=False)

    def __post_init__(self):
        if self.jac is None:
            self.jac = self._autograd_jac  # convenience (slow!) fallback
        self.t = _t(self.t0, dtype=self.y0.dtype, device=self.y0.device)
        self.y = self.y0.clone()
        self.dydt = self.rhs(self.t, self.y)
        self.ctrl = Controller(dtype=self.y.dtype, device=self.y.device)

    # ---------------- private helpers ---------------- #
    def _alloc(self):
        n, d, dev = self.y.numel(), self.y.dtype, self.y.device
        Z = lambda: torch.empty(n, dtype=d, device=dev)
        self.dfdy = torch.empty((n, n), dtype=d, device=dev)
        self.k1, self.k2, self.k3, self.k4, self.k5 = Z(), Z(), Z(), Z(), Z()
        self.yerr, self.yout = Z(), Z()
        self._allocated = True

    def _autograd_jac(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # warning: quadratic in n, slow for large systems
        y = y.detach().requires_grad_(True)
        return torch.autograd.functional.jacobian(lambda yy: self.rhs(t, yy), y)

    # ---------------- stepping interface ------------- #
    def step(self, h_try: float | torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self._allocated:
            self._alloc()

        h = _t(float(h_try), dtype=self.y.dtype, device=self.y.device)
        self.dfdy[:] = self.jac(self.t, self.y)

        while True:
            self._rosenbrock_step(h)
            err = self._error()
            ok, h_new = self.ctrl.success(err, h)
            if ok:
                break
            h = h_new
            if torch.abs(h) <= torch.abs(self.t) * torch.finfo(self.y.dtype).eps:
                raise RuntimeError("Step size underflow")

        self.dydt = self.rhs(self.t + h, self.yout)
        self.t = self.t + h
        self.y[:] = self.yout
        return self.y, self.t, self.ctrl.hnext

    def solve(self, t_end: float, h0: float) -> List[Tuple[float, torch.Tensor]]:
        """
        Integrate until *t_end*, returning a list of (t, y) snapshots.
        """
        h = h0
        out = [(float(self.t), self.y.clone())]
        while float(self.t) < t_end:
            if float(self.t + h) > t_end:
                h = t_end - float(self.t)
            y, t, h_next = self.step(h)
            out.append((float(t), y.clone()))
            h = float(h_next)
        return out

    # ---------------- internals: single ROS4 step ---- #
    def _rosenbrock_step(self, h: torch.Tensor):
        c = self.coeffs
        a_mat = -self.dfdy.clone()
        a_mat.diagonal().add_(1.0 / (c.gam * h))
        solve = lambda rhs: torch.linalg.solve(a_mat, rhs)

        # stage 1
        self.k1[:] = solve(self.dydt)  # dfdx = 0 for autonomous systems

        # stage 2
        y2 = self.y + c.a21 * self.k1
        f2 = self.rhs(self.t + c.c2 * h, y2)
        self.k2[:] = solve(f2 + c.c21 * self.k1 / h)

        # stage 3
        y3 = self.y + c.a31 * self.k1 + c.a32 * self.k2
        f3 = self.rhs(self.t + c.c3 * h, y3)
        self.k3[:] = solve(f3 + (c.c31 * self.k1 + c.c32 * self.k2) / h)

        # stage 4
        y4 = self.y + c.a41 * self.k1 + c.a42 * self.k2 + c.a43 * self.k3
        f4 = self.rhs(self.t + c.c4 * h, y4)
        self.k4[:] = solve(f4 + (c.c41 * self.k1 + c.c42 * self.k2 + c.c43 * self.k3) / h)

        # stage 5 + error
        y5 = (
            self.y
            + c.a51 * self.k1
            + c.a52 * self.k2
            + c.a53 * self.k3
            + c.a54 * self.k4
        )
        f5 = self.rhs(self.t + h, y5)
        self.k5[:] = solve(f5 + (c.c51 * self.k1 + c.c52 * self.k2 + c.c53 * self.k3 + c.c54 * self.k4) / h)

        y_temp = y5 + self.k5
        f6 = self.rhs(self.t + h, y_temp)
        rhs6 = (
            f6
            + (
                c.c61 * self.k1
                + c.c62 * self.k2
                + c.c63 * self.k3
                + c.c64 * self.k4
                + c.c65 * self.k5
            )
            / h
        )
        self.yerr[:] = solve(rhs6)
        self.yout[:] = y_temp + self.yerr

    # ---------------- local error measure ----------- #
    def _error(self) -> torch.Tensor:
        sk = self.atol + self.rtol * torch.maximum(torch.abs(self.y), torch.abs(self.yout))
        return torch.sqrt(torch.mean((self.yerr / sk) ** 2))


# -----------------------------------------------------------------------------#
# demo: Van-der-Pol oscillator                                                 #
# -----------------------------------------------------------------------------#
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import time
    from textwrap import dedent
    from scipy.integrate import solve_ivp

    torch.set_default_dtype(torch.float64)
    device = "cpu"               # "cuda" if you moved tensors to GPU

    mu = 1.0                     # try 10 or 100 for stiffness timing

    # RHS and Jacobian -------------------------------------------------
    def vdp(t, y):
        x, xd = y
        return [xd, mu * (1 - x**2) * xd - x]

    def vdp_jac(t, y):
        x, xd = y
        return [[0.0, 1.0],
                [-2.0 * mu * x * xd - 1.0, mu * (1.0 - x**2)]]

    # ------------- our Rosenbrock (ROS4) ------------------------------
    y0_torch = torch.tensor([2.0, 0.0], device=device)

    ros_solver = RosenbrockSolver(
        rhs=lambda t, y: torch.tensor(vdp(t, y.tolist()),
                                      dtype=y.dtype, device=y.device),
        y0=y0_torch,
        t0=0.0,
        jac=lambda t, y: torch.tensor(vdp_jac(t, y.tolist()),
                                      dtype=y.dtype, device=y.device),
        atol=1e-7, rtol=1e-6,
    )

    t0 = time.perf_counter()
    ros_traj = ros_solver.solve(t_end=20.0, h0=0.1)
    ros_secs = time.perf_counter() - t0

    ros_ts  = [t for t, _ in ros_traj]
    ros_xs  = [float(y[0]) for _, y in ros_traj]
    ros_xds = [float(y[1]) for _, y in ros_traj]

    # ---------------- SciPy Radau -------------------------------------
    t0 = time.perf_counter()
    sol = solve_ivp(
        fun=vdp,
        t_span=(0.0, 20.0),
        y0=[2.0, 0.0],
        method="Radau",
        jac=vdp_jac,
        atol=1e-7, rtol=1e-6,
    )
    rad_secs = time.perf_counter() - t0

    rad_ts  = sol.t
    rad_xs  = sol.y[0]
    rad_xds = sol.y[1]

    # --------------- summary ------------------------------------------
    print(dedent(f"""
        ── timing summary (μ = {mu}) ─────────────────────────────────
        ROS4  : {ros_secs:.4f} s  | steps = {len(ros_traj)-1}
        SciPy Radau: {rad_secs:.4f} s  | steps = {len(rad_ts)-1}
    """))

    # --------------- plot --------------------------------------------
    #plt.plot(ros_ts, ros_xs,  label="ROS4  x(t)",  lw=1.8)
    #plt.plot(rad_ts, rad_xs,  label="Radau x(t)", lw=1.2, alpha=0.6)
    #plt.xlabel("t"); plt.legend(); plt.title("Van der Pol – timing comparison")
    #plt.savefig("rosenbrock_vdp.png", dpi=300)
