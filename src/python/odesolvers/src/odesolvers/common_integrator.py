# common_integrator.py  (formerly radau_integrator.py)
import math, torch
from typing import Callable, List, Optional
from dataclasses import dataclass, field
from .prepare_step         import prepare_step      # file is prepare.py
from .bdf2_step       import bdf2_step
from .stepcontext         import StepContext
from .estrad          import estrad
from .interp_radau    import radau_interpolate
from .event_zero      import check_events, EventState


# ---------------------------------------------------------------------------

@dataclass
class OutputBuffer:
    t: List[float]            = field(default_factory=list)
    y: List[torch.Tensor]     = field(default_factory=list)


class CommonIntegrator:
    """Adaptive stiff integrator with pluggable step kernels
       (currently Radau-IIA and BDF-2)."""

    # ---- construction -----------------------------------------------------
    def __init__(self, f: Callable, t0: float, y0: torch.Tensor,
                 tfinal: float, *, mode="radau",
                 rtol=1e-6, atol=1e-9,
                 mass: Optional[torch.Tensor] = None,
                 events: Optional[Callable]  = None,
                 output: Optional[Callable]  = None,
                 stages=3, gmres=True, h_init=1e-3,
                 h_min=1e-12, h_max=1.0, max_steps=100000):

        self.mass  = torch.eye(len(y0)) if mass is None else mass
        self.ctx   = StepContext(y=y0.clone(), t=t0, h=h_init,
                                 tfinal=tfinal,
                                 ode_f=f, jac_f=None,
                                 rtol=rtol, atol=atol,
                                 max_it=max_steps)
        self.ctx.y_nm1 = y0.clone()

        self.mode      = mode.lower()            # "radau" | "bdf2"
        self.stages    = stages                  # ignored for bdf2
        self.gmres     = gmres
        self.h_min     = h_min
        self.h_max     = h_max
        self.events    = events
        self.output    = output

        self.event_state = EventState()
        self.outbuf      = OutputBuffer()


        # initial output
        if output is None:
            self.outbuf.t.append(t0)
            self.outbuf.y.append(y0.clone())

    # ---- step-size controller --------------------------------------------
    def adapt_step_size(self, err, newton_iter, Nit, order,
                        Safe=0.9, FacL=5.0, FacR=0.2):
        fac  = min(Safe, (2*Nit+1)/(2*Nit+newton_iter))
        quot = max(FacR, min(FacL, err**(1/order)/fac))
        hnew = self.ctx.h / quot
        return max(self.h_min, min(self.h_max, hnew))

    # ---- event & output helpers ------------------------------------------
    def handle_events(self, cont):
        if self.events is None:
            return False
        tout, yout, _, term = check_events(
            self.events,
            self.ctx.t - self.ctx.h, self.ctx.h,
            self.ctx.C if cont is not None else None,
            self.ctx.y, cont, torch.zeros_like(self.ctx.y),
            self.event_state)
        for t_i, y_i in zip(tout, yout):
            if self.output is None:
                self.outbuf.t.append(t_i)
                self.outbuf.y.append(y_i.clone())
            else:
                self.output(t_i, y_i)
        return any(term)

    def record_output(self):
        if self.output is not None:
            self.output(self.ctx.t, self.ctx.y)
        else:
            self.outbuf.t.append(self.ctx.t)
            self.outbuf.y.append(self.ctx.y.clone())

    # ---- main loop --------------------------------------------------------
    def _outer_pass(self):
        """Do *one* accept/reject cycle; return True if step accepted."""
        kernel = self.mode            # alias

        ctx = self.ctx
        last = False
        if (ctx.t + ctx.h - ctx.tfinal) * math.copysign(1, ctx.h) > 0:
            ctx.h = ctx.tfinal - ctx.t
            last  = True

        order       = 2 if kernel == "bdf2" else self.stages + 1
        prep_stages = 0 if kernel == "bdf2" else self.stages
        # --- factorisation / table load --------------------------------------
        if kernel != "bdf2":                   # Radau path → γ already known
            prepare_step(ctx, stages=prep_stages,
                 mass=self.mass.to(ctx.y.dtype),
                 gmres=self.gmres)


        # Newton solve -----------------------------------------------------
        if kernel == "radau":
            newt = simplified_newton(
                ctx.y, ctx.t, ctx.h, self.mass,
                OdeFcn=ctx.ode_f, ode_args=(),
                T=ctx.T, TI=ctx.TI, C=ctx.C, ValP=ctx.ValP,
                solve_fns=ctx.solve_fns,
                RelTol1=ctx.rtol * torch.ones_like(ctx.y))
        else:                                   # BDF-2
            newt = bdf2_step(ctx, self.mass, gmres=self.gmres)
            
        if newt.converged and newt.stats.SolveNbr > 1 and newt.stats.SolveNbr % 4 == 0:
            ctx.need_jac = True        # refresh occasionally for very stiff ODEs

        if newt.rejected:
            ctx.h = newt.h_new
            ctx.need_fact = newt.need_new_lu
            return False                        # retry

        # error estimate ---------------------------------------------------
        if kernel == "bdf2":
            y_next = ctx.y + newt.z[:, 0]
            lte    = y_next - 2*ctx.y + ctx.y_nm1        # numerator
            errvec = (lte / 12)   
            scale  = ctx.atol + ctx.rtol * torch.abs(y_next)
            err    = torch.linalg.norm(errvec / scale, ord=torch.inf)
        else:
            f0  = ctx.ode_f(ctx.t, ctx.y)
            err, _ = estrad(newt.z, ctx.Dd, ctx.h, ctx.solve_fns[0],
                            self.mass, ctx.rtol*torch.ones_like(ctx.y),
                            f0, First=True, Reject=False,
                            Stat=None, OdeFcn=ctx.ode_f,
                            t=ctx.t, y=ctx.y)

        if err > 1.0:
            ctx.h = self.adapt_step_size(err, newt.converged, Nit=7,
                                         order=order)
            return False                      # step rejected

        # step accepted ----------------------------------------------------
        old_y = ctx.y.clone()
        ctx.y += newt.z[:, 0]
        if kernel == "bdf2":
            ctx.y_nm1 = old_y               # slide history

        ctx.t += ctx.h
        ctx.h  = self.adapt_step_size(max(err, 1e-12), newt.converged,
                                      Nit=7, order=order)

        # dense output / events
        cont = None if kernel == "bdf2" else newt.z.flip(1).clone()
        if self.handle_events(cont):
            ctx.tfinal = ctx.t              # terminal event

        self.record_output()
        return True                         # step accepted

    def run(self):
        while self.ctx.stats["step"] < self.ctx.max_it and self.ctx.t < self.ctx.tfinal:
            accepted = self._outer_pass()
            if accepted:
                self.ctx.stats["step"] += 1
        return torch.tensor(self.outbuf.t), torch.stack(self.outbuf.y)

    # ---------------------------------------------------------------------
    def step_once(self):
        """Advance exactly one accepted step and return (t, y)."""
        steps_before = self.ctx.stats["step"]
        while self.ctx.stats["step"] == steps_before:
            self._outer_pass()
        return self.ctx.t, self.ctx.y.clone()