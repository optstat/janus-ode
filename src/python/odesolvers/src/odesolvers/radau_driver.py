# radau_driver.py ----------------------------------------------------------
from python.odesolvers.stepcontext import StepContext
from python.prepare_step  import prepare_step
from .simplified_newton import simplified_newton
from .estrad   import estrad
import torch

def radau_step(ctx: StepContext, mass: torch.Tensor,
               stages: int = 3, gmres: bool = True):

    # 1. lightweight prep
    prepare_step(ctx, stages=stages, mass=mass, gmres=gmres)

    # 2. simplified Newton
    newt = simplified_newton(
        ctx.y, ctx.t, ctx.h, mass,
        OdeFcn   = ctx.ode_f,
        ode_args = (),
        T=ctx.T, TI=ctx.TI, C=ctx.C, ValP=ctx.ValP,
        solve_fns=ctx.solve_fns,
        RelTol1  = ctx.rtol * torch.ones_like(ctx.y),
        stats    = None)

    ctx.stats["newt"] += 1

    if newt.rejected:                       # Newton failed
        ctx.h   = newt.h_new
        ctx.need_fact = newt.need_new_lu
        return False                        # retry

    # 3. local error
    f0   = ctx.ode_f(ctx.t, ctx.y)
    err, _ = estrad(newt.z, ctx.Dd, ctx.h, ctx.solve_fns[0],
                    mass, ctx.rtol*torch.ones_like(ctx.y),
                    f0, First=True, Reject=False,
                    Stat=None, OdeFcn=ctx.ode_f,
                    t=ctx.t, y=ctx.y)

    if err > 1.0:                           # step rejected by error test
        ctx.h   *= max(0.1, 0.8*err**(-1/4))
        ctx.need_fact = False               # Jacobian unchanged
        ctx.stats["rej"] += 1
        return False

    # 4. step accepted
    ctx.y   = ctx.y + newt.z[:, 0]
    ctx.t  += ctx.h
    ctx.h  *= min(5.0, 0.8*err**(-1/4))
    ctx.stats["step"] += 1
    return True
