# event_zero.py
import math, torch
from dataclasses import dataclass
from typing import Callable, Sequence, Optional
from .interp_radau import radau_interpolate


@dataclass
class EventState:
    """Stores data between successive calls."""
    t_prev: Optional[float]          = None      # t₁
    g_prev: Optional[torch.Tensor]   = None      # g(t₁)
    slope:  Optional[torch.Tensor]   = None      # sign of g˙ at t₁


# --------------------------------------------------------------------------- #
def check_events(
    evfcn:   Callable,                # user-supplied g(t), g(t,y), or g(t,y,*)
    t:       float,                   # step start
    h:       float,                   # step size
    C:       torch.Tensor,            # stage abscissae (s,)
    y:       torch.Tensor,            # y(t)     (n,)
    cont:    torch.Tensor,            # continuation polys (n,s)
    f0:      torch.Tensor,            # f(t,y)   (n,)
    state:   EventState,              # persistent store
    *evargs                        ) -> tuple[list[float],
                                              list[torch.Tensor],
                                              list[int],
                                              list[bool]]:
    """
    Detect the *first* zero of each event component over [t, t+h].

    Returns
    -------
    tout   : list of event times   (may be empty)
    yout   : matching y(tout[k])
    iout   : component index (1-based like MATLAB)
    term   : True if event has terminal flag (Stop in original code)
    """

    # helper to evaluate g with flexible signature
    def g_fun(t_eval: float, y_eval: torch.Tensor | None = None):
        sig = evfcn.__code__.co_argcount
        if sig == 1:
            return torch.as_tensor(evfcn(t_eval), dtype=y.dtype, device=y.device)
        elif sig == 2:
            return torch.as_tensor(evfcn(t_eval, y_eval), dtype=y.dtype, device=y.device)
        else:
            return torch.as_tensor(evfcn(t_eval, y_eval, *evargs),
                                   dtype=y.dtype, device=y.device)

    # ---------- initialisation (first call) --------------------------------
    if state.t_prev is None:
        g1 = g_fun(t, y)
        slope = torch.sign(f0)                         # crude slope guess
        hit_now = (g1 == 0) & (torch.sign(f0) == slope)
        if hit_now.any():                              # event exactly at t
            idx = hit_now.nonzero(as_tuple=False)[:, 0] + 1
            return [t], [y], idx.tolist(), [False]*len(idx)
        state.t_prev, state.g_prev, state.slope = t, g1, slope
        return [], [], [], []                         # nothing yet

    # ---------- evaluate at t₂ = t + h -------------------------------------
    t2   = t + h
    g2   = g_fun(t2, y)

    tout, yout, iout, term = [], [], [], []
    g1, slope = state.g_prev, state.slope
    t1        = state.t_prev

    # parameters for Pegasus
    EAbsTol = 1e-9
    t_tol   = max(65536 * torch.finfo(y.dtype).eps * max(abs(t1), abs(t2)),
                  1e-12 * abs(h))

    # loop over event components
    for k in range(g1.numel()):
        if g1[k] * g2[k] >= 0:            # no sign change
            continue
        if torch.sign(g2[k] - g1[k]) * slope[k] < 0:   # wrong direction
            continue

        # --- Pegasus iteration --------------------------------------------
        a, fa = t1, g1[k].item()
        b, fb = t2, g2[k].item()
        c, fc = b, fb
        for _ in range(50):               # max iterations
            # secant step
            c_prev = c
            c = (a*fb - b*fa) / (fb - fa)
            # interpolate state and evaluate g
            y_c = radau_interpolate([c], t, y, h, C, cont)[:, 0]
            fc = g_fun(c, y_c)[k].item()

            # convergence?
            if (abs(fc) < EAbsTol and
                abs(b - a) < t_tol + t_tol*max(abs(a), abs(b))):
                break

            # pegasus update
            if fb * fc < 0:
                a, fa = b, fb
            else:
                fa *= fb / (fb + fc)
            b, fb = c, fc

        tout.append(c)
        yout.append(y_c)
        iout.append(k + 1)          # MATLAB 1-based indexing
        term.append(False if slope is None else False)

    # ---------- update persistent state ------------------------------------
    state.t_prev, state.g_prev = t2, g2

    return tout, yout, iout, term
