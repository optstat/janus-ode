# interp_radau.py
import torch
from typing import Sequence


def radau_interpolate(
    t_interp: Sequence[float] | torch.Tensor,
    t:        float,
    y:        torch.Tensor,          # shape (n,)
    h:        float,
    C:        torch.Tensor,          # stage abscissae (s,)
    cont:     torch.Tensor           # continuation polys  (n, s)
) -> torch.Tensor:
    """
    Dense output for Radau-IIA (matches MATLAB ntrprad).

    Parameters
    ----------
    t_interp : 1-D array of target times      (len = m)
    t        : current step start time
    y        : y(t)  (state at start)         (n,)
    h        : step size
    C        : stage abscissae                (s,)
    cont     : continuation coefficients      (n, s)
               (= 'cont' array produced inside the stepper)

    Returns
    -------
    y_interp : y evaluated at each t_interp   (n, m)
    """
    # --- normalise target times to s = (τ − 1) -----------------------------
    τ   = torch.as_tensor(t_interp, dtype=y.dtype, device=y.device)
    s   = (τ - t) / h                                   # shape (m,)
    Cm  = C - 1.0                                       # (s,)

    # --- Horner scheme over all targets at once ----------------------------
    #   yi = cont[:, -1] * (s - Cm[-1]) +
    #        cont[:, -2] * Π_{k=-2}^{-1} (s - Cm[k]) + ...
    n, S = cont.shape
    m    = τ.numel()

    yi = (s - Cm[-1]).unsqueeze(0) * cont[:, -1].unsqueeze(1)   # (n, m)
    for k in range(S - 2, -1, -1):
        yi = (s - Cm[k]).unsqueeze(0) * (yi + cont[:, k].unsqueeze(1))

    return yi + y.unsqueeze(1)                                # (n, m)
