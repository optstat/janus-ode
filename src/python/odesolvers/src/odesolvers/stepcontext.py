# context.py ---------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Callable, Optional   # top of the file

import torch, math

@dataclass
class StepContext:
    # user-visible state
    y:       torch.Tensor
    t:       float
    h:       float
    tfinal:  float

    # solver options
    ode_f:   callable
    jac_f:   Optional[Callable]  
    rtol:    float
    atol:    float
    max_it:  int
    s_set:   tuple[int,...] = (3,)          # allowed stage counts

    # live Radau tables (filled in prepare_step)
    T: torch.Tensor   | None = None
    TI:torch.Tensor   | None = None
    C: torch.Tensor   | None = None
    ValP:torch.Tensor | None = None
    Dd: torch.Tensor  | None = None

    # disposable work
    solve_fns:  list = field(default_factory=list)
    J:          torch.Tensor | None = None
    stats:      dict = field(default_factory=lambda: dict(
                       step=0, rej=0, newt=0, fact=0, fcall=0))

    # flags between calls
    need_jac:   bool = True
    need_fact:  bool = True
    newt_slow:  bool = False         # last Newton rejected
