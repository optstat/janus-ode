#!/usr/bin/env python3
# examples/vdp_driver.py
"""
Sanity-check the CommonIntegrator on the stiff Van-der-Pol oscillator.
You can switch methods with  --method {bdf2,radau}
"""

import argparse, time, torch, sys
from odesolvers.common_integrator import CommonIntegrator


# --------------------------------------------------------------------------
def vdp_rhs(t: float, y: torch.Tensor, mu: float = 1000.0) -> torch.Tensor:
    """Stiff Van-der-Pol RHS (μ large)."""
    return torch.tensor([y[1],
                         mu * ((1.0 - y[0]**2) * y[1] - y[0])],
                        dtype=y.dtype)

# ---------------------------------------------------------------------------
# quick-and-dirty step-size sweep for BDF-2
# ---------------------------------------------------------------------------

def quick_bdf2_sweep():
    import torch
    from odesolvers.common_integrator import CommonIntegrator

    torch.set_default_dtype(torch.float64)

    def vdp(t, y, mu=1000.0):
        return torch.tensor([y[1],
                             mu * ((1.0 - y[0]**2) * y[1] - y[0])],
                            dtype=y.dtype)

    y0 = torch.tensor([2.0, 0.0])
    t0, tf = 0.0, 0.01

    for h0 in (1e-4, 5e-5, 1e-5, 5e-6):
        solver = CommonIntegrator(vdp, t0, y0, tf,
                                  mode="bdf2",
                                  h_init=h0,
                                  gmres=False)        # or True
        t, y = solver.run()
        print(f"h0 = {h0:8.1e}   "
              f"steps = {solver.ctx.stats['step']:4}   "
              f"y(tf)[1] = {y[-1,1]:.5e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("bdf2", "radau"),
                        default="bdf2", help="integration backend")
    parser.add_argument("--mu", type=float, default=1000.0,
                        help="stiffness parameter μ")
    parser.add_argument("--dt0", type=float, default=1e-4,
                        help="initial step size")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)

    t0, tf = 0.0, 0.01
    y0     = torch.tensor([2.0, 0.0], dtype=torch.float64)

    print(f"\nMethod : {args.method.upper()}")
    print(f"μ       : {args.mu}")
    print(f"t span  : [{t0}, {tf}]")

    # build integrator -----------------------------------------------------
    solver = CommonIntegrator(lambda t, y: vdp_rhs(t, y, args.mu),
                              t0, y0, tf,
                              mode=args.method,
                              stages=3,           # ignored when mode=bdf2
                              gmres=False,        # dense LU for this demo
                              h_init=args.dt0)

    t_start = time.perf_counter()
    tout, yout = solver.run()
    t_end   = time.perf_counter()

    print("\n--- results -------------------------------------------------")
    print(f"accepted steps : {solver.ctx.stats['step']}")
    print(f"wall time      : {1e3*(t_end - t_start):.1f} ms")
    print(f"y(tf)          : {yout[-1].tolist()}")

    # very quick self-consistency check
    if args.method == "radau":
        # crude reference: compare to SciPy's Radau (or another run)
        pass


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if "--sweep" in sys.argv:           # call:  python vdp_driver.py --sweep
        sys.argv.remove("--sweep")      # remove flag so argparse won't choke
        quick_bdf2_sweep()
    else:                               # normal CLI: --method, --mu, --dt0 ...
        main()    