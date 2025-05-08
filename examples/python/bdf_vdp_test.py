# driver_vdp_compare.py
"""Compare the custom PyTorch BDF solver (bdf_torch.py) with SciPy's Radau
on the Van‑der‑Pol oscillator.

Usage
-----
Save *bdf_torch.py* and this driver in the same directory, then run

    python driver_vdp_compare.py

Two plots will appear: the trajectories x(t), y(t) and (if SciPy is
available) the difference BDF − Radau.  Runtime statistics print to the
console.
"""
from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

# ------------------------------------------------------------------
# 1.  Import local PyTorch BDF solver
# ------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))  # Ensure same‑dir import wins
try:
    from bdf_torch import BDFTorch
except ImportError as exc:
    raise SystemExit("bdf_torch.py not found next to this driver.") from exc

# ------------------------------------------------------------------
# 2.  Van‑der‑Pol definitions (torch + numpy)
# ------------------------------------------------------------------
MU = 5.0

def vdp_torch(t: float, y: torch.Tensor) -> torch.Tensor:
    return torch.stack([y[1], MU * (1 - y[0] ** 2) * y[1] - y[0]])

def jac_vdp_torch(t: float, y: torch.Tensor) -> torch.Tensor:
    return torch.tensor([
        [0.0, 1.0],
        [-2 * MU * y[0] * y[1] - 1.0, MU * (1 - y[0] ** 2)],
    ], dtype=y.dtype, device=y.device)

# NumPy/SciPy versions ---------------------------------------------

def vdp_np(t, y):
    return np.array([y[1], MU * (1 - y[0] ** 2) * y[1] - y[0]])

def jac_vdp_np(t, y):
    return np.array([
        [0.0, 1.0],
        [-2 * MU * y[0] * y[1] - 1.0, MU * (1 - y[0] ** 2)],
    ])

# ------------------------------------------------------------------
# 3.  Integrate with BDFTorch
# ------------------------------------------------------------------
print("→ Running BDFTorch …", flush=True)
y0 = torch.tensor([2.0, 0.0])
solver = BDFTorch(vdp_torch, 0.0, y0, 20.0, jac=jac_vdp_torch,first_step=0.1)

bdf_t, bdf_y = [], []
start = time.perf_counter()
while solver.step():
    bdf_t.append(solver.t)
    bdf_y.append(solver.y.detach().cpu().numpy())
    if solver.t >= 20.0:
        break
bdf_sec = time.perf_counter() - start
bdf_y = np.stack(bdf_y)
print(f"   BDFTorch finished in {bdf_sec:.3f} s with {len(bdf_t)} accepted steps.")

# ------------------------------------------------------------------
# 4.  Integrate with SciPy Radau (optional)
# ------------------------------------------------------------------
try:
    from scipy.integrate import solve_ivp
    from scipy.interpolate import interp1d
    have_scipy = True
except ImportError:
    have_scipy = False
    print("SciPy not installed – skipping Radau comparison.")

if have_scipy:
    print("→ Running SciPy Radau …", flush=True)
    start = time.perf_counter()
    sol = solve_ivp(vdp_np, (0.0, 20.0), np.array([2.0, 0.0]),
                    method="Radau", jac=jac_vdp_np,
                    rtol=1.0e-6, atol=1.0e-9)
    rad_sec = time.perf_counter() - start
    print(f"   Radau finished in {rad_sec:.3f} s with {len(sol.t)} steps.")

# ------------------------------------------------------------------
# 5.  Plot results
# ------------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(bdf_t, bdf_y[:, 0], label="BDFTorch  x(t)")
plt.plot(bdf_t, bdf_y[:, 1], label="BDFTorch  y(t)")
if have_scipy:
    plt.plot(sol.t, sol.y[0], "--", label="Radau  x(t)")
    plt.plot(sol.t, sol.y[1], "--", label="Radau  y(t)")
plt.title("Van‑der‑Pol oscillator (μ = 5)")
plt.xlabel("t")
plt.legend()
plt.tight_layout()

if have_scipy:
    fx = interp1d(sol.t, sol.y[0], kind="cubic")
    fy = interp1d(sol.t, sol.y[1], kind="cubic")
    diff_x = bdf_y[:, 0] - fx(bdf_t)
    diff_y = bdf_y[:, 1] - fy(bdf_t)
    plt.figure(figsize=(8, 3))
    plt.plot(bdf_t, diff_x, label="Δx (BDF–Radau)")
    plt.plot(bdf_t, diff_y, label="Δy (BDF–Radau)")
    plt.title("Solution difference")
    plt.xlabel("t")
    plt.legend()
    plt.tight_layout()

plt.show()
