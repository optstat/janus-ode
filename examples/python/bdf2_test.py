#!/usr/bin/env python3
import numpy as np
from bdf2 import bdf2_integrate          # solver only
from scipy.integrate import solve_ivp

# stiff Van-der-Pol with μ = 1000
MU = 1.0
def vdp(t, y):
    return np.array([y[1],
                     MU * ((1.0 - y[0]**2) * y[1] - y[0])])
tf=1.0
# integrate with homemade BDF-2
t, y_bdf = bdf2_integrate(vdp, (0.0, tf),
                          np.array([2.0, 0.0]),
                          n=100_000)

# high-accuracy reference with SciPy-Radau
sol = solve_ivp(vdp, (0.0, tf), [2.0, 0.0],
                method="Radau", rtol=1e-10, atol=1e-12)

err = np.linalg.norm(y_bdf[-1] - sol.y[:, -1], ord=np.inf)
print("‖error‖_inf =", err)
