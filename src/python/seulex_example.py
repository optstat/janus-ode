import torch
import math
from LU import LU
from seulex import Seulex
device = 'cpu'


# Define the dynamics class for the Van der Pol oscillator
mu=1.0
factor = 10.0
class VanDerPolDynamics:
    def jacobian(self, t, y):
        dfdy = torch.zeros((2, 2), dtype=y.dtype, device=y.device)
        dfdy[0, 0] = 0.0
        dfdy[0, 1] = 1.0*factor
        dfdy[1, 0] = (-2.0 * mu * y[0] * y[1] - 1.0)*factor
        dfdy[1, 1] = mu * (1 - y[0] ** 2)*factor
        dfdx = torch.zeros(2, dtype=y.dtype)
        return dfdx, dfdy

    def __call__(self, t, y):
        dydt = torch.zeros_like(y, dtype=y.dtype, device=y.device)
        dydt[0] = y[1]*factor
        dydt[1] = (mu * (1 - y[0] ** 2) * y[1] - y[0])*factor
        return dydt


#create a class of dynamics for the  bouncing ball
class BouncingBallDynamics:
    def jacobian(self, t, y):
        D = y.shape
        dfdy = torch.zeros((D, D), dtype=y.dtype, device=y.device)

        dfdy[0, 0] = 0.0
        dfdy[0, 1] = 1.0
        if y[0] <= 0.0:
            dfdy[0, 1] = -1.0

        dfdx = torch.zeros(2, dtype=y.dtype, device=y.device)
        return dfdx, dfdy

    def __call__(self, t, y):
        dydt = torch.zeros_like(y, dtype=y.dtype, device=y.device)
        dydt[1] = -9.8
        dydt[0] = y[1]
        if y[0] <= 0.0:
            dydt[0] = torch.abs(y[1])
        return dydt

# Set up the initial condition and solver parameters
x0 = torch.tensor([0.0], dtype=torch.float64, device=device)  # Starting point
y0 = torch.tensor([2.0, 0.0], dtype=torch.float64, device=device)  # Starting dynamics
atol = 1.0e-16 # Absolute tolerance
rtol = 1.0e-16# Relative tolerance

# Create an instance of the Seulex solver
solver = Seulex(y0, x0, atol, rtol)

# Create an instance of the Van der Pol dynamics
dynamics = VanDerPolDynamics()

# Perform a single step using the solver
ft = torch.ones([1,], dtype=torch.float64, device=device)*10.0
htry =torch.ones([1,], dtype=torch.float64, device=device)*0.01
ts = [0.0]
x1s=[y0[0].cpu()]
x2s=[y0[1].cpu()]
solver.step(htry, dynamics)
print("Solution after first step:")
print("Time: ", solver.x)
print("Solution: ", solver.y)

print(f"Next step {solver.hnext}")
print(f'Step actually taken {solver.hdid}')
assert solver.hnext > 0.0
x1s.append(solver.y[0].cpu())
x2s.append(solver.y[1].cpu())
ts.append(solver.hdid.cpu())
count = 1
while (ft-solver.x) > 0.0:
    if solver.hnext > ft - solver.x:
        htry = ft - solver.x
    else:
        htry = solver.hnext
    if htry <= torch.abs(solver.x)*solver.EPS:
        htry = solver.hdid
    solver.step(htry, dynamics)
    ts.append(ts[-1]+solver.hdid.cpu().item())
    x1s.append(solver.y[0].cpu())
    x2s.append(solver.y[1].cpu())

    # Print the result
    print("Solution after a single step:")
    print("Time: ", solver.x)
    print("Solution: ", solver.y)
    print(f"Next step {solver.hnext}")
    print(f'Step actually taken {solver.hdid}')
    count += 1


print(f'Number of steps {count}')

from matplotlib import pyplot as plt
#Do a scatter plot with a small circle
plt.scatter(x1s, x2s)
plt.savefig("../images/vanderpol.png")

