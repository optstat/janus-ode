import torch
from seulext import SeulexT
from seulext import SeulexState
device = 'cpu'
import math

# Define the dynamics class for the Van der Pol oscillator
mu=1000.0
M=1
factor = 10.0
torch.set_num_threads(torch.get_num_threads())
class VanDerPolDynamicsBatch:
    def jacobian(self, t, y):
        M, D = y.shape
        dfdy = torch.zeros((M, D, D), dtype=y.dtype, device=y.device)
        dfdy[:, 0, 0] = 0.0
        dfdy[:, 0, 1] = 1.0*factor
        dfdy[:, 1, 0] = (-2.0 * mu * y[:, 0] * y[:, 1] - 1.0)*factor
        dfdy[:, 1, 1] = mu * (1 - y[:, 0] ** 2)*factor
        return dfdy

    def __call__(self, t, y):
        dydt = torch.zeros_like(y, dtype=y.dtype, device=y.device)
        dydt[:, 0] = y[:, 1]*factor
        dydt[:, 1] = (mu * (1 - y[:, 0] ** 2) * y[:, 1] - y[:, 0])*factor
        return dydt




x0 = torch.zeros([M,], dtype=torch.float64, device=device)  # Starting point
y0 = torch.ones([M,2], dtype=torch.float64, device=device)#+torch.rand([M,2], dtype=torch.float64, device=device)
#y0[0,:]= torch.tensor([0.5, 0.5], dtype=torch.float64)# Starting dynamics
atol = 1.0e-6# Absolute tolerance
rtol = 1.0e-3# Relative tolerance

KMAXX = 13
state = SeulexState(y0, x0, atol, rtol, KMAXX)
solver = SeulexT()

# Create an instance of the Van der Pol dynamics
dynamics = VanDerPolDynamicsBatch()
#Time the solver
import time
start = time.time()
# Perform a single step using the solver
ft = torch.ones([M,], dtype=torch.float64, device=device)*90.3191893005205060

htry = torch.ones([M,], dtype=torch.float64, device=device)*0.01
x1s  = torch.zeros((M, 1),  device=device)
x2s  = torch.zeros((M, 1), device=device)
x1s[:,0] = y0[:, 0]
x2s[:,0] = y0[:, 1]
state = solver.step(htry, dynamics, state)
print("Solution after first step:")
print("Time: ", state.x)
print("Solution: ", state.y)
print(f"Next step {state.hnext}")
print(f'Step actually taken {state.hdid}')


#concatenate the solution

count = 1
while (ft-state.x).any() > 0.0:
    count +=1
    htry = state.hnext.clone()
    mask = state.hnext > ft- state.x
    if mask.any():
        htry[mask] = ft[mask]-state.x[mask]
    mask2 = htry > state.EPS
    #print(f'mask2={mask2}')
    if math.isclose(state.x, 90.3191293896332894, rel_tol=1.0e-10, abs_tol=1.0e-10):
        print(f'x={state.x}')

    if mask2.any():
        state[mask2] = solver.step(htry[mask2], dynamics, state[mask2])
    else:
        break



    # x1s.append(solver.y[0])
    # x2s.append(solver.y[1])
    # Print the result
    print("Solution after a single step:")
    print("Time: ", state.x)
    print("Solution: ", state.y)
    print(f"Next step {state.hnext}")
    print(f'Step actually taken {state.hdid}')

end = time.time()
print(f'Time taken {end-start}')
print(f'Total number of steps {count}')