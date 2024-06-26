import jax
import jax.numpy as jnp
import numpy as np
jax.config.update("jax_enable_x64", True)
from seulexted import SeulexTed
from seulexted import SeulexTedState
from tensormatdual import TensorMatDual
from tensordual import TensorDual
import time
import math
device = 'cpu'

# Define the dynamics class for the Van der Pol oscillator
mu=10.0
M=1
N=1 #Only need derivatives wrt h
factor = 1.0
class VanDerPolDynamicsDual:
    def jacobian(self, t, y):
        M, D, N = y.d.shape
        #In the jacobian the dual part is the square of the vector part
        dfdy = TensorDual.createZero(jnp.zeros((M, D, D), dtype=y.r.dtype, device=y.r.device), N)
        x1 = TensorDual(y.r[:M, 0:1], y.d[:M, 0:1, :N])
        x2 = TensorDual(y.r[:M, 1:2], y.d[:M, 1:2, :N])
        df1dx1  = x1*0.0
        df1dx2  = x1*0.0+factor
        df2dx1  = (-2.0 * mu * x1 * x2 - 1.0)*factor
        df2dx2  = mu * (1 - x1 ** 2)*factor
        dfdy.r[:, 0, 0]    = df1dx1.r
        dfdy.d[:, 0, 0, :] = df1dx1.d
        dfdy.r[:, 0, 1]    = df1dx2.r
        dfdy.d[:, 0, 1, :] = df1dx2.d
        dfdy.r[:, 1, 0]    = df2dx1.r
        dfdy.d[:, 1, 0, :] = df2dx1.d
        dfdy.r[:, 1, 1]    = df2dx2.r
        dfdy.d[:, 1, 1, :] = df2dx2.d
        return dfdy

    def __call__(self, t, y):
        dydt = 0.0*y
        x1 = TensorDual(y.r[:M, 0:1], y.d[:M, 0:1, :N])
        x2 = TensorDual(y.r[:M, 1:2], y.d[:M, 1:2, :N])
        dx1dt = x2*factor
        dx2dt = (mu * (1 - x1 ** 2) * x2 - x1)*factor
        dydt.r[:, 0] = dx1dt.r
        dydt.d[:, 0, :] = dx1dt.d
        dydt.r[:, 1] = dx2dt.r
        dydt.d[:, 1, :] = dx2dt.d
        return dydt




x0 = TensorDual.createZero(jnp.zeros([M,1], dtype=jnp.float64), N)  # Starting point
#The sensitivity here is wrt to the initial conditions
y0 = TensorDual.createZero(jnp.ones([M,2], dtype=jnp.float64), N)#+\
     #TensorDual.createZero(torch.rand([M,2], dtype=torch.float64, device=device), N)*0.01
#y0.d = torch.eye(N, dtype=y0.r.dtype, device=y0.r.device).unsqueeze(0).repeat(M, 1, 1)
y0.r.at[:, 0].set(2.0)
y0.r.at[:, 1].set(0.0)
#y0[0,:]= torch.tensor([0.5, 0.5], dtype=torch.float64)# Starting dynamics
atol = 1.0e-16# Absolute tolerance
rtol = 1.0e-9# Relative tolerance

KMAXX = 13
state = SeulexTedState(y0, x0, atol, rtol, KMAXX)
solver = SeulexTed()

# Create an instance of the Van der Pol dynamics
dynamics = VanDerPolDynamicsDual()

# Perform a single step using the solver
ft = TensorDual.ones_like(x0)*25.0

x1s  = jnp.zeros((M, 1), dtype=jnp.float64)
x2s  = jnp.zeros((M, 1), dtype=jnp.float64)
ts   = jnp.zeros((M, 1), dtype=jnp.float64)
dx1dx1s  = jnp.ones((M, 1, 1),  dtype=jnp.float64)
dx1dx2s  = jnp.zeros((M, 1, 1), dtype=jnp.float64)
dx2dx1s  = jnp.zeros((M, 1, 1),  dtype=jnp.float64)
dx2dx2s  = jnp.ones((M, 1, 1), dtype=jnp.float64)
x1s.at[:,0].set(y0.r[:, 0])
x2s.at[:,0].set(y0.r[:, 1])

htry = TensorDual.ones_like(x0)*0.01
htry.d.at[:].set(1.0) #This is the parameter wrt which we want the derivative
state = solver.step(htry, dynamics, state)
print("Solution after first step:")
print("Time: ", state.x)
print("Solution: ", state.y)
print(f"Next step {state.hnext}")
print(f'Step actually taken {state.hdid}')
#concatenate the solution
x1s = jnp.concatenate((x1s, state.y.r[:, 0:1]), axis=1)
x2s = jnp.concatenate((x2s, state.y.r[:, 1:2]), axis=1)
ts  = jnp.concatenate((ts,  state.x.r[:, 0:1]),  axis=1)


#Use Richarsdon Extrapolation to check the gradients
h = 1.0e-8
htryph = htry+h
stateph = SeulexTedState(y0, x0, atol, rtol, KMAXX)
solverph = SeulexTed()
statephn = solver.step(htryph, dynamics, stateph)
htrymh = htry-h
statemh = SeulexTedState(y0, x0, atol, rtol, KMAXX)
solvermh = SeulexTed()
statemhn = solver.step(htrymh, dynamics, statemh)

htry = jnp.ones([M,], dtype=jnp.float64)*0.01
state = SeulexTedState(y0, x0, atol, rtol, KMAXX)
solver = SeulexTed()
staten = solver.step(htry, dynamics, stateph)
J = state.y.square().sum()

#Now calculate an objective function
Jph =  stateph.y.square().sum()
Jmh = statemh.y.square().sum()
#Now calculate the gradient using finite differences
dJdh = (Jph-Jmh)/(2.0*h)


print(f'dJdh={dJdh} versus {J.d}')



now = time.time()
count = 1
while (ft-state.x).r > 0.0:
    count +=1
    htry = state.hnext.clone()
    mask = state.hnext > ft- state.x
    if mask.any():
        htry.r[mask] = ft.r[mask]-state.x.r[mask]
        htry.d[mask] = ft.d[mask]-state.x.d[mask]
    mask2 = htry > state.EPS
    #print(f'mask2={mask2}')

    if mask2.any():
        htrymask2 = TensorDual(htry.r[mask2], htry.d[mask2])
        state[mask2] = solver.step(htrymask2, dynamics, state[mask2])
    else:
        break



    # x1s.append(solver.y[0])
    # x2s.append(solver.y[1])
    # Print the result
    print("Solution after a single step:")
    print("Time: ", state.x)
    if math.isclose(state.x.r, 90.3191293896332894, rel_tol=1.0e-10, abs_tol=1.0e-10):
        print(f'x={state.x}')

    print("Solution: ", state.y)
    print(f"Next step {state.hnext}")
    print(f'Step actually taken {state.hdid}')
    x1s = jnp.concatenate((x1s, state.y.r[:, 0:1]), dim=1)
    x2s = jnp.concatenate((x2s, state.y.r[:, 1:2]), dim=1)
    ts = jnp.concatenate((ts, state.x.r[:, 0:1]), dim=1)
    dx1dx1s = jnp.concatenate((dx1dx1s, state.y.d[:, 0:1, 0:1]), dim=1)
    dx1dx2s = jnp.concatenate((dx1dx2s, state.y.d[:, 0:1, 1:2]), dim=1)
    dx2dx1s = jnp.concatenate((dx2dx1s, state.y.d[:, 1:2, 0:1]), dim=1)
    dx2dx2s = jnp.concatenate((dx2dx2s, state.y.d[:, 1:2, 1:2]), dim=1)

end = time.time()
print(f'Time taken {end-now}')
print(f'Total number of steps {count}')

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['agg.path.chunksize'] = 1000
markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

fig = plt.figure()
# Add the first plot

ax = fig.add_subplot(111)
# make the plot bigger
fig.set_size_inches(18.5, 10.5)
# add a second plot
# add space between plots
for idx in range(M):
    ax.scatter(ts[idx, :].cpu().numpy(), x1s[idx, :].cpu().numpy(), label='' + str(idx),
               marker=np.random.choice(markers))
# put the legend outside the plot
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# label the x axis
ax.set_xlabel('t')
# label the y axis
ax.set_ylabel('x1')
plt.show()
plt.close(fig)

fig = plt.figure()
# Add the first plot

ax2 = fig.add_subplot(111)
# make the plot bigger
fig.set_size_inches(18.5, 10.5)
# add a second plot
# add space between plots
for idx in range(M):
    ax2.scatter(ts[idx, :].cpu().numpy(), x2s[idx, :].cpu().numpy(), label='' + str(idx),
                marker=np.random.choice(markers))
# put the legend outside the plot
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# label the x axis
ax2.set_xlabel('t')
# label the y axis
ax2.set_ylabel('x2')
plt.show()
plt.close(fig)

fig = plt.figure()
# Add the first plot

ax3 = fig.add_subplot(111)
# make the plot bigger
fig.set_size_inches(18.5, 10.5)
# add a second plot
# add space between plots
for idx in range(M):
    ax3.scatter(x1s[idx, :].cpu().numpy(), x2s[idx, :].cpu().numpy(), label='' + str(idx),
                marker=np.random.choice(markers))
# put the legend outside the plot
ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# label the x axis
ax3.set_xlabel('x1')
# label the y axis
ax3.set_ylabel('x2')
plt.show()
plt.close(fig)

fig = plt.figure()
# Add the first plot

ax4 = fig.add_subplot(111)
# make the plot bigger
fig.set_size_inches(18.5, 10.5)
# add a second plot
# add space between plots
for idx in range(M):
    ax4.scatter(ts[idx, :].cpu().numpy(), dx1dx1s[idx, :].cpu().numpy(), label='' + str(idx),
                marker=np.random.choice(markers))
# put the legend outside the plot
# label the x axis
ax4.set_xlabel('t')
# label the y axis
ax4.set_ylabel('dx1dx10')
plt.show()
# close the figure
plt.close(fig)

fig = plt.figure()
# Add the first plot

ax = fig.add_subplot(111)
# make the plot bigger
fig.set_size_inches(18.5, 10.5)
# add a second plot
# add space between plots
for idx in range(M):
    ax.scatter(ts[idx, :].cpu().numpy(), dx1dx2s[idx, :].cpu().numpy(), label='' + str(idx),
               marker=np.random.choice(markers))
# put the legend outside the plot
# label the x axis
ax.set_xlabel('t')
# label the y axis
ax.set_ylabel('dx1dx20')
plt.show()
# close the figure
plt.close(fig)

fig = plt.figure()
# Add the first plot
ax = fig.add_subplot(111)
# make the plot bigger
fig.set_size_inches(18.5, 10.5)
# add a second plot
# add space between plots
for idx in range(M):
    ax.scatter(ts[idx, :].cpu().numpy(), dx2dx1s[idx,  :].cpu().numpy(), label='' + str(idx),
               marker=np.random.choice(markers))
# put the legend outside the plot
# label the x axis
ax.set_xlabel('t')
# label the y axis
ax.set_ylabel('dx2dx10')
plt.show()
# close the figure
plt.close(fig)

fig = plt.figure()
# Add the first plot
ax = fig.add_subplot(111)
# make the plot bigger
fig.set_size_inches(18.5, 10.5)
# add a second plot
# add space between plots
for idx in range(M):
    ax.scatter(ts[idx, :].cpu().numpy(), dx2dx2s[idx, :].cpu().numpy(), label='' + str(idx),
               marker=np.random.choice(markers))
# put the legend outside the plot
# label the x axis
ax.set_xlabel('t')
# label the y axis
ax.set_ylabel('dx2dx20')
plt.show()
# close the figure
plt.close(fig)
