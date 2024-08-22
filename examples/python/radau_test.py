import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the Van der Pol oscillator
def van_der_pol(t, y, mu):
    return [y[1], mu * (1 - y[0]**2) * y[1] - y[0]]

def van_der_pol_rev(t, y, mu):
    return [-y[1], -(mu * (1 - y[0]**2) * y[1] - y[0])]

# Initial conditions and parameters
mu = 4.0
y0 = [2.0, 0.0]  # Initial condition
t_span = (0, 300)
max_step = 0.1
# Set the precision for Radau solver
rtol = 1e-9
atol = 1e-12

# Estimate the period
period = (3.0-2.0*np.log(2.0))*mu + 2.0*3.141592653589793/1000.0/3.0
print(f"Estimated period for mu={mu} : {period}")

# Integrate the system forward over one period using Radau
t_span_forward = (0, period)
sol_forward = solve_ivp(van_der_pol, t_span_forward, y0, args=(mu,), method='Radau', rtol=rtol, atol=atol)
print(f'Forward solution={sol_forward}')
# Integrate the system backward over one period using Radau
y0r = sol_forward.y[:, -1]
sol_backward = solve_ivp(van_der_pol_rev, t_span_forward, y0r, args=(mu,), method='Radau', rtol=rtol, atol=atol)
print(f'Reverse solution={sol_backward}')

# Plot the results
#plt.plot(sol_forward.t, sol_forward.y[0], label='Forward')
#plt.plot(sol_backward.t, sol_backward.y[0], label='Backward')
#plt.xlabel('Time')
#plt.ylabel('y[0]')
#plt.legend()
#plt.title(f'Van der Pol Oscillator with mu={mu}')
#plt.show()