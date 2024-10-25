import numpy as np
from scipy.integrate import solve_ivp
#Reference test for the Augmented Van der Pol oscillator
#To be compared with the example of the augmented Van der Pol oscillator 
# Define the Van der Pol oscillator
def aug_vdp_deriv(t, y):
    # State vector components
    p1 = y[0]
    p2 = y[1]
    x1 = y[2]
    x2 = y[3]
    
    # Control law for u2star
    if p2 * (1 - x1**2) * x2 < 0:
        u2star = 100
    else:
        u2star = 1
    
    # State vector derivative
    dp1dt = p2 * u2star * (-2.0 * x1) * x2 - p2
    dp2dt = p1 + p2 * u2star * (1.0 - x1**2)
    dx1dt = x2  # Assuming u1star = 0 for now, as it's not defined
    dx2dt = u2star * (1.0 - x1**2) * x2 - x1  # Assuming u3star = 0 for now, as it's not defined

    return np.array([dp1dt, dp2dt, dx1dt, dx2dt])



def aug_vdp_jac(t, y):
    # State vector components
    p1 = y[0]
    p2 = y[1]
    x1 = y[2]
    x2 = y[3]
    
    # Control law for u2star
    if p2 * (1 - x1**2) * x2 < 0:
        u2star = 100
    else:
        u2star = 1
    
    # Initialize Jacobian matrix
    jac = np.zeros((4, 4))
    
    # Fill in Jacobian entries
    jac[0, 1] = u2star * (-2 * x1) * x2 - 1.0
    jac[0, 2] = -p2 * u2star * 2 * x2
    jac[0, 3] = -p2 * u2star * 2 * x1
    
    jac[1, 0] = 1.0
    jac[1, 1] = u2star * (1 - x1**2)
    jac[1, 2] = p2 * u2star * (-2 * x1)
    
    jac[2, 3] = 1.0
    
    jac[3, 2] = u2star * (-2 * x1 * x2) - 1.0
    jac[3, 3] = u2star * (1 - x1**2)
    
    return jac

# Initial conditions and parameters
y0 = [5.0, 5.0, 0.0, 1.5]  # Initial condition
t_span = (0, 10)

# Set the precision for Radau solver
rtol = 1e-6
atol = 1e-9



# Call solve_ivp using the Radau method, specifying both the function and Jacobian
solution = solve_ivp(
    fun=aug_vdp_deriv,
    t_span=t_span,
    y0=y0,
    method='Radau',
    jac=aug_vdp_jac,
    rtol=rtol,
    atol=atol,
    dense_output=True  # Allows for easy interpolation of the solution
)
# Access solution values (solution.t for times, solution.y for states)
t_values = solution.t
y_values = solution.y
# Print the final point
print(f'Final point: t={t_values[-1]}, y={y_values[:, -1]}')
