from tensordual import TensorDual
import torch
import torch
import matplotlib.pyplot as plt




# Generate code for dorpi5 using only pytorch and no other libraries
import torch



#
def dopri5_step(ode, t, y, h, rtol, netargs):
    """Take a single step of the dopri5 solver."""
    # Compute the derivative at the current state.
    k1 = ode(t, y, netargs)
    k2 = ode(t + 1 / 5 * h, y + 1 / 5 * h * k1, netargs)
    k3 = ode(t + 3 / 10 * h, y + 3 / 40 * h * k1 + 9 / 40 * h * k2, netargs)
    k4 = ode(t + 4 / 5 * h, y + 44 / 45 * h * k1 - 56 / 15 * h * k2 + 32 / 9 * h * k3, netargs)
    k5 = ode(t + 8 / 9 * h,
             y + 19372 / 6561 * h * k1 - 25360 / 2187 * h * k2 + 64448 / 6561 * h * k3 - 212 / 729 * h * k4, netargs)
    k6 = ode(t + h,
             y + 9017 / 3168 * h * k1 - 355 / 33 * h * k2 + 46732 / 5247 * h * k3 + 49 / 176 * h * k4 - 5103 / 18656 * h * k5, netargs)

    # Compute the 5th-order and 4th-order estimates of the next state.
    y5 = y + 35 / 384 * h * k1 + 500 / 1113 * h * k3 + 125 / 192 * h * k4 - 2187 / 6784 * h * k5 + 11 / 84 * h * k6
    y4 = y + 5179 / 57600 * h * k1 + 7571 / 16695 * h * k3 + 393 / 640 * h * k4 - 92097 / 339200 * h * k5 + 187 / 2100 * h * k6

    # Estimate the local error.
    e = TensorDual.norm(y5 - y4)
    max_safety_factor = 2.0
    min_safety_factor = 0.2
    safety_factor = 0.8
    h_next = h * min(max_safety_factor,
                    max(min_safety_factor, safety_factor * (rtol / e) ** (1 / 5)))
    # Choose the next step size.
    #h_next = 0.9 * h * (rtol / e) ** (1 / 5)

    # Use the 5th-order estimate as the next state.
    return y5, h_next, e


def dopri5(f, y, t, h, rtol, atol, netargs):
    """Use dopri5 to integrate the ODE defined by derivative_fn."""
    # Convert initial_state and times to tensors.


    # Integrate the ODE using dopri5.
    y_out = [y]
    t_out = [t[0]]
    i = 0
    href = h
    while (t_out[-1] < t[-1]).all():
        # Take a single step of the dopri5 solver.
        t0 = t_out[-1]
        y0 = y
        y_next, h_next, e = dopri5_step(f, t0, y0, h, rtol, netargs)

        if e < rtol+atol:
            # Accept the step.
            y_out.append(y_next)
            t_out.append(t0 + h)
            h = href
            y = y_next
            i = i + 1
        else:
            # Reject the step and try again with a smaller step size.
            h = h_next





    return t_out, y_out



def van_der_pol_oscillator(t, y, netargs):
    mu = 5.0
    dydt = torch.zeros_like(y)
    dydt[:,0] = y[:,1]
    dydt[:,1] = mu * (1 - y[:,0] ** 2) * y[:,1] - y[:,0]
    return dydt
#add __main__ here

if __name__ == "__main__":
    # Initial conditions
    y0 = torch.tensor([[1.0, 1.0]], dtype=torch.float64)

    # Time span
    t_span = torch.tensor([0.0, 20.0], dtype=torch.float64)

    # Solve the ODE using the dopri5_torch solver
    t, y = dopri5(van_der_pol_oscillator, y0, t_span, 0.01, 1.0e-2, 1.0e-2, None)

    # Plot the results
    print(f'Total number of steps: {len(t)}')
    plt.figure(figsize=(10, 5))
    plt.plot(y[:,:, 0], y[:,:, 1])
    plt.title('Van der Pol Oscillator (mu=5) using Dopri5 (PyTorch)')
    plt.legend()
    plt.grid()
    plt.show()
