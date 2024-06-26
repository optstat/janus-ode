import torch

def DormandPrince8(f, t0, t_end, y0, h, atol=1e-10, rtol=1e-6):
    # coefficients
    c2 = 1 / 5
    c3 = 3 / 10
    c4 = 4 / 5
    c5 = 8 / 9
    c6 = 1
    c7 = 1
    a21 = 1 / 5
    a31 = 3 / 40
    a32 = 9 / 40
    a41 = 44 / 45
    a42 = -56 / 15
    a43 = 32 / 9
    a51 = 19372 / 6561
    a52 = -25360 / 2187
    a53 = 64448 / 6561
    a54 = -212 / 729
    a61 = 9017 / 3168
    a62 = -355 / 33
    a63 = 46732 / 5247
    a64 = 49 / 176
    a65 = -5103 / 18656
    a71 = 35 / 384
    a73 = 500 / 1113
    a74 = 125 / 192
    a75 = -2187 / 6784
    a76 = 11 / 84
    b1 = 35 / 384
    b3 = 500 / 1113
    b4 = 125 / 192
    b5 = -2187 / 6784
    b6 = 11 / 84
    b7 = 0
    bhat1 = 5179 / 57600
    bhat3 = 7571 / 16695
    bhat4 = 393 / 640
    bhat5 = -92097 / 339200
    bhat6 = 187 / 2100
    bhat7 = 1 / 40

    t = t0
    y = y0
    while True:
        # evaluate derivatives
        k1 = f(t, y)
        k2 = f(t + c2 * h, y + h * (a21 * k1))
        k3 = f(t + c3 * h, y + h * (a31 * k1 + a32 * k2))
        k4 = f(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3))
        k5 = f(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
        k6 = f(t + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
        k7 = f(t + c7 * h, y + h * (a71 * k1 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6))

        # compute solution and error estimates
        ynew = y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7)
        yhat = y + h * (bhat1 * k1 + bhat3 * k3 + bhat4 * k4 + bhat5 * k5 + bhat6 * k6 + bhat7 * k7)
        dy = ynew - yhat

        # compute error estimate
        scale = atol + rtol * torch.max(torch.abs(ynew), torch.abs(y))
        if torch.norm(scale) == 0:
            err = 0
        else:
            err = torch.max(torch.abs(dy) / scale)


        # adjust step size
        if err < 1 and err > 0:
            h = 0.9 * h * (1 / err) ** 0.125
            if h > 2 * h:
                h = 2 * h
            elif h < 0.1 * h:
                h = 0.1 * h
        elif err == 0:
            h = 2 * h
        else:
            h = 0.9 * h * (1 / err) ** 0.2

        # update state
        t = t + h
        y = ynew

        # return current state and time step size
        # return current state and time step size
        if t >= t_end:
            return y, h

def f(t, y):
    return -y

# initial conditions
t0 = 0
y0 = torch.tensor([1.0])

# desired final time
tf = 5.0

# step size
h = 0.001

# error tolerance
atol = 1e-10
rtol = 1e-6

# create the solver
solver = DormandPrince(f, t0, y0, h, atol=atol, rtol=rtol)

# solve the differential equation
t = t0
y = y0
while t < tf:
    # take a step
    y, h = next(solver)
    if t + h > tf:
        h = tf - t
    # update time
    t = t + h

    # print current state
    print(f"t={t:.4f} y={y.item():.6f} h={h:.6f}")
