import torch

def DormandPrince8_dual(f, t0, y0, h, args, atol=1e-12, rtol=1e-8):
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
        k1 = f(t, y, args)
        k2 = f(t + c2 * h, y + h * (a21 * k1), args)
        k3 = f(t + c3 * h, y + h * (a31 * k1 + a32 * k2), args)
        k4 = f(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3), args)
        k5 = f(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), args)
        k6 = f(t + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), args)
        k7 = f(t + c7 * h, y + h * (a71 * k1 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6), args)

        # compute solution and error estimates
        ynew = y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7)
        yhat = y + h * (bhat1 * k1 + bhat3 * k3 + bhat4 * k4 + bhat5 * k5 + bhat6 * k6 + bhat7 * k7)
        dy = ynew - yhat

        # compute error estimate
        scale = atol + rtol * torch.max(torch.abs(ynew.r), torch.abs(y.r))
        if torch.norm(scale) == 0:
            err = 0
        else:
            err = torch.max(torch.abs(dy.r) / scale)


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
        yield y, h



def dopri8_dual_span(f, y0, times, args, atol=1e-12, rtol=1e-8):
    # coefficients

    # Define all the coefficients here as shown in the original code
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

    t = times[0]
    y = y0
    i = 1  # index to keep track of times list
    h = times[i] - t  # set the initial step size to the first time difference
    ys = [y.clone()]  # initialize the list of values of y at each h

    while t < times[-1]:  # loop until the end time in the times list
        # Stop the integration when h becomes smaller than the smallest time difference in the times list
        #if h < min([times[i] - times[i+1] for i in range(len(times)-1)]):
        #    break

        # evaluate derivatives
        k1 = f(t, y, args)
        k2 = f(t + c2 * h, y + h * (a21 * k1), args)
        k3 = f(t + c3 * h, y + h * (a31 * k1 + a32 * k2), args)
        k4 = f(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3), args)
        k5 = f(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), args)
        k6 = f(t + c6 * h, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5), args)
        k7 = f(t + c7 * h, y + h * (a71 * k1 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6), args)

        # compute solution and error estimates
        ynew = y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6 + b7 * k7)
        yhat = y + h * (bhat1 * k1 + bhat3 * k3 + bhat4 * k4 + bhat5 * k5 + bhat6 * k6 + bhat7 * k7)
        dy = ynew - yhat

        # compute error estimate
        scale = atol + rtol * torch.max(torch.abs(ynew.r), torch.abs(y.r))
        if torch.norm(scale) == 0:
            err = 0
        else:
            err = torch.max(torch.abs(dy.r) / scale)

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


        t = t + h
        y = ynew
        # check if the current time is equal to the next time in the times list
        if t >= times[i] and i < len(times) - 1:
            ys.append(y.clone())  # append the current value of y to the list
            i += 1  # increment the index to move to the next time
            if times[i] - t >0:
                h = times[i] - t  # set the new step size to the difference between the next time and the current time
            else:
                h = 0.0
    return ys  # return the list of values of y at each h
