import numpy as np
import torch
from dopri853ted import Dopri853TeD
from tensordual import TensorDual
#Initialize the pytorch random number generator
torch.manual_seed(0)

device = torch.device('cpu')
M = 100
ft = 1.0
y0r = torch.zeros((M, 3), dtype=torch.float64).to(device)
y0r[:, 0] = 2.0+torch.rand((M)).double()*0.1
y0r[:, 1] = 0.0+torch.rand((M)).double()*0.1
y0r[:, 2] = torch.rand((M)).double()
identity_list = [torch.eye(3, dtype=torch.float64) for _ in range(M)]
y0d = torch.stack(identity_list).to(device)
y0 = TensorDual(y0r, y0d)
atol = 1e-8
rtol = 1e-8


def vdpd(t, x):
    x1 = TensorDual(x.r[:, 0], x.d[:, 0])
    x2 = TensorDual(x.r[:, 1], x.d[:, 1])
    mus = TensorDual(x.r[:, 2], x.d[:, 2])
    dydx = TensorDual.zeros_like(x)
    dydx1 = x2
    dydx2 = mus*(1 -x1 *x1) * x2 - x1
    dydx.r[:, 0:1] = dydx1.r
    dydx.d[:, 0:1] = dydx1.d
    dydx.r[:, 1:2] = dydx2.r
    dydx.d[:, 1:2] = dydx2.d
    #dydxr = torch.squeeze(torch.stack([dydx1.r, dydx2.r, dydx3.r], dim=1))
    #dydxd = torch.squeeze(torch.stack([dydx1.d, dydx2.d, dydx3.d], dim=1))
    #dydx = TensorDual(dydxr, dydxd)
    if dydx.r.isnan().any():
        print("Nan detected in real")
        exit(1)
    if dydx.d.isnan().any():
        print("Nan detected in dual")
        #exit(1)
    return dydx


def calc_yf(y0):
    M = y0.r.shape[0]
    h = torch.ones(M).double() * 1.0e-2
    #The trick is to create a dual tensor that has the identity matrix as the derivative
    dydx = vdpd(0.0, y0) #Need to pass in the initial dynamics

    ftd = TensorDual(torch.ones(M, 1).double() * ft, torch.zeros(M, 1, 3).double())
    ys = torch.unsqueeze(y0.r, dim=2)
    dydx0 = torch.unsqueeze(y0.d, dim=3)
    x0 = TensorDual.zeros_like(ftd)
    solver = Dopri853TeD(y0, dydx, x0, atol, rtol, vdpd)
    solver.step(h)


    while (solver.x < ftd).any():
        h = solver.hnext
        if ((solver.x + h) > ftd).any():
            mask = (solver.x + h) > ftd
            h[mask] = torch.squeeze((ftd.r[mask] - solver.x.r[mask]), 1)
        #print(f'h= {h}')
        #print(f't= {solver.x}')
        #print(f'yout= {solver.yout}')
        solver.step(h)
        dydx0 = torch.cat((dydx0, torch.unsqueeze(solver.yout.d, 3)), dim=3)
    return solver.yout.clone()


def ridders_jac(y0, ii, jj, dt=1e-6):
    ntab = 10
    con = 1.4
    con2 = con * con
    big = torch.finfo(torch.float64).max
    safe = 2.0
    hh = dt
    a = torch.empty((ntab, ntab), dtype=torch.float64)

    if hh == 0.0:
        raise ValueError("dt must be nonzero in ridders_derivative.")

    y0p = y0.clone()
    y0p.r[:,jj]  = y0.r[:,jj]+hh
    y0m = y0.clone()
    y0m.r[:,jj] = y0.r[:,jj] - hh
    yfp = calc_yf(y0p)
    yfm = calc_yf(y0m)
    a[0, 0]   = (yfp.r[:,ii]-yfm.r[:,ii]) / (2.0 * hh)
    err = big

    for i in range(1, ntab):
        hh /= con
        y0p = y0.clone()
        y0p.r[:, jj] = y0.r[:, jj] + hh
        y0m = y0.clone()
        y0m.r[:, jj] = y0.r[:, jj] - hh
        yfp = calc_yf(y0p)
        yfm = calc_yf(y0m)
        a[0, i] = (yfp.r[:,ii] - yfm.r[:,ii]) / (2.0 * hh)
        fac = con2
        for j in range(1, i + 1):
            a[j, i] = (a[j - 1, i] * fac - a[j - 1, i - 1]) / (fac - 1.0)
            fac = con2 * fac
            errt = max(abs(a[j, i] - a[j - 1, i]), abs(a[j, i] - a[j - 1, i - 1]))
            if errt <= err:
                err = errt
                ans = a[j, i]
        if abs(a[i, i] - a[i - 1, i - 1]) >= safe * err:
            break

    return ans

if __name__ == '__main__':
    yf = calc_yf(y0)
    print(f'yf = {yf}')
    for i in range(M):
        y0ati = TensorDual(y0.r[i:i+1], y0.d[i:i+1])
        jac_0_0 = ridders_jac(y0ati,0, 0)
        jac_0_1 = ridders_jac(y0ati,0, 1)
        jac_0_2 = ridders_jac(y0ati, 0, 2)
        jac_1_0 = ridders_jac(y0ati,1, 0)
        jac_1_1 = ridders_jac(y0ati,1, 1)
        jac_1_2 = ridders_jac(y0ati, 1, 2)
        jac_2_0 = ridders_jac(y0ati,2, 0)
        jac_2_1 = ridders_jac(y0ati,2, 1)
        jac_2_2 = ridders_jac(y0ati, 2, 2)
        print(f'yf.d[i:i+1, 0, 0] versus jac_0_0 {yf.d[i:i+1, 0, 0]} {jac_0_0}')
        print(f'yf.d[i:i+1, 0, 1] versus jac_0_1 {yf.d[i:i+1, 0, 1]} {jac_0_1}')
        print(f'yf.d[i:i+1, 0, 2] versus jac_0_1 {yf.d[i:i + 1, 0, 2]} {jac_0_2}')
        print(f'yf.d[i:i+1, 1, 0] versus jac_1_0 {yf.d[i:i+1, 1, 0]} {jac_1_0}')
        print(f'yf.d[i:i+1, 1, 1] versus jac_1_1 {yf.d[i:i+1, 1, 1]} {jac_1_1}')
        print(f'yf.d[i:i+1, 1, 2] versus jac_1_1 {yf.d[i:i + 1, 1, 2]} {jac_1_2}')
        print(f'yf.d[i:i+1, 2, 0] versus jac_2_0 {yf.d[i:i+1, 2, 0]} {jac_2_0}')
        print(f'yf.d[i:i+1, 2, 1] versus jac_2_1 {yf.d[i:i+1, 2, 1]} {jac_2_1}')
        print(f'yf.d[i:i+1, 2, 2] versus jac_2_1 {yf.d[i:i + 1, 2, 2]} {jac_2_2}')

        #   print(f'ft={ft} yf={yf}')

        assert torch.allclose(yf.d[i:i+1, 0, 0], jac_0_0, atol=1.0e-2, rtol=1.0e-2)
        assert torch.allclose(yf.d[i:i+1, 0, 1], jac_0_1, atol=1.0e-2, rtol=1.0e-2)
        assert torch.allclose(yf.d[i:i+1, 1, 0], jac_1_0, atol=1.0e-2, rtol=1.0e-2)
        assert torch.allclose(yf.d[i:i+1, 1, 1], jac_1_1, atol=1.0e-2, rtol=1.0e-2)
