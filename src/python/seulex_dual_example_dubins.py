import torch
from seulexted import SeulexTed
from seulexted import SeulexTedState
from tensormatdual import TensorMatDual
from tensordual import TensorDual
import time
import math
import matplotlib.pyplot as plt
import numpy as np

device = 'cpu'

# Define the dynamics class for the Van der Pol oscillator
mu=1.0
M=1
N=10000 #Only need derivatives wrt h
factor = 1.0
#y0[0,:]= torch.tensor([0.5, 0.5], dtype=torch.float64)# Starting dynamics
atol = 1.0e-32# Absolute tolerance
rtol = 1.0e-14# Relative tolerance

KMAXX = 13
#Initialize the random number generator for pytorch
torch.manual_seed(0)

class DubinsDynamicsDual:
    def jacobian(self, t, y, params):
        M, D, N = y.d.shape
        #In the jacobian the dual part is the square of the vector part
        dfdy = TensorMatDual.createZero(torch.zeros((M, D, D), dtype=y.r.dtype, device=y.r.device), N)
        x1 = TensorDual(y.r[:M, 0:1], y.d[:M, 0:1, :N])
        x2 = TensorDual(y.r[:M, 1:2], y.d[:M, 1:2, :N])
        x3 = TensorDual(y.r[:M, 2:3], y.d[:M, 2:3, :N])
        u = params
        df1dx1  = x2*0.0
        df1dx2  = x2*0.0
        df1dx3  = -x3.sin()*factor
        #-(1.0-x1*x1)*x2*mu*factor
        df2dx1  = x1*0.0
        df2dx2  = x2*0.0
        df2dx3  = x3.cos()*factor
        df3dx1 = x1*0.0
        df3dx2 = x2*0.0
        df3dx3 = x3*0.0
        dfdy.r[:, 0, 0]    = df1dx1.r.view(dfdy.r[:, 0, 0].shape)
        dfdy.d[:, 0, 0, :] = df1dx1.d.view(dfdy.d[:, 0, 0, :].shape)
        dfdy.r[:, 0, 1]    = df1dx2.r.view(dfdy.r[:, 0, 1].shape)
        dfdy.d[:, 0, 1, :] = df1dx2.d.view(dfdy.d[:, 0, 1, :].shape)
        dfdy.r[:, 0, 2]    = df1dx3.r.view(dfdy.r[:, 0, 2].shape)
        dfdy.d[:, 0, 2, :] = df1dx3.d.view(dfdy.d[:, 0, 2, :].shape)
        dfdy.r[:, 1, 0]    = df2dx1.r.view(dfdy.r[:, 1, 0].shape)
        dfdy.d[:, 1, 0, :] = df2dx1.d.view(dfdy.d[:, 1, 0, :].shape)
        dfdy.r[:, 1, 1]    = df2dx2.r.view(dfdy.r[:, 1, 1].shape)
        dfdy.d[:, 1, 1, :] = df2dx2.d.view(dfdy.d[:, 1, 1, :].shape)
        dfdy.r[:, 1, 2]    = df2dx3.r.view(dfdy.r[:, 1, 2].shape)
        dfdy.d[:, 1, 2, :] = df2dx3.d.view(dfdy.d[:, 1, 2, :].shape)
        dfdy.r[:, 2, 0]    = df3dx1.r.view(dfdy.r[:, 2, 0].shape)
        dfdy.d[:, 2, 0, :] = df3dx1.d.view(dfdy.d[:, 2, 0, :].shape)
        dfdy.r[:, 2, 1]    = df3dx2.r.view(dfdy.r[:, 2, 1].shape)
        dfdy.d[:, 2, 1, :] = df3dx2.d.view(dfdy.d[:, 2, 1, :].shape)
        dfdy.r[:, 2, 2]    = df3dx3.r.view(dfdy.r[:, 2, 2].shape)
        dfdy.d[:, 2, 2, :] = df3dx3.d.view(dfdy.d[:, 2, 2, :].shape)
        if torch.isnan(dfdy.r).any():
            print(f'dfdy.r={dfdy.r}')
        if torch.isnan(dfdy.d).any():
            print(f'dfdy.d={dfdy.d}')
        assert not torch.isnan(dfdy.r).any()
        assert not torch.isnan(dfdy.d).any()
        return dfdy

    def __call__(self, t, y, params):
        M, D, N = y.d.shape
        dydt = y.clone()*0.0
        x1 = TensorDual(y.r[:M, 0:1], y.d[:M, 0:1, :N])
        x2 = TensorDual(y.r[:M, 1:2], y.d[:M, 1:2, :N])
        x3 = TensorDual(y.r[:M, 2:3], y.d[:M, 2:3, :N])
        u = params
        dx1dt = x3.cos()*factor
        dx2dt = x3.sin()*factor
        dx3dt = u*factor
        dydt.r[:, 0]    = dx1dt.r.view(dydt.r[:, 0].shape)
        dydt.d[:, 0, :] = dx1dt.d.view(dydt.d[:, 0, :].shape)
        dydt.r[:, 1]    = dx2dt.r.view(dydt.r[:, 1].shape)
        dydt.d[:, 1, :] = dx2dt.d.view(dydt.d[:, 1, :].shape)
        dydt.r[:, 2] = dx3dt.r.view(dydt.r[:, 2].shape)
        dydt.d[:, 2, :] = dx3dt.d.view(dydt.d[:, 2, :].shape)
        return dydt

    def jac_test(self):
        M=1
        D=3
        N=D
        yr = torch.rand((M, D), dtype=torch.float64)
        yd = torch.zeros((M, D, N), dtype=torch.float64)
        for i in range(N):
            yd[:, i, i] = 1.0
        y = TensorDual(yr, yd)
        yc = y.clone()
        jac = self.jacobian(0.0, y)
        Jc = jac.square().sum(1).sum(2)
        for i in range(D):
            dx = 1.0e-8
            ycp = yc.clone()
            ycp.r[:, i] += dx
            jacp = self.jacobian(0.0, ycp)
            Jcp = jacp.square().sum(1).sum(2)
            ycm = yc.clone()
            ycm.r[:, i] -= dx
            jacm = self.jacobian(0.0, ycm)
            Jcm = jacm.square().sum(1).sum(2)
            Jcfd = (Jcp - Jcm)/(2.0*dx)
            print(f'Jcfd={Jcfd.r[0]}')
            print(f'Jc dual gradient={Jc.d[0, 0, 0, i]}')
            assert torch.allclose(Jcfd.r[0], Jc.d[0, 0, 0, i], atol=1.0e-3, rtol=1.0e-3)


    def dyns_test(self):
        M=1
        D=3
        N=D
        yr = torch.rand((M, D), dtype=torch.float64)
        yd = torch.zeros((M, D, N), dtype=torch.float64)
        for i in range(N):
            yd[:, i, i] = 1.0
        y = TensorDual(yr, yd)
        yc = y.clone()
        dydt = self(0.0, y)
        Jc = dydt.square().sum()
        for i in range(D):
            dx = 1.0e-8
            ycp = yc.clone()
            ycp.r[:, i] += dx
            dydtp = self(0.0, ycp)
            Jcp = dydtp.square().sum()
            ycm = yc.clone()
            ycm.r[:, i] -= dx
            dydtm = self(0.0, ycm)
            Jcm = dydtm.square().sum()
            Jcfd = (Jcp - Jcm)/(2.0*dx)
            print(f'Jcfd={Jcfd.r[0]}')
            print(f'Jc dual gradient={Jc.d[0, 0, i]}')
            assert torch.allclose(Jcfd.r[0], Jc.d[0, 0, i], atol=1.0e-3, rtol=1.0e-3)


#if __debug__:
#    dynamics = VanDerPolDynamicsDual()
#    dynamics.jac_test()
#    dynamics.dyns_test()


#Given the dynamics, initial conditions, initial time tolerances and paramaters
#calculate the trajectories and sensitivities
def odeintdual(dyns, y0in, x0in, ft, params, nparams_step=0,  rtol=1.0e-8, atol=1.0e-8):
    assert isinstance(x0in, torch.Tensor), "tin must by in the form of a pytorch tensor"
    assert y0in.dim() == 2, "y0in must be a 2 dimensional tensor"
    assert x0in.dim() == 2, "tin must be a 2 dimensional tensor"
    M, D = y0in.shape[0], y0in.shape[1]
    Ntot = D
    #The final sensitivities are stored in matrix
    sens = torch.zeros((M, D, Ntot), dtype=y0in.dtype, device=y0in.device)
    sens[:, :D, :D] = torch.eye(D, dtype=y0in.dtype, device=y0in.device)
    #Calculate the dual dimension
    solver = SeulexTed()
    if not isinstance(ft, torch.Tensor):
        ft = torch.tensor(ft, dtype=y0in.dtype, device=y0in.device)
    xcurr = x0in.clone()
    ycurr = y0in.clone()
    htry = torch.ones((M), dtype=y0in.dtype, device=y0in.device)*0.001
    state = SeulexTedState(ycurr, xcurr, params, nparams_step, atol, rtol)

    while state.x.r < ft:
        #statecheck = SeulexTedState(state.y.r, state.x.r, params, nparams_step, state.atol, state.rtol)
        #solvercheck = SeulexTed()
        #We will assume each step is starting from
        solver.step(htry, dyns, ft, state)

        htry = state.hnext.clone()

        if (state.x +htry) > ft:
            htry = ft - torch.squeeze(state.x.r)
        print(f'htry={htry}')

        #solvercheck.step(state.hdid.clone(), dyns, ft+100, statecheck)
        #dualcheck = statecheck.y.d[:, :D, :D]@dualy
        #dualy = dualcheck.clone()
        #currdual = state.y.d[:, :D, :D]
        #print(f'At count {state.count} dualcheck -currdual={dualcheck-currdual}')
        #if torch.sum((dualcheck-currdual).square()) > 1.0e-6:
        #    state.y.d[:, :D, :D] = dualcheck.clone()
    sensic = state.y.d
    #sensparams = sens[:, :D, D:D+nparams_step*nparams_step]
    #sensft = torch.unsqueeze(sens[:, :D, -1], -1)
    yf = state.y.r[:, :D]
    return state.x.r, yf, sensic#, sensparams, sensft


def calc_yf(x0, y0, ft, params, nparams_step,  atol, rtol):
    dynamics = DubinsDynamicsDual()
    tout, yout, sensic = odeintdual(dynamics, y0, x0, ft, params, nparams_step,  rtol=rtol, atol=atol)
    #sensall = torch.cat((sensic, sensparam, sensft), dim=2)
    #Jsensic = J.d[:, 0, 0:D]
    #Jsensparam = J.d[:, 0, D:D+params.shape[1]]
    #Jsensft = J.d[:, 0, -1]
    return yout, sensic




def fd_jac(x0, y0, ft, params, nparams_step, atol, rtol, ii, jj, h=1.0e-6):
    y0p = y0.clone()
    y0p[0,jj]  = y0[0,jj]+h
    y0pp = y0.clone()
    y0pp[0,jj] = y0[0,jj] + 2.0*h
    y0m = y0.clone()
    y0m[0,jj] = y0[0,jj] - h
    y0mm = y0.clone()
    y0mm[0,jj] = y0[0,jj] - 2.0*h
    yfpp, _ = calc_yf(x0, y0pp, ft, params, nparams_step, atol, rtol)
    yfp , _   = calc_yf(x0, y0p, ft, params, nparams_step, atol, rtol)
    yfm , _   = calc_yf(x0, y0m, ft, params, nparams_step, atol, rtol)
    yfmm, _ = calc_yf(x0, y0mm, ft, params, nparams_step, atol, rtol)
    return (-yfpp[0,ii]+8.0*yfp[0,ii]-8.0*y0[0,ii]+yfm[0,ii])/(12.0*h)


def ridders_jac(x0, y0, ft, params, nparams_step, atol, rtol, ii, jj, h=1.0e-6):
    ntab = 10
    con = torch.tensor(1.2, dtype=torch.float64)
    con2 = con * con
    big = torch.finfo(torch.float64).max
    safe = torch.tensor(2.0, dtype=torch.float64)
    hh = torch.tensor(h, dtype=torch.float64)
    a = torch.zeros((ntab, ntab), dtype=torch.float64)

    if h == 0.0:
        raise ValueError("h must be nonzero in ridders_derivative.")

    y0p = y0.clone()
    y0p[0,jj]  = y0[0,jj]+hh
    y0m = y0.clone()
    y0m[0,jj] = y0[0,jj] - hh
    yfp , _   = calc_yf(x0, y0p, ft, params, nparams_step, atol, rtol)
    yfm , _   = calc_yf(x0, y0m, ft, params, nparams_step, atol, rtol)
    a[0, 0]   = (yfp[0,ii]-yfm[0,ii]) / (2.0 * hh)
    err = big

    for i in range(1, ntab):
        hh /= con
        y0p = y0.clone()
        y0p[0, jj] = y0[0, jj] + hh
        y0m = y0.clone()
        y0m[0, jj] = y0[0, jj] - hh
        yfp, _ = calc_yf(x0, y0p, ft, params, nparams_step, atol, rtol)
        yfm, _ = calc_yf(x0, y0m, ft, params, nparams_step, atol, rtol)
        a[0, i] = (yfp[0,ii] - yfm[0,ii]) / (2.0 * hh)
        fac = con2
        for j in range(1, i + 1):
            a[j, i] = (a[j - 1, i] * fac - a[j - 1, i - 1]) / (fac - 1.0)
            fac = con2 * fac
            errt = torch.max(torch.abs(a[j, i] - a[j - 1, i]), torch.abs(a[j, i] - a[j - 1, i - 1]))
            if errt <= err:
                err = errt
                ans = a[j, i]
        if torch.abs(a[i, i] - a[i - 1, i - 1]) >= (safe * err):
            break

    return ans, err

def ridders_ft(x0, y0, ft, params, nparams_step, ii, atol, rtol, h=1e-3):
    ntab = 5
    con = 1.4
    con2 = con * con
    big = torch.finfo(torch.float64).max
    safe = 2.0
    hh = h
    a = torch.empty((ntab, ntab), dtype=torch.float64)

    if h == 0.0:
        raise ValueError("h must be nonzero in ridders_derivative.")

    ftp = ft.clone()
    ftp  = ftp+hh
    ftm = ft.clone()
    ftm = ftm - hh
    yfp , _   = calc_yf(x0, y0, ftp, params, nparams_step, atol, rtol)
    yfm , _   = calc_yf(x0, y0, ftm, params, nparams_step, atol, rtol)
    a[0, 0]   = (yfp[:,ii]-yfm[:,ii]) / (2.0 * hh)
    err = big

    for i in range(1, ntab):
        hh /= con
        ftp = ft.clone()
        ftp = ftp + hh
        ftm = ft.clone()
        ftm = ftm - hh
        yfp, _ = calc_yf(x0, y0, ftp, params, nparams_step, atol, rtol)
        yfm, _ = calc_yf(x0, y0, ftm, params, nparams_step, atol, rtol)
        a[0, i] = (yfp[:,ii] - yfm[:,ii]) / (2.0 * hh)
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






dx1dx1err = []
dx1dx2err = []
dx1dx3err = []
dx2dx1err = []
dx2dx2err = []
dx2dx3err = []
dx1dfterr = []
dx2dfterr = []
dx3dx1err = []
dx3dx2err = []
dx3dx3err = []
dx1dx1 = []
dx1dx2 = []
dx1dx3 = []
dx2dx1 = []
dx2dx2 = []
dx2dx3 = []
dx3dx1 = []
dx3dx2 = []
dx3dx3 = []
params = (torch.ones((M, N), dtype=torch.float64, device=device)-0.5)*torch.rand((1))*10.0
nparams_step = 1
for i in range(100):
    for ti in range(0, 1, 1):
        x0 = torch.zeros([M,1], dtype=torch.float64, device=device)  # Starting point
        #The sensitivity here is wrt to the initial conditions
        y0 = torch.ones([M,3], dtype=torch.float64, device=device)+\
             torch.rand([M,3], dtype=torch.float64, device=device)*0.01
        #y0=torch.tensor([[ 0.1015910265541602, -1.8354477614271085]], dtype=torch.float64)
        #y0[:,0] = -0.4
        #y0[:,1] = -0.1
         #TensorDual.createZero(torch.rand([M,2], dtype=torch.float64, device=device), N)*0.01
        #y0[:, 0] = torch.abs(torch.rand(1, dtype=torch.float64))
        #y0[:, 1] = torch.abs(torch.rand(1, dtype=torch.float64))
        print(f'Running sample i={i}')
        #if i < 8:
        #    continue
        y0c =y0.clone()
        ft = torch.tensor([2.0], dtype=torch.float64, device=device)
        yf, yfsens = calc_yf(x0, y0c, ft, params, nparams_step, atol, rtol)
        dx1dx1.append(yfsens[0, 0, 0])
        dx1dx2.append(yfsens[0, 0, 1])
        dx1dx3.append(yfsens[0, 0, 2])
        dx2dx1.append(yfsens[0, 1, 0])
        dx2dx2.append(yfsens[0, 1, 1])
        dx2dx3.append(yfsens[0, 1, 2])
        dx3dx1.append(yfsens[0, 2, 0])
        dx3dx2.append(yfsens[0, 2, 1])
        dx3dx3.append(yfsens[0, 2, 2])
        print(f'yf={yf}')
        jac00, errjac00= ridders_jac(x0, y0c,  ft, params, nparams_step, atol, rtol,  0, 0)
        print(f'for i ={i} at pos=0,0 dimension 0 dy1fdy1={jac00} versus {yfsens[0,0,0]} for y0={y0c} and ft={ft}')
        #assert torch.allclose(jac00, yfsens[0, 0, 0])
        dx1dx1err.append(torch.abs(jac00-yfsens[0, 0, 0]))
        jac01, errjac01 = ridders_jac(x0, y0c, ft, params, nparams_step, atol, rtol, 0, 1)
        print(f'for i ={i} at pos= 0,1 dimension 1 dy2fdy1={jac01} versus {yfsens[0,0, 1]} for y0={y0c} and ft={ft}')
        #assert torch.allclose(jac01, yfsens[0, 0, 1])
        dx1dx2err.append(torch.abs(jac01-yfsens[0, 0, 1]))
        jac03, errjac03 = ridders_jac(x0, y0c, ft, params, nparams_step, atol, rtol, 0, 2)
        print(f'for i ={i} at pos= 0,3 dimension 3 dy4fdy1={jac03} versus {yfsens[0,0, 3]} for y0={y0c} and ft={ft}')
        #assert torch.allclose(jac01, yfsens[0, 0, 1])
        dx1dx3err.append(torch.abs(jac03-yfsens[0,0,2]))

        jac10, errjac10 = ridders_jac(x0, y0c, ft, params, nparams_step, atol, rtol,  1, 0)
        print(f'for i ={i} at pos 1,0 dimension 0 dy2fdy1={jac10} versus {yfsens[0,1,0]} for y0={y0c} and ft={ft}')
        #assert torch.allclose(jac10, yfsens[0, 1, 0])
        dx2dx1err.append(torch.abs(jac10-yfsens[0, 1, 0]))

        jac11, errjac11 = ridders_jac(x0, y0c, ft, params, nparams_step, atol, rtol, 1, 1)
        print(f'for i ={i} at pos= 1,1 dimension 1 dy2fdy2={jac11} versus {yfsens[0,1, 1]} for y0={y0c} and ft={ft}')
        #assert torch.allclose(jac11, yfsens[0, 1, 1])
        dx2dx2err.append(torch.abs(jac11-yfsens[0, 1, 1]))

        jac12, errjac12 = ridders_jac(x0, y0c, ft, params, nparams_step, atol, rtol, 1, 2)
        print(f'for i ={i} at pos= 1,2 dimension 2 dy2fdy3={jac12} versus {yfsens[0,1, 2]} for y0={y0c} and ft={ft}')
        #assert torch.allclose(jac11, yfsens[0, 1, 1])
        dx2dx3err.append(torch.abs(jac12-yfsens[0, 1, 2]))

        jac20, errjac20 = ridders_jac(x0, y0c, ft, params, nparams_step, atol, rtol, 2, 0)
        print(f'for i ={i} at pos= 2,0 dimension 0 dy3fdy1={jac20} versus {yfsens[0,2, 0]} for y0={y0c} and ft={ft}')
        #assert torch.allclose(jac20, yfsens[0, 2, 0])
        dx3dx1err.append(torch.abs(jac20-yfsens[0,2, 0]))


        jac21, errjac21 = ridders_jac(x0, y0c, ft, params, nparams_step, atol, rtol, 2, 1)
        print(f'for i ={i} at pos= 2,1 dimension 1 dy3fdy2={jac21} versus {yfsens[0,2, 1]} for y0={y0c} and ft={ft}')
        #assert torch.allclose(jac21, yfsens[0, 2, 1])
        dx3dx2err.append(torch.abs(jac21-yfsens[0,2, 1]))

        jac22, errjac22 = ridders_jac(x0, y0c, ft, params, nparams_step, atol, rtol, 2, 2)
        print(f'for i ={i} at pos= 2,2 dimension 2 dy3fdy3={jac22} versus {yfsens[0,2, 2]} for y0={y0c} and ft={ft}')
        #assert torch.allclose(jac22, yfsens[0, 2, 2])
        dx3dx3err.append(torch.abs(jac22-yfsens[0,2, 2]))

        #dy0dft = ridders_jac(x0, y0c, ft,params, nparams_step, 0, atol, rtol)
        #print(f'for i ={i} at pos=0  dy1fdft={dy0dft} versus {yfsens[0, 0, 2]} for y0={y0c} and ft={ft}')
        #assert torch.allclose(jac11, yfsens[0, 1, 1])
        #dx1dfterr.append(torch.abs(dy0dft-yfsens[0, 0, -1]))

        #dy1dft = ridders_jac(x0, y0c, ft, params, nparams_step,1, atol, rtol)
        #print(f'for i ={i} at pos=0  dy2fdft={dy1dft} versus {yfsens[0, 1, 2]} for y0={y0c} and ft={ft}')
        #assert torch.allclose(jac11, yfsens[0, 1, 1])
        #dx2dfterr.append(torch.abs(dy1dft-yfsens[0, 1, -1]))



    fig = plt.figure()
    plt.plot(dx1dx1err, label='dx1dx1err')
    plt.plot(dx1dx2err, label='dx1dx2err')
    plt.plot(dx1dx3err, label='dx1dx3err')
    plt.plot(dx2dx1err, label='dx2dx1err')
    plt.plot(dx2dx2err, label='dx2dx2err')
    plt.plot(dx2dx3err, label='dx2dx3err')
    plt.plot(dx3dx1err, label='dx3dx1err')
    plt.plot(dx3dx2err, label='dx3dx2err')
    plt.plot(dx3dx3err, label='dx3dx3err')

    #plt.plot(dx1dfterr, label='dx1dfterr')
    #plt.plot(dx2dfterr, label='dx2dfterr')
    plt.legend()
    plt.title('Dubins vehicle sensitivity error')
    plt.savefig('dxerr_dubins.png')
    plt.close(fig)

    fig = plt.figure()
    plt.plot(dx1dx1, label='dx1dx1')
    plt.plot(dx1dx2, label='dx1dx2')
    plt.plot(dx1dx3, label='dx1dx3')
    plt.plot(dx2dx1, label='dx2dx1')
    plt.plot(dx2dx2, label='dx2dx2')
    plt.plot(dx2dx3, label='dx2dx3')
    plt.legend()
    plt.savefig('dx_dubins.png')
    plt.close(fig)


end = time.time()
#print(f'Time taken {end-now}')
#print(f'Total number of steps {count}')
exit(1)


plt.rcParams['agg.path.chunksize'] = 10
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
