import torch


from tensormatdual import TensorMatDual
from tensordual import TensorDual

from qrted import QRDual
import copy
torch.set_printoptions(precision=16)
torch.set_num_threads(torch.get_num_threads())

class SeulexTedState:
    def __init__(self, yin, xin, params, nparams_step, atoll, rtoll, KMAXX=13):
        assert isinstance(yin, torch.Tensor)
        assert isinstance(xin, torch.Tensor)
        self.M, self.D = yin.shape
        self.nparams_step = nparams_step

        if params.dim() == 1:
            #Replicate the parameters for each sample
            params= params.repeat(self.M, 1)
        self.N = self.D + params.shape[1]+1 #The final time is a special parameter
        self.params = TensorDual(params, torch.zeros((self.M, params.shape[1], self.N), dtype=params.dtype, device=params.device))
        for i in range(params.shape[1]):
            self.params.d[:, i, self.D+i] = 1.0
        dualy = torch.zeros((self.M, self.D, self.N), dtype=yin.dtype, device=yin.device)
        dualy[:, torch.arange(self.D), torch.arange(self.D)] = 1.0

        realy = torch.zeros((self.M, self.D), dtype=yin.dtype, device=yin.device)
        realy[:, 0:self.D] = yin
        dualx = torch.zeros((self.M, 1, self.N), dtype=yin.dtype, device=yin.device)
        y = TensorDual(realy, dualy)
        x = TensorDual(xin, dualx)
        if not isinstance(atoll, torch.Tensor):
            self.atol = torch.tensor(atoll, dtype=y.r.dtype, device=y.r.device)
        else:
            self.atol = atoll
        if not isinstance(rtoll, torch.Tensor):
            self.rtol = torch.tensor(rtoll, dtype=y.r.dtype, device=y.r.device)
        else:
            self.rtol = rtoll
        self.x     = x.clone()
        self.y     = y.clone()
        self.count = 0
        self.KMAXX = KMAXX
        self.IMAXX = self.KMAXX + 1
        self.nseq = torch.zeros([self.M, self.IMAXX,], dtype=torch.int64, device=y.device)
        self.cost = TensorDual.createZero(torch.zeros([self.M, self.IMAXX,], dtype=y.r.dtype, device=y.device), y.d.shape[2])
        self.table = TensorMatDual.createZero(torch.zeros((self.M, self.IMAXX, self.D), dtype=y.r.dtype, device=y.r.device), y.d.shape[2])
        self.dfdy = TensorMatDual.createZero(torch.zeros((self.M, self.D, self.D), dtype=y.dtype, device=y.device), y.d.shape[2])
        self.calcjac = torch.tensor(self.M*[False], device=y.device)
        self.coeff = TensorMatDual.createZero(torch.zeros((self.M, self.IMAXX, self.IMAXX), dtype=y.dtype, device=y.device), y.d.shape[2])
        self.costfunc = TensorDual.createZero(torch.ones((self.M,1), dtype=y.dtype, device=y.device), y.d.shape[2])
        self.costjac = TensorDual.createZero(torch.ones((self.M,1), dtype=y.dtype, device=y.device)*5.0, y.d.shape[2])
        self.costlu = TensorDual.createZero(torch.ones((self.M,1), dtype=y.dtype, device=y.device), y.d.shape[2])
        self.costsolv = TensorDual.createZero(torch.ones((self.M,1), dtype=y.dtype, device=y.device), y.d.shape[2])
        self.EPS = torch.tensor(torch.finfo(y.dtype).eps).to(y.device)
        self.jac_redo = TensorDual.createZero(torch.ones((self.M, 1), dtype=y.dtype, device=y.device)*torch.min(torch.tensor(1.0e-4), self.rtol), y.d.shape[2])
        self.theta = 2.0*self.jac_redo
        self.one = torch.ones((self.M, 1), dtype=torch.float64, device=y.device)
        self.nseq[:, 0] = 2
        self.nseq[:, 1] = 3
        for i in range(2, self.IMAXX):
            self.nseq[:, i] = 2 * self.nseq[:, i - 2]
        #cost[0] = costjac + costlu + nseq[0] * (costfunc + costsolve);
        #for (Int k=0;k < KMAXX;k++)
        #    cost[k + 1] = cost[k] + (nseq[k + 1] - 1) * (costfunc + costsolve) + costlu;
        nseq0 = self.nseq[:, 0]
        costtemp = self.costjac + self.costlu + nseq0* (self.costfunc + self.costsolv)
        self.cost.r[:, 0:1] = costtemp.r
        self.cost.d[:, 0:1] = costtemp.d
        #vectorize this loop
        for k in range(self.KMAXX):
            kp1 = k+1
            costk = TensorDual(self.cost.r[:, k], self.cost.d[:, k])
            nseqkp1 = self.nseq[:, kp1]
            costkp1 = costk + (nseqkp1 - 1)*(self.costfunc + self.costsolv) + self.costlu
            self.cost.r[:, kp1:kp1+1] = costkp1.r
            self.cost.d[:, kp1:kp1+1] = costkp1.d
        self.hnext = TensorDual.createZero(-1.0e99*torch.ones((self.M,1), dtype=y.dtype, device=y.device), y.d.shape[2])
        self.hdid  = TensorDual.createZero(torch.zeros((self.M, 1), dtype=y.dtype, device=y.device), y.d.shape[2])

        self.logfact = -torch.log10(self.rtol + self.atol) * 0.6 + 0.5
        #k_targ depends only on the tolerances which are the same for all samples
        maxt = (torch.max(torch.tensor(1.0, dtype=torch.float64), torch.min(torch.tensor(self.KMAXX - 1), torch.round(self.logfact))).floor().long())
        self.k_targ = torch.ones([self.M],   dtype=torch.int64, device = y.device)*maxt
        #print(f'setting k_targ to {self.k_targ} in constructor')

        for k in range (self.IMAXX):
            for l in range(k):
                ratio = self.nseq[:, k].to(y.r.dtype) / self.nseq[:, l].to(y.r.dtype)
                self.coeff.r[:, k, l] = 1.0/ (ratio-1.0)



        #Override setindex method
    def __setitem__(self, mask, newstate):
        self.y.r[mask] = newstate.y.r.clone()
        self.y.d[mask] = newstate.y.d.clone()

        self.x.r[mask] = newstate.x.r.clone()
        self.x.d[mask] = newstate.x.d.clone()
        self.count     = newstate.count

        self.atol = newstate.atol
        self.rtol = newstate.rtol
        self.KMAXX = newstate.KMAXX
        self.IMAXX = newstate.IMAXX
        self.nseq[mask] = newstate.nseq.clone()

        self.cost.r[mask] = newstate.cost.r.clone()
        self.cost.d[mask] = newstate.cost.d.clone()

        self.table.r[mask] = newstate.table.r.clone()
        self.table.d[mask] = newstate.table.d.clone()

        self.dfdy.r[mask] = newstate.dfdy.r.clone()
        self.dfdy.d[mask] = newstate.dfdy.d.clone()

        self.calcjac[mask] = newstate.calcjac.clone()

        self.coeff.r[mask] = newstate.coeff.r.clone()
        self.coeff.d[mask] = newstate.coeff.d.clone()

        self.costfunc.r[mask] = newstate.costfunc.r.clone()
        self.costfunc.d[mask] = newstate.costfunc.d.clone()

        self.costjac.r[mask] = newstate.costjac.r.clone()
        self.costjac.d[mask] = newstate.costjac.d.clone()

        self.costlu.r[mask] = newstate.costlu.r.clone()
        self.costlu.d[mask] = newstate.costlu.d.clone()

        self.costsolv.r[mask] = newstate.costsolv.r.clone()
        self.costsolv.d[mask] = newstate.costsolv.d.clone()

        self.EPS = newstate.EPS

        self.jac_redo.r[mask] = newstate.jac_redo.r.clone()
        self.jac_redo.d[mask] = newstate.jac_redo.d.clone()

        self.theta.r[mask] = newstate.theta.r.clone()
        self.theta.d[mask] = newstate.theta.d.clone()

        self.one = newstate.one


        self.hnext.r[mask] = newstate.hnext.r.clone()
        self.hnext.d[mask] = newstate.hnext.d.clone()

        self.hdid.r[mask] = newstate.hdid.r.clone()
        self.hdid.d[mask] = newstate.hdid.d.clone()

        self.logfact = newstate.logfact

        self.k_targ[mask] = newstate.k_targ.clone()

        self.coeff.r[mask] = newstate.coeff.r.clone()
        self.coeff.d[mask] = newstate.coeff.d.clone()

        self.M = self.y.r.shape[0]

        self.count = newstate.count



    def __getitem__(self, mask):
        yr = self.y.r[mask].clone()
        yd = self.y.d[mask].clone()
        y = TensorDual(yr, yd)
        xr = self.x.r[mask].clone()
        xd = self.x.d[mask].clone()
        x = TensorDual(xr, xd)
        newState = SeulexTedState(y, x, self.atol, self.rtol, self.KMAXX)
        newState.count = self.count
        newState.nseq = self.nseq[mask].clone()
        newState.cost.r = self.cost.r[mask].clone()
        newState.cost.d = self.cost.d[mask].clone()
        newState.table.r = self.table.r[mask].clone()
        newState.table.d = self.table.d[mask].clone()
        newState.dfdy.r = self.dfdy.r[mask].clone()
        newState.dfdy.d = self.dfdy.d[mask].clone()
        newState.calcjac = self.calcjac[mask].clone()
        newState.coeff.r = self.coeff.r[mask].clone()
        newState.coeff.d = self.coeff.d[mask].clone()
        newState.costfunc.r = self.costfunc.r[mask].clone()
        newState.costfunc.d = self.costfunc.d[mask].clone()
        newState.costjac.r = self.costjac.r[mask].clone()
        newState.costjac.d = self.costjac.d[mask].clone()
        newState.costlu.r = self.costlu.r[mask].clone()
        newState.costlu.d = self.costlu.d[mask].clone()
        newState.costsolv.r = self.costsolv.r[mask].clone()
        newState.costsolv.d = self.costsolv.d[mask].clone()
        newState.jac_redo.r = self.jac_redo.r[mask].clone()
        newState.jac_redo.d = self.jac_redo.d[mask].clone()
        newState.theta.r = self.theta.r[mask].clone()
        newState.theta.d = self.theta.d[mask].clone()
        newState.hnext.r = self.hnext.r[mask].clone()
        newState.hnext.d = self.hnext.d[mask].clone()
        newState.hdid.r = self.hdid.r[mask].clone()
        newState.hdid.d = self.hdid.d[mask].clone()
        newState.k_targ = self.k_targ[mask].clone()
        newState.coeff.r = self.coeff.r[mask].clone()
        newState.coeff.d = self.coeff.d[mask].clone()

        newState.M = newState.y.r.size()[0]
        newState.count = self.count
        return newState

    def reset_init(self):
        self.x.d *=0.0
        self.coeff.d *=0.0
        self.cost.d *=0.0
        self.costfunc.d *=0.0
        self.costjac.d *=0.0
        self.costlu.d*=0.0
        self.costsolv.d *=0.0
        self.dfdy.d*=0.0
        self.jac_redo.d*=0.0
        self.table.d*=0.0
        self.theta.d*=0.0
        self.hnext = TensorDual.createZero(-1.0e99*torch.ones((self.M,1), dtype=self.y.dtype, device=self.y.device), self.y.d.shape[2])
        self.hdid  = self.hdid*0.0
        dualy = torch.zeros((self.M, self.D, self.N), dtype=self.y.dtype, device=self.y.device)
        dualy[:, torch.arange(self.D), torch.arange(self.D)] = 1.0
        self.y.d = dualy
        self.k_targ = torch.ones([self.M], dtype=torch.int64, device=self.y.device)*self.KMAXX



class SeulexTed:
    def __init__(self):
        pass
    #htry is in [M,]
    #dyns is in [M, D]
    #sate is a SeulexState and is expected to be filters so that the stapsize is non zero

    def step(self, htry, dyns, ftin, state: SeulexTedState) -> SeulexTedState:
        #y_dual = state.y.d[:, :state.D, :state.D].clone()  # Make a copy of the dual part of the state
        #state.reset_init()
        self._step(htry, dyns, ftin, state)
        #state.y.d = torch.einsum('mij, mjk -> mik', y_dual, state.y.d.clone() )  # Multiply the dual part by the original state

    def _step(self, htry, dyns, ftin, state : SeulexTedState)->SeulexTedState:
        STEPFAC1 = TensorDual(torch.tensor(0.6, dtype=torch.float64, device=state.y.device), state.x.d*0.0)
        STEPFAC2 = TensorDual(torch.tensor(0.93, dtype=torch.float64, device=state.y.device), state.x.d*0.0)
        STEPFAC3 = TensorDual(torch.tensor(0.1, dtype=torch.float64, device=state.y.device), state.x.d*0.0)
        STEPFAC4 = TensorDual(torch.tensor(4.0, dtype=torch.float64, device=state.y.device), state.x.d*0.0)
        STEPFAC5 = TensorDual(torch.tensor(0.5, dtype=torch.float64, device=state.y.device), state.x.d*0.0)
        KFAC1    = TensorDual(torch.tensor(0.7, dtype=torch.float64, device=state.y.device), state.x.d*0.0)
        KFAC2    = TensorDual(torch.tensor(0.9, dtype=torch.float64, device=state.y.device), state.x.d*0.0)
        firstk = torch.BoolTensor(state.x.r.shape[0]).fill_(True).to(device=state.y.r.device)
        firstk = torch.BoolTensor(state.x.r.shape[0]).fill_(True).to(device=state.y.r.device)
        if not isinstance(htry, TensorDual):
            htry = TensorDual.createZero(htry, state.y.d.shape[2])
        else:
            htry.d *=0.0
        if not isinstance(ftin, TensorDual):
            ft = TensorDual.createZero(ftin,   state.y.d.shape[2])
            ft.d[:,:,-1] = 1.0 #Assumes that the final sensitivity is wrt the final time


        #state.reset_init(paramsloc, ft) #reset the initial conditions so that the sensitivities are calculated wrt the initial conditions
        #adjust the step size if necessary to meet the terminal time
        if (ft -state.x) < htry:
            htry = ft - state.x
        state.count = state.count+1
        assert state.count < state.N, f'count = {state.count} is greater than N = {state.N}'

        hopt = TensorDual.createZero(torch.zeros((state.M, state.IMAXX), dtype=state.y.r.dtype, device=state.y.r.device), state.y.d.shape[2])
        work = TensorDual.createZero(torch.zeros((state.M, state.IMAXX), dtype=state.y.r.dtype, device=state.y.r.device), state.y.d.shape[2])
        fac  = TensorDual.createZero(torch.zeros((state.M, 1), dtype=state.y.r.dtype, device=state.y.r.device), state.y.d.shape[2])
        first_step= torch.tensor([True]*state.M, device=state.y.device)
        last_step = torch.tensor([False]*state.M, device=state.y.device)
        reject = torch.tensor([False]*state.M, device=state.y.device)
        prev_reject = torch.tensor([False]*state.M, device=state.y.device)
        errold = TensorDual.createZero(torch.zeros((state.M, 1), dtype=state.y.r.dtype, device=state.y.r.device), state.y.d.shape[2])

        work.r[:, 0:1] = 1.0e30
        h = htry.clone()
        ysav = state.y.clone()
        forward= torch.tensor([False]*state.M, device=state.y.device)

        forward[(h > 0).all()] = True
        forward[(h <= 0).all()] = False

        if ((h != state.hnext) & ~ first_step).any():
            mask = (h != state.hnext) & ~ state.first_step
            state.last_step[mask]   = True

        if reject.any():
            last_step[state.reject]   = False
            prev_reject[state.reject] = True
            thetatemp = 2.0 * state.jac_redo[state.reject]
            state.theta.r[state.reject]=thetatemp.r
            state.theta.d[state.reject]=thetatemp.d

        scale        = state.atol + state.rtol * abs(state.y)

        hnew = abs(h)
        #if torch.allclose(hnew, torch.tensor([0.660544, 0.168978]), atol=1e-6):
        #    print(f'hnew is close to {hnew}')

        #print(f'hnew = {hnew} set in step')
        #k is a counter used internally by compute_jac
        k=torch.zeros([state.M,], dtype=torch.int64, device=state.y.device)
        #Force calculation of the jacobian

        def compute_step():
            print(f'compute_step called')
            #This is a break mask.  If it is False it means that the loop should be broken
            #The state passed in ensures that the samples do not contain any zero step sizes
            bmask = torch.BoolTensor(state.y.r.shape[0]).fill_(True).to(device=state.y.device)
            while ((firstk | reject) ).any():
                m0 = (firstk | reject)
                #print(f'm0 = {m0} set in compute_jac')
                m1 = (state.theta > state.jac_redo) & (~state.calcjac) & (abs(h) > state.EPS) & m0
                #print(f'm1 = {m1} set in compute_jac')
                if torch.any(m1):
                    print(f'm1 = {m1} set in compute_jac')
                    #print(f'state.dfdy.shape = {state.dfdy.shape}')
                    #print(f'state.x.shape = {state.x.shape}')
                    #print(f'state.y.shape = {state.y.shape}')
                    xm1 = TensorDual(state.x.r[m1], state.x.d[m1])
                    ym1 = TensorDual(state.y.r[m1], state.y.d[m1])
                    paramsm1 = TensorDual(state.params.r[m1, (state.count-1)*state.nparams_step:state.count*state.nparams_step], \
                                            state.params.d[m1, (state.count-1)*state.nparams_step:state.count*state.nparams_step])
                    dfdym1 = dyns.jacobian(xm1, ym1, paramsm1)
                    state.dfdy.r[m1] = dfdym1.r
                    state.dfdy.d[m1] = dfdym1.d
                    state.calcjac[m1] = torch.tensor(True).to(device=state.y.device)





                m2 = (firstk | reject)  & bmask & m0



                firstk[m2]= False
                reject[m2]= False

                #if (torch.abs(h) <= torch.abs(self.x)*self.EPS).any():
                #    raise ValueError('Step size too small in seulex')
                k[m2] = 0
                bmask[:] = True #Reset the break mask for the next loop
                while (m2[m2] & (k[m2]  <= (state.k_targ[m2]+1)) & bmask[m2]).any():
                    if (~bmask).all():
                        break

                    #print(f'bmask={bmask}')
                    m21 = m2 & bmask & (k <= (state.k_targ+1))
                    if m21.any():
                            #    def dy(state, y, x, htot, k, scale, theta, dfdy, dyns):

                            #if torch.allclose(state.x[m21], torch.tensor(9.9346237143, dtype=torch.float64), atol=1.0e-6):
                            #    print(f'x is close to {state.x}')
                            #The TensorDual is just a thin wrapper around the actual tensor
                            ysavtemp = TensorDual(ysav.r[m21], ysav.d[m21])
                            xtemp = TensorDual(state.x.r[m21], state.x.d[m21])
                            hnewtemp = TensorDual(hnew.r[m21], hnew.d[m21])
                            km21 = k[m21]
                            nseqm21 = state.nseq[m21, :]
                            scalem21 = TensorDual(scale.r[m21], scale.d[m21])
                            thetam21 = TensorDual(state.theta.r[m21], state.theta.d[m21])
                            dfdym21 = TensorMatDual(state.dfdy.r[m21], state.dfdy.d[m21])

                            #Test the dy_dual function
                            paramoff = (state.count-1)*state.nparams_step
                            paramsm21 = TensorDual(state.params.r[m21, paramoff:paramoff+1], \
                                                   state.params.d[m21, paramoff:paramoff+1, :])
                            success, thetares, yseq = self.stability_check(ysavtemp, xtemp, hnewtemp, km21, \
                                                                           nseqm21, scalem21, \
                                                                           thetam21, dfdym21, paramsm21, dyns)




                            def test_dy_dual():
                                M, D, N   = ysavtemp.d.shape
                                ysavtempc = ysavtemp.clone()
                                xtempc    = xtemp.clone()
                                hnewtempc = hnewtemp.clone()
                                km21c = km21.clone()
                                scalem21c = scalem21.clone()

                                thetam21c = thetam21.clone()
                                dfdym21c  = dfdym21.clone() #This is the jacobian requires some care
                                #We need to create a dual that has a 1 for all the input
                                #Scaele is very hard to test because of the very small values involved
                                NN = ysavtempc.d.shape[1]+xtempc.d.shape[1]+hnewtempc.d.shape[1]+\
                                     thetam21c.d.shape[1]+dfdym21c.d.shape[1]*dfdym21c.d.shape[2]
                                count = 0
                                ysavtempc.d = torch.zeros([M, ysavtemp.d.shape[1], NN], dtype=ysavtempc.d.dtype, device=ysavtempc.d.device)
                                for i in range(ysavtemp.d.shape[1]):
                                    ysavtempc.d[:, i, count] = 1.0
                                    count = count + 1
                                xtempc.d    = torch.zeros([M, xtemp.d.shape[1],    NN], dtype=ysavtempc.d.dtype, device=ysavtempc.d.device)
                                for i in range(xtemp.d.shape[1]):
                                    xtempc.d[:, i, count] = 1.0
                                    count = count + 1
                                hnewtempc.d = torch.zeros([M, hnewtemp.d.shape[1], NN], dtype=ysavtempc.d.dtype, device=ysavtempc.d.device)
                                for i in range(hnewtemp.d.shape[1]):
                                    hnewtempc.d[:, i, count] = 1.0
                                    count = count + 1
                                thetam21c.d = torch.zeros([M, thetam21.d.shape[1], NN], dtype=ysavtempc.d.dtype, device=ysavtempc.d.device)
                                for i in range(thetam21.d.shape[1]):
                                    thetam21c.d[:, i, count] = 1.0
                                    count = count + 1
                                dfdym21c.d  = torch.zeros([M, dfdym21.d.shape[1], dfdym21.d.shape[2], NN], dtype=ysavtempc.d.dtype, device=ysavtempc.d.device)
                                for i in range(dfdym21.d.shape[1]):
                                    for j in range(dfdym21.d.shape[2]):
                                        dfdym21c.d[:, i, j, count] = 1.0
                                        count = count + 1
                                #Set the scale dependencies to zero
                                scalem21c.d = torch.zeros([M, scalem21.d.shape[1], NN], dtype=ysavtempc.d.dtype, device=ysavtempc.d.device)


                                successc, thetaresc, yseqc = self.stability_check(ysavtempc, xtempc, hnewtempc, km21c, \
                                                                                  nseqm21, scalem21c, \
                                                                                  thetam21c, dfdym21c, dyns)
                                Jthetaresc = thetaresc.square().sum()
                                Jyseqc = yseqc.square().sum()
                                #Test changes in ysavtempc
                                for i in range(NN):

                                    dx = 1.0e-8
                                    ysavtempc2p = ysavtempc.clone()
                                    xtempc2p    = xtempc.clone()
                                    hnewtempc2p = hnewtempc.clone()
                                    thetam21c2p = thetam21c.clone()
                                    dfdym21c2p  = dfdym21c.clone() #This is the jacobian requires some care
                                    if i < ysavtempc.d.shape[1]:
                                        ysavtempc2p.r[:, i] = ysavtempc.r[:, i] + 2*dx
                                    elif i < ysavtempc.d.shape[1] + xtempc.d.shape[1]:
                                        dx
                                        idx = i - ysavtempc.d.shape[1]
                                        xtempc2p.r[:, idx] = xtempc.r[:, idx] + 2*dx
                                    elif i < ysavtempc.d.shape[1] + xtempc.d.shape[1] + hnewtempc.d.shape[1]:
                                        idx = i - ysavtempc.d.shape[1] - xtempc.d.shape[1]
                                        hnewtempc2p.r[:, idx] = hnewtempc.r[:, idx] + 2*dx
                                    elif i < ysavtempc.d.shape[1] + xtempc.d.shape[1] + hnewtempc.d.shape[1] +\
                                             thetam21c.d.shape[1]:
                                        idx = i - ysavtempc.d.shape[1] - xtempc.d.shape[1] - hnewtempc.d.shape[1]
                                        thetam21c2p.r[:, idx] = thetam21c.r[:, idx] + 2*dx
                                    elif i < ysavtempc.d.shape[1] + xtempc.d.shape[1] + hnewtempc.d.shape[1] +\
                                             thetam21c.d.shape[1]  + dfdym21c.d.shape[1]*dfdym21c.d.shape[2]:
                                        idx = i - ysavtempc.d.shape[1] - xtempc.d.shape[1] - hnewtempc.d.shape[1] - \
                                              thetam21c.d.shape[1]
                                        nj = idx%D #Column index
                                        ni = idx//D #Row index
                                        dfdym21c2p.r[:, ni, nj] = dfdym21c.r[:, ni, nj] + 2*dx
                                    successc2p, thetaresc2p, yseqc2p = self.stability_check(ysavtempc2p, xtempc2p, hnewtempc2p, km21c, \
                                                                                            nseqm21, scalem21c, \
                                                                                            thetam21c2p, dfdym21c2p, dyns)

                                    Jthetaresc2p = thetaresc2p.square().sum()
                                    Jyseqc2p = yseqc2p.square().sum()


                                    ysavtempcp = ysavtempc.clone()
                                    xtempcp    = xtempc.clone()
                                    hnewtempcp = hnewtempc.clone()
                                    thetam21cp = thetam21c.clone()
                                    dfdym21cp  = dfdym21c.clone() #This is the jacobian requires some care
                                    if i < ysavtempc.d.shape[1]:
                                        ysavtempcp.r[:, i] = ysavtempc.r[:, i] + dx
                                    elif i < ysavtempc.d.shape[1] + xtempc.d.shape[1]:
                                        idx = i - ysavtempc.d.shape[1]
                                        xtempcp.r[:, idx] = xtempc.r[:, idx] + dx
                                    elif i < ysavtempc.d.shape[1] + xtempc.d.shape[1] + hnewtempc.d.shape[1]:
                                        idx = i - ysavtempc.d.shape[1] - xtempc.d.shape[1]
                                        hnewtempcp.r[:, idx] = hnewtempc.r[:, idx] + dx
                                    elif i < ysavtempc.d.shape[1] + xtempc.d.shape[1] + hnewtempc.d.shape[1] +\
                                             thetam21c.d.shape[1]:
                                        idx = i - ysavtempc.d.shape[1] - xtempc.d.shape[1] - hnewtempc.d.shape[1]
                                        thetam21cp.r[:, idx] = thetam21c.r[:, idx] + dx
                                    elif i < ysavtempc.d.shape[1] + xtempc.d.shape[1] + hnewtempc.d.shape[1] +\
                                             thetam21c.d.shape[1]  + dfdym21c.d.shape[1]*dfdym21c.d.shape[2]:
                                        idx = i - ysavtempc.d.shape[1] - xtempc.d.shape[1] - hnewtempc.d.shape[1] - \
                                              thetam21c.d.shape[1]
                                        nj = idx%D #Column index
                                        ni = idx//D #Row index
                                        dfdym21cp.r[:, ni, nj] = dfdym21c.r[:, ni, nj] + dx
                                    successcp, thetarescp, yseqcp = self.stability_check(ysavtempcp, xtempcp, hnewtempcp, km21c, \
                                                                                         nseqm21, scalem21c, \
                                                                                         thetam21cp, dfdym21cp, dyns)

                                    Jthetarescp = thetarescp.square().sum()
                                    Jyseqcp = yseqcp.square().sum()

                                    ysavtempcm = ysavtempc.clone()
                                    xtempcm    = xtempc.clone()
                                    hnewtempcm = hnewtempc.clone()
                                    thetam21cm = thetam21c.clone()
                                    dfdym21cm  = dfdym21c.clone() #This is the jacobian requires some care
                                    if i < ysavtempc.d.shape[1]:
                                        ysavtempcm.r[:, i] = ysavtempc.r[:, i] - dx
                                    elif i < ysavtempc.d.shape[1] + xtempc.d.shape[1]:
                                        idx = i - ysavtempc.d.shape[1]
                                        xtempcm.r[:, idx] = xtempc.r[:, idx] - dx
                                    elif i < ysavtempc.d.shape[1] + xtempc.d.shape[1] + hnewtempc.d.shape[1]:
                                        idx = i - ysavtempc.d.shape[1] - xtempc.d.shape[1]
                                        hnewtempcm.r[:, idx] = hnewtempc.r[:, idx] - dx
                                    elif i < ysavtempc.d.shape[1] + xtempc.d.shape[1] + hnewtempc.d.shape[1] +\
                                            thetam21c.d.shape[1]:
                                        idx = i - ysavtempc.d.shape[1] - xtempc.d.shape[1] - hnewtempc.d.shape[1]
                                        thetam21cm.r[:, idx] = thetam21c.r[:, idx] - dx
                                    elif i < ysavtempc.d.shape[1] + xtempc.d.shape[1] + hnewtempc.d.shape[1] + \
                                             thetam21c.d.shape[1]  + dfdym21c.d.shape[1]*dfdym21c.d.shape[2]:
                                        idx = i - ysavtempc.d.shape[1] - xtempc.d.shape[1] - hnewtempc.d.shape[1] -\
                                              thetam21c.d.shape[1]

                                        nj = idx%D #Column index
                                        ni = idx//D #Row index
                                        dfdym21cm.r[:, ni, nj] = dfdym21c.r[:, ni, nj] - dx


                                    successcm, thetarescm, yseqcm = self.stability_check(ysavtempcm, xtempcm, hnewtempcm, km21c, \
                                                                                         nseqm21, scalem21c, \
                                                                                         thetam21cm, dfdym21cm, dyns)
                                    Jthetarescm  = thetarescm.square().sum()
                                    Jyseqcm      = yseqcm.square().sum()



                                    ysavtempc2m = ysavtempc.clone()
                                    xtempc2m    = xtempc.clone()
                                    hnewtempc2m = hnewtempc.clone()
                                    thetam21c2m = thetam21c.clone()
                                    dfdym21c2m  = dfdym21c.clone() #This is the jacobian requires some care
                                    if i < ysavtempc.d.shape[1]:
                                        ysavtempc2m.r[:, i] = ysavtempc.r[:, i] - 2*dx
                                    elif i < ysavtempc.d.shape[1] + xtempc.d.shape[1]:
                                        idx = i - ysavtempc.d.shape[1]
                                        xtempc2m.r[:, idx] = xtempc.r[:, idx] - 2*dx
                                    elif i < ysavtempc.d.shape[1] + xtempc.d.shape[1] + hnewtempc.d.shape[1]:
                                        idx = i - ysavtempc.d.shape[1] - xtempc.d.shape[1]
                                        hnewtempc2m.r[:, idx] = hnewtempc.r[:, idx] - 2*dx
                                    elif i < ysavtempc.d.shape[1] + xtempc.d.shape[1] + hnewtempc.d.shape[1] +\
                                            thetam21c.d.shape[1]:
                                        idx = i - ysavtempc.d.shape[1] - xtempc.d.shape[1] - hnewtempc.d.shape[1]
                                        thetam21c2m.r[:, idx] = thetam21c.r[:, idx] - 2*dx
                                    elif i < ysavtempc.d.shape[1] + xtempc.d.shape[1] + hnewtempc.d.shape[1] + \
                                             thetam21c.d.shape[1]  + dfdym21c.d.shape[1]*dfdym21c.d.shape[2]:
                                        idx = i - ysavtempc.d.shape[1] - xtempc.d.shape[1] - hnewtempc.d.shape[1] -\
                                              thetam21c.d.shape[1]

                                        nj = idx%D #Column index
                                        ni = idx//D #Row index
                                        dfdym21c2m.r[:, ni, nj] = dfdym21c.r[:, ni, nj] - 2*dx


                                    successc2m, thetaresc2m, yseqc2m = self.stability_check(ysavtempc2m, xtempc2m, hnewtempc2m, km21c, \
                                                                                            nseqm21, scalem21c, \
                                                                                            thetam21c2m, dfdym21c2m, dyns)
                                    Jthetaresc2m  = thetaresc2m.square().sum()
                                    Jyseqc2m      = yseqc2m.square().sum()


                                    Jthetarescfd = (-Jthetaresc2p+8*Jthetarescp - 8*Jthetarescm+Jthetaresc2m)/(12.0*dx)
                                    Jyseqcfd     = (-Jyseqc2p +8*Jyseqcp- 8*Jyseqcm+Jyseqc2m)/(12.0*dx)
                                    print(f'i={i}')
                                    for m in range(M):
                                        print(f'sample = {m} Jthetaresc.d[:,0,i] = {Jthetaresc.d[m,0, i]} Jthetarescfd.r[:,i] = {Jthetarescfd.r[m,0]}')
                                        if not torch.allclose(Jthetarescfd.r[m,0], Jthetaresc.d[m,0, i], atol=1.0e-4):
                                            stopHere = True
                                        #assert torch.allclose(Jthetarescfd.r[m,0], Jthetaresc.d[m,0, i], atol=1.0e-4)
                                        print(f'sample = {m} Jyseqc.d[:.i] = {Jyseqc.d[m,0, i]} Jyseqcfd.r[:,0] = {Jyseqcfd.r[m,0]}')
                                        if not torch.allclose(Jyseqc.d[m, 0, i], Jyseqcfd.r[m,0], atol=1.0e-4, rtol=1.0e-4):
                                            stopHere = True
                                        #assert torch.allclose(Jyseqc.d[m, 0, i], Jyseqcfd.r[m,0], atol=1.0e-4, rtol=1.0e-4)

                            #if __debug__:
                            #    if state.count == 11:
                            #        test_dy_dual()


                            state.theta.r[m21] = thetares.r
                            state.theta.d[m21] = thetares.d

                            if torch.isnan(state.theta.d).any():
                                print(f'theta is nan')
                            #success2, theta2, yseq2 = self.dy_serial(ysav[m21, :], state.x[m21], hc[m21], k[m21], \
                            #                                          state.nseq[m21, :],  scale[m21, :], \
                            #                                          state.theta[m21], state.dfdy[m21, :,  :], dyns)
                            #if not torch.allclose(success.cuda(), success2.cuda(), rtol=1e-15, atol=1e-15):
                            #    print(f'{success2} not close to {success}')
                            #if not torch.allclose(state.theta[m21], theta2, rtol=1e-15, atol=1e-15):
                            #    print(f'{state.theta[m21]} not close to {theta2}')
                            #    print(f'state.x={state.x[m21]}')

                            #print(f'state.theta after dy={state.theta}')
                            #print(f'yseq after dy={yseq}')
                            #print(f'success shape={success.shape}')
                            #print(f'k={k}')
                            #print(f'k_targ={state.k_targ}')
                            #Here success is a break mask
                            m211 = m21 & bmask
                            m211[m21]=m211[m21] & ~success
                            if m211.any():
                                print(f'm211 called')
                                hnewtemp   = TensorDual(hnew.r[m211], hnew.d[m211])
                                hnewres    = abs(hnewtemp) * STEPFAC5
                                hnew.r[m211] = hnewres.r
                                hnew.d[m211] = hnewres.d
                                print(f'hnew[m211] = {hnew.r[m211]} {hnew.d[m211]} set in m211')
                                reject[m211] = True
                                bmask[m211] = False
                                print(f'bmask = {bmask} set in m211')
                            m212 = m21 & bmask & (k ==0)
                            if m212.any():
                                print(f'm212 called')
                                state.y.r[m212] = yseq.r[m212[m21]]
                                state.y.d[m212] = yseq.d[m212[m21]]
                            m213 = m21 & bmask
                            m213[m21] = m213[m21] & (k[m21] != 0 & ~success)
                            if m213.any():
                                print(f'm213 called')
                                yseqtemp = TensorDual(yseq.r[m213], yseq.d[m213])
                                state.table.r[m213, k[m213] - 1] = yseqtemp.r
                                state.table.d[m213, k[m213] - 1] = yseqtemp.d
                                km21 = k[m213]
                                ytemp = TensorDual(state.y.r[m213], state.y.d[m213])
                                tabletemp = TensorMatDual(state.table.r[m213], state.table.d[m213])
                                coefftemp = TensorMatDual(state.coeff.r[m213], state.coeff.d[m213])
                                #state.y[m213, :], state.table[m213, :, :] = self.polyextr(k[m213], state.y[m213, :],
                                #                                                          state.table[m213, :, :],
                                #                                                          state.coeff[m213, :, :])
                                yres, tableres = self.extrapolate(km21, ytemp, tabletemp, coefftemp)
                                state.y.r[m213] = yres.r
                                state.y.d[m213] = yres.d
                                state.table.r[m213] = tableres.r
                                state.table.d[m213] = tableres.d
                                #print(f'table: {state.table.r} {state.table.d} set in m213')
                                ysavtemp = TensorDual(ysav.r[m213], ysav.d[m213])
                                scaleres = state.atol + state.rtol * abs(ysavtemp)
                                scale.r[m213] = scaleres.r
                                scale.d[m213] = scaleres.d
                                err = TensorDual.createZero(torch.zeros((state.M, 1), dtype=state.y.r.dtype, device=state.y.r.device), state.y.d.shape[2])
                                #err = torch.sqrt(torch.sum(torch.square((state.y-state.table[0][:]) / scale))/state.n)
                                # err[mask] = torch.sqrt(torch.sum(torch.square((state.y[mask, :] - state.table[mask, 0, :]) / scale[mask, :]), dim=0) / state.D)
                                #print(f'yr={state.y.r[m213]} yd={state.y.d[m213]}')
                                ytemp =TensorDual(state.y.r[m213], state.y.d[m213])
                                tabletemp = TensorDual(state.table.r[m213, 0], state.table.d[m213, 0])
                                delta = (ytemp - tabletemp)
                                #print(f'delta={delta}')
                                scalem21 = TensorDual(scale.r[m213], scale.d[m213])
                                sq_inpt = delta.r*scalem21.reciprocal()
                                #print(f'sq_inpt={sq_inpt}')
                                sq_term = sq_inpt.square()
                                #print(f'sq_term={sq_term}')
                                errres = (sq_term.sum() * torch.tensor(1.0*state.D, dtype=torch.float64).reciprocal()).sqrt()
                                err.r[m213] = errres.r
                                err.d[m213] = errres.d
                                #print(f'err={err}')

                                #print(f'scaler={scale.r[m213]} scaled={scale.d[m213]}')
                                #print(f'ysavr={ysav.r[m213]} ysavd={ysav.d[m213]}')
                                errm213 = TensorDual(err.r[m213], err.d[m213])
                                t1 = errm213 * state.EPS.reciprocal()
                                j2 = errm213
                                # m2131 = m213 & bmask
                                m2131 = m213 & bmask
                                km213 = k[m213]
                                erroldm213 = TensorDual(errold.r[m213], errold.d[m213])
                                m2131[m213] = m2131[m213] & (j2 > t1) | ((km213 > 1) & (errm213 >= erroldm213))
                                if m2131.any():
                                    # print(f'mask before state.reject = {mask}')
                                    reject[m2131] = True
                                    hnewm2131 = TensorDual(hnew.r[m2131], hnew.d[m2131])
                                    hnewm2131res = abs(hnewm2131) * STEPFAC5
                                    hnew.r[m2131] = hnewm2131res.r
                                    hnew.d[m2131] = hnewm2131res.d
                                    print(f'hnewr[m2131] = {hnew.r[m2131]}  hnewd = {hnew.d[m2131]} set in m2131')
                                    bmask[m2131] = False
                                    #print(f'bmask set at the end of m2131 to {bmask}')
                                # bmask has been updated so also update m213
                                m213 = m213 & bmask
                                if m213.any():
                                    print(f'm213 called with m213={m213} and bmask={bmask}')
                                    errm213 = TensorDual(err.r[m213], err.d[m213])
                                    erroldm213res = TensorDual.max_dual(4.0 * errm213, errm213)
                                    errold.r[m213] = erroldm213res.r
                                    errold.d[m213] = erroldm213res.d
                                    expo = state.one*torch.tensor(k.clone().detach()*1.0, dtype=torch.float64).reciprocal()
                                    #print(f'expo={expo}')
                                    facmin = pow(STEPFAC3, expo)
                                    m2132 = m213 & bmask & (err == 0.0)
                                    if m2132.any():
                                        print(f'm2132 called')
                                        facmin2132 = TensorDual(facmin.r[m2132], facmin.d[m2132])
                                        facm2132res = facmin2132.reciprocal()
                                        fac.r[m2132] = facm2132res.r
                                        fac.d[m2132] = facm2132res.d
                                    m2133 = ~m2132 & bmask
                                    if m2133.any():
                                        print(f'm2133 called')
                                        errm2133   = TensorDual(err.r[m2133], err.d[m2133])
                                        expom2133  = expo[m2133]
                                        facmin2133 = TensorDual(facmin.r[m2133], facmin.d[m2133])
                                        #fac[m2133] = STEPFAC2 / torch.pow(err[m2133] / STEPFAC1, expo[m2133])

                                        facm2133 = STEPFAC2 * pow((errm2133 / STEPFAC1), expom2133).reciprocal()
                                        #print(f'fac step 1: {facm2133.r} {facm2133.d} set in m2133')
                                        #fac[m2133] = torch.max(facmin[m2133] / STEPFAC4,
                                        #                       torch.min(facmin[m2133].reciprocal(), fac[m2133]))
                                        facm2133res = TensorDual.max_dual(facmin2133/STEPFAC4,
                                                               TensorDual.min_dual(facmin2133.reciprocal(), facm2133))
                                        fac.r[m2133] = facm2133res.r
                                        fac.d[m2133] = facm2133res.d
                                        #print(f'fac: {fac.r} {fac.d} set in m2133')
                                    #print(f'facmin: {facmin}')
                                    #print(f'hnew: {hnew.r} {hnew.d}')
                                    hnewm213 = TensorDual(hnew.r[m213],   hnew.d[m213])
                                    facm213  = TensorDual(fac.r[m213], fac.d[m213])
                                    #hopt[m213, k[m213]] = torch.abs(hnew[m213] * fac[m213])
                                    hoptres = abs(hnewm213 * facm213)
                                    hopt.r[m213, k[m213]] = torch.squeeze(hoptres.r)
                                    hopt.d[m213, k[m213]] = torch.squeeze(hoptres.d)
                                    #if hopt[m213, k[m213]].shape[0] > 1:
                                    #    if torch.allclose(hopt[m213, k[m213]][1], torch.tensor(0.734853), atol=1e-6):
                                    #        print(f'hopt is close to 0.734853]')

                                    #print(f'hopt[{m213},{k[m213]} ]={hopt.r[m213, k[m213]]} hoptrd={hopt.d[m213, k[m213]]} set in m213')
                                    costm213 = TensorDual(state.cost.r[m213, k[m213]], state.cost.d[m213, k[m213]])
                                    hoptm213 = TensorDual(hopt.r[m213, k[m213]], hopt.d[m213, k[m213]])
                                    workres = (costm213*hoptm213.reciprocal())
                                    work.r[m213, k[m213]]= torch.squeeze(workres.r)
                                    work.d[m213, k[m213]]= torch.squeeze(workres.d)
                                    #print(f'workr={work.r[m213]} workd={work.d[m213]}')
                                    m2134 = m213 & bmask & (first_step | last_step) & (err <= 1.0)
                                    if m2134.any():
                                        print(f'm2134 called')
                                        bmask[m2134] = False
                                        #print(f'bmask set at m2134={bmask}')
                                    m2135 = m213 & bmask & (
                                                    (k == (state.k_targ - 1)) & ~ prev_reject & \
                                                    ~ first_step & ~ last_step)
                                    if m2135.any():
                                        print(f'm2135 called')
                                        m21351 = m2135 & (err <= 1.0) & bmask
                                        if m21351.any():
                                            print(f'm21351 called')
                                            bmask[m21351] = False
                                            #print(f'bmask set at m21351={bmask}')
                                        ktargindx = torch.unsqueeze(state.k_targ,1)
                                        nseqktarg = torch.squeeze(state.nseq.gather(1, ktargindx))
                                        ktargp1indx = torch.unsqueeze(state.k_targ+1,1)
                                        nseqktargp1 = torch.squeeze(state.nseq.gather(1, ktargp1indx))
                                        #err > self.nseq[self.k_targ]*self.nseq[self.k_targ+1]*4.0:
                                        m21352= m2135 & bmask & ((err) > (nseqktarg * nseqktargp1 * 4.0))
                                        if m21352.any():
                                            print(f'm21352 called')
                                            state.reject[m21352] = True
                                            state.k_targ[m21352] = k[m21352]
                                            km1indcs = torch.unsqueeze(k-1,1)
                                            km1indcs[km1indcs < 0] = 0
                                            workkm1 = work.gather(1, km1indcs)
                                            kindcs = torch.unsqueeze(k,1)
                                            kindcs[kindcs < 0] = 0
                                            workk = work.gather(1, kindcs)
                                            #print(f'setting k_targ in m21352 = {state.k_targ[m21352]}')
                                            m213521 =m21352 & bmask & (state.k_targ > 1) & (
                                                    workkm1 < KFAC1 * workk)
                                            if m213521.any():
                                                #print(f'm213521 called')
                                                state.k_targ[m213521] = state.k_targ[m213521] - 1
                                                #print(f'setting k_targ in m213521 = {state.k_targ[m213521]}')
                                                hnew.r[m21352] = hopt.r[m21352, state.k_targ[m21352]]
                                                hnew.d[m21352] = hopt.d[m21352, state.k_targ[m21352]]
                                                #print(f'setting hnew in m21352 = {hnew[m21352]}')
                                                bmask[m21352] = False
                                                #print(f'bmask in m21352 = {bmask}')
                                    m2136 = m213 & bmask & (k == state.k_targ)
                                    if m2136.any():
                                        print(f'm2136 called')
                                        m21361 = m2136 & bmask & (err <= 1.0)
                                        if m21361.any():
                                            print(f'm21361 called')
                                            bmask[m21361] = False
                                            #print(f'bmask in m21361 = {bmask}')
                                        m2136 = m2136 & bmask
                                        if m2136.any():
                                            print(f'm2136 called')
                                            nseq_indcs = torch.unsqueeze(k+1, 1)
                                            temp = torch.squeeze(state.nseq.gather(1,nseq_indcs) * 2.0, 1)
                                            m21362 = m2136 & bmask & (err > temp)
                                            if m21362.any():
                                                print(f'm21362 called')
                                                reject[m21362] = True

                                                km1indcs = torch.unsqueeze(k-1, 1)
                                                km1indcs[km1indcs<0]=0 #This won't matter since the negative indices are filters out
                                                #print(f'km1indcs={km1indcs}')
                                                kindcs = torch.unsqueeze(k, 1)
                                                kindcs[kindcs<0]=0 #This won't matter since the negative indices are filters out
                                                #print(f'kindcs={kindcs}')
                                                wm1 = work.gather(1, km1indcs)
                                                w =   work.gather(1, kindcs)
                                                #  if self.k_targ > 1 and work[k-1] < KFAC1*work[k]:
                                                m213621= m21362 & bmask & (state.k_targ > 1) & (wm1 < (KFAC1 * w))
                                                if m213621.any():
                                                    print(f'm213621 called')
                                                    state.k_targ[m213621] = state.k_targ[m213621] - 1
                                                    #print(f'setting k_targ in m213621 = {state.k_targ[m213621]}')
                                                hnew.r[m21362] = hopt.r[m21362, state.k_targ[m21362]].view(hnew.r[m21362].shape)
                                                hnew.d[m21362] = hopt.d[m21362, state.k_targ[m21362]].view(hnew.d[m21362].shape)
                                                #print(f'setting hnew in m21362 = {hnew.r[m21362]} {hnew.d[m21362]}' )
                                                if m21362.any():
                                                    print(f'm21362 called')
                                                    bmask = bmask & ~m21362
                                                    #print(f'bmask in m21362 = {bmask}')
                                m213 = m213 & bmask  # Check to make sure a break has not been introduced beforhand
                                if m213.any():
                                    print(f'm213 called')
                                    m2137 = m213 & bmask & (k == (state.k_targ + 1))
                                    if m2137.any():
                                        print(f'm2137 called')
                                        m21371 = m2137 & bmask & (err > 1.0)
                                        if m21371.any():
                                            print(f'm21371 called')
                                            reject[m21371] = True
                                            temp = torch.unsqueeze(state.k_targ,1)
                                            #if self.k_targ > 1 and work[self.k_targ - 1] < KFAC1 * work[self.k_targ]:
                                            wm1 = work.gather(1, temp - 1)
                                            w = work.gather(1, temp)
                                            temps = torch.squeeze(temp)
                                            m213711 = m21371 & bmask & (temps > 1) & (wm1 < (KFAC1 * w))
                                            if m213711.any():
                                                print(f'm213711 called')
                                                state.k_targ[m213711] = state.k_targ[m213711] - 1
                                                #print(f'setting k_targ in m213711 = {state.k_targ[m213711]}')
                                            hnew.r[m21371] = hopt.r[m21371, state.k_targ[m21371]].view(hnew.r[m21371].shape)
                                            hnew.d[m21371] = hopt.d[m21371, state.k_targ[m21371]].view(hnew.d[m21371].shape)
                                            #print(f'setting hnew in m21371 = {hnew.r[m21371]} {hnew.d[m21371]}')

                                        bmask = bmask & ~m2137
                                        #print(f'bmask set at end of m2137 = {bmask}')
                                        #print(f'state.reject = {state.reject}')
                            m21 = m21 & bmask #Apply the break mask in case it has changed
                            if m21.any():
                                #print(f'bmask = {bmask}')
                                #print(f'm21 called')
                                #print(f'k = {k}')
                                #print(f'k_targ = {state.k_targ}')
                                k[m21]=k[m21]+1
                                #print(f'Incremented k = {k} with mask {bmask}')

                #Reset the flags
                m3 = reject #This has dimension M

                if m3.any():
                    print(f'm3 called')
                    bmask[m3]= True #Allow the step to be retried
                    prev_reject[m3] = True
                    m31 = ~state.calcjac & m3 #This has dimension M
                    if m31.any():
                        print(f'm31 called')
                        thetam31    = TensorDual(state.theta.r[m31], state.theta.d[m31])
                        jac_redom31 = TensorDual(state.jac_redo.r[m31], state.jac_redo.d[m31])
                        thetares = thetam31 * jac_redom31
                        state.theta.r[m31] = thetares.r
                        state.theta.d[m31] = thetares.d

        #print(f'Leaving compute_jac with k = {k}')
        compute_step()
        state.calcjac = torch.BoolTensor(state.M).fill_(False).to(device=state.y.device)
        #print(f'setting calcjac to {state.calcjac} after compute_step')
        state.xold = state.x.clone()
        state.x = state.x + h
        state.hdid = h.clone() #Need to clone because h is a tensor
        #print(f'hdid={state.hdid} after compute_step')
        state.first_step = torch.BoolTensor(state.M).fill_(False).to(device=state.y.device)
        ms1 = (k == 1)
        kopt = k.clone() #Need to be careful here as
        if ms1.any():
            print(f'ms1 called')
            kopt[ms1] = 2
            #print(f'kopt = {kopt} in ms1')
        ms2 = k <= state.k_targ
        if ms2.any():
            print(f'ms2 called')
            kopt[ms2] = k[ms2]
            kms2 = k[ms2]
            ms21 = ms2.clone()
            workkms2m1 = TensorDual(work.r[ms2, k[ms2] - 1], work.d[ms2, k[ms2] - 1])
            workkms2   = TensorDual(work.r[ms2, k[ms2]], work.d[ms2, k[ms2]])
            ms21[ms2] =  (workkms2m1 < (workkms2*KFAC1))

            if ms21.any():
                print(f'ms21')
                kopt[ms21] = k[ms21] - 1
            t1 = TensorDual(work.r[ms2, k[ms2]], work.d[ms2, k[ms2]])
            workms2 = TensorDual(work.r[ms2, k[ms2] - 1], work.d[ms2, k[ms2] - 1])
            t2 = (KFAC2 * workms2)
            ms22 = ms2.clone()
            ms22[ms2]=  (t1 < t2)
            if ms22.any():
                print(f'ms22')
                kopt[ms22] = torch.min(k[ms22] + 1, torch.tensor(state.KMAXX))
        ms3 = (k !=1) & (k > state.k_targ)
        if ms3.any():
            print(f'ms3')
            kopt[ms3] = k[ms3] - 1
            ms31 = ms3.clone()
            workms3m1 = TensorDual(work.r[ms3, k[ms3] - 1], work.d[ms3, k[ms3] - 1])
            workms3m2 = TensorDual(work.r[ms3, k[ms3] - 2], work.d[ms3, k[ms3] - 2])
            ms31[ms3] = ms31[ms3] & (k[ms3] > 2) & (workms3m2 < KFAC1 * workms3m1)
            #k > 2 and work[k - 2] < KFAC1 * work[k - 1]:
            if ms31.any():
                print(f'ms31')
                kopt[ms31]=k[ms31] - 2
            ms32 = ms3.clone()
            workms3k    = TensorDual(work.r[ms3, k[ms3]], work.d[ms3, k[ms3]])
            workms3kopt = TensorDual(work.r[ms3, kopt[ms3]], work.d[ms3, kopt[ms3]])
            ms32[ms3] = ms32[ms3] & (workms3k < (KFAC2 * workms3kopt))
            #work[k] < KFAC2 * work[kopt]:
            if ms32.any():
                print(f'ms32')
                kopt[ms32] = torch.min(k[ms32], torch.ones_like(k[ms32])*state.KMAXX)
        ms4 = prev_reject.clone() #Need to clone because the reference is changed later on
        if ms4.any():
            print(f'ms4')
            state.k_targ[ms4] = torch.min(kopt[ms4], k[ms4])
            #print(f'setting k_targ in ms4 = {state.k_targ[ms4]}')
            #print(f'kopt[ms4] = {kopt[ms4]}')
            #print(f'k[ms4] = {k[ms4]}')
            hms4 = TensorDual(h.r[ms4], h.d[ms4])
            hoptms4 = TensorDual(hopt.r[ms4, state.k_targ[ms4]], hopt.d[ms4, state.k_targ[ms4]])
            hnewms4 = TensorDual.min_dual(abs(hms4), hoptms4)
            hnew.r[ms4] = hnewms4.r
            hnew.d[ms4] = hnewms4.d
            #print(f'setting hnew in ms4 = {hnew.r[ms4]} {hnew.d[ms4]}')
            #if torch.allclose(hnew[ms4], torch.tensor(0.307131), atol=1e-6):
            #    print(f'hnew close to {hnew}')
            prev_reject[ms4] = False
            prev_reject[ms4] = False
        ms5 = ~ms4
        if ms5.any():
            print(f'ms5')
            ms51 = ~ms4 & (kopt <= k)
            if ms51.any():
                print(f'ms51')
                hnew.r[ms51] = hopt.r[ms51, kopt[ms51]].view(hnew.r[ms51].shape)
                hnew.d[ms51] = hopt.d[ms51, kopt[ms51]].view(hnew.d[ms51].shape)
                #print(f'setting hnew in ms51 = {hnew.r[ms51]} {hnew.d[ms51]}')
            ms52 = ~ms4 & (kopt > k)
            if ms52.any():
                print(f'ms52 = {ms52}')
                t1 = (k < state.k_targ)
                ke = k.unsqueeze(1)
                km1e = (k - 1).unsqueeze(1)
                t2 = work.gather(1, ke) < (KFAC2 * work.gather(1, km1e))
                ms521 = ms52 & t1 & t2
                if ms521.any():
                    print(f'ms521={ms521}')
                    #print(f'k={k}')
                    #print(f'hopt[{ms521, k[ms521]}]={hopt[ms521, k[ms521]]}')
                    #print(f'hopt={hopt[ms521, k[ms521]]}')
                    #print(f'hopt={hopt[ms521, k[ms521]]}')
                    #print(f'state.cost={state.cost[ms521, kopt[ms521] + 1]}')
                    #print(f'factor1={(hopt[ms521, k[ms521]] * state.cost[ms521, kopt[ms521] + 1])}')
                    #print(f'factor2={state.cost[ms521, k[ms521]]}')
                    hoptms521 = TensorDual(hopt.r[ms521, k[ms521]], hopt.d[ms521, k[ms521]])
                    costms521p1 = TensorDual(state.cost.r[ms521, kopt[ms521] + 1], state.cost.d[ms521, kopt[ms521] + 1])
                    costms521 = TensorDual(state.cost.r[ms521, k[ms521]], state.cost.d[ms521, k[ms521]])
                    hnewres = (hoptms521 * costms521p1)*costms521.reciprocal()
                    hnew.r[ms521] = hnewres.r
                    hnew.d[ms521] = hnewres.d
                    print(f'setting hnew in ms521 = {hnew.r[ms521]}, {hnew.d[ms521]}')
                ms522 = ms52 & ~(t1 & t2)
                if ms522.any():
                    print(f'ms522')
                    hoptms522 = TensorDual(hopt.r[ms522, k[ms522]], hopt.d[ms522, k[ms522]])
                    costms522 = TensorDual(state.cost.r[ms522, k[ms522]], state.cost.d[ms522, k[ms522]])
                    costms522kopt = TensorDual(state.cost.r[ms522, kopt[ms522]], state.cost.d[ms522, kopt[ms522]])
                    hnewms522 = (hoptms522 * costms522kopt)*costms522.reciprocal()
                    hnew.r[ms522] = hnewms522.r
                    hnew.d[ms522] = hnewms522.d
                   #print(f'setting hnew in ms522 = {hnew.r[ms522]} {hnew.d[ms522]}')
            if ms5.any():
                print(f'ms5 called')
                state.k_targ[ms5] = kopt[ms5]
                #print(f'setting k_targ in ms5 = {state.k_targ[ms5]}')
        forwards = forward
        if forwards.any():
            state.hnext.r[forwards] = hnew.r[forwards]
            state.hnext.d[forwards] = hnew.d[forwards]
        if ~forwards.any():
            state.hnext.r[~forwards] = -hnew.r[~forwards]
            state.hnext.d[~forwards] = -hnew.d[~forwards]



    def grad_check(self, fun, val):
        #check to see if the tensor has all zeros
        valt = val.clone()
        M = val.r.shape[0]
        D = val.r.shape[1]
        N = val.r.shape[1]
        valt.d = torch.zeros((M, D, N), dtype=val.r.dtype, device=val.r.device)
        for i in range(D):
            valp = valt.clone()
            delta = 1.0e-8*torch.max(1.0, valt.r[:,i])
            valp.r[:,i] += delta
            resp = fun(valp)
            valm = valt.clone()
            valm.r[:,i] -= delta
            resm = fun(valm)
            valt.d = valt.d * 0.0
            valt.d[:,i,i] = 1.0
            res = fun(valt)
            assert torch.allclose(res.d[:,i], (resp.r - resm.r)/(2.0*delta), atol=1.0e-5)




    def stability_check(self, y, x, htot, k, nseq, scale, thetain, dfdy, params, dyns):
        #print(f'k={k}')
        theta = thetain.clone()
        nsteps = nseq[torch.arange(nseq.shape[0], dtype=torch.int64), k] #This is of dimension [M]
        #convert nsteps to double
        nsteps = nsteps.double()


        h = htot/nsteps


        a = -dfdy #This avoids side effects
        D = y.r.shape[1]
        identity_matrix = torch.eye(D, dtype=y.dtype, device=y.device)
        identity_matrix_expanded = identity_matrix.unsqueeze(0).expand(k.shape[0], D, D)
        identity_matrix_expanded_dual = TensorMatDual.createDual(identity_matrix_expanded, y.d.shape[2])
        hrec = h.reciprocal()
        diagonal_tensor = hrec*identity_matrix_expanded_dual
        a = a + diagonal_tensor


        aqr = QRDual(a)

        # def test_ludual():
        #     #Test for gradients using FD
        #     M, D, D, N = a.d.shape
        #     ac = a.clone()
        #     ac.d = torch.zeros([M, D, D, D*D], dtype=a.d.dtype, device=a.d.device)
        #     for i in range(D):
        #         for j in range(D):
        #             ac.d[:, i, j, i*D+j] = 1.0
        #     aqrc = QRDual(ac)
        #     Jac = aqrc.qt.sum(dim=1).sum(dim=2).square()
        #
        #     for i in range(D):
        #         for j in range(D):
        #             dx = 1.0e-8*max(1.0, ac.r[:,i,j])
        #             acp = ac.clone()
        #             acp.r[:,i,j] = ac.r[:,i, j]+dx
        #             aqrp = QRDual(acp)
        #             Jp = aqrp.qt.sum(dim=1).sum(dim=2).square()
        #             acm = ac.clone()
        #             acm.r[:,i,j] = ac.r[:,i, j]-dx
        #             alum = QRDual(acm)
        #             Jm = alum.qt.sum(dim=1).sum(dim=2).square()
        #             Jfd = (Jp-Jm)/(2.0*dx)
        #             print(f'Jac duals={Jac.d[:, :, :, i*D+j]} Jfd={Jfd.r}')
        #             assert torch.allclose(Jac.d[:, :, :, i*D+j], Jfd.r, atol=1.0e-5)
        #
        # if __debug__:
        #    test_ludual()



        xnew = x + h
        ytemp = y.clone()
        delta = dyns(xnew, ytemp, params)
        if torch.isnan(delta.r).any() or torch.isnan(delta.d).any():
            print(f'Found nan in delta.r or delta.d')
        assert torch.isnan(delta.r).any() == False, "Found nan in delta.r"
        assert torch.isnan(delta.d).any() == False, "Found nan in delta.d"
        # def test_dyns():
        #     #Test the dynamics
        #     ytempc = ytemp.clone()
        #     M, D, N = ytempc.d.shape
        #     ytempc.d = torch.zeros([M, D, D], dtype=ytempc.r.dtype, device=ytempc.r.device)
        #     for i in range(D):
        #         ytempc.d[:, i, i] = 1.0
        #     deltac = dyns(xnew, ytempc)
        #     Jdeltac = deltac.sum().square()
        #     for i in range(D):
        #         dx = 1.0e-8*max(1.0, ytempc.r[:,i])
        #         ytempcp = ytempc.clone()
        #         ytempcp.r[:,i] = ytempc.r[:,i]+dx
        #         deltacp = dyns(xnew, ytempcp)
        #         Jdeltacp = deltacp.sum().square()
        #         ytempcm = ytempc.clone()
        #         ytempcm.r[:,i] = ytempc.r[:,i]-dx
        #         deltacm = dyns(xnew, ytempcm)
        #         Jdeltacm = deltacm.sum().square()
        #         Jdeltafd = (Jdeltacp-Jdeltacm)/(2.0*dx)
        #         print(f'delta duals={Jdeltac.d[:, 0, i]} Jdeltafd={Jdeltafd.r[:,0]}')
        #         assert torch.allclose(Jdeltac.d[:, 0, i], Jdeltafd.r[:,0], atol=1.0e-5)
        #if __debug__:
        #     test_dyns()
        tdelta=aqr.solvev(delta)
        if torch.isnan(tdelta.r).any() or torch.isnan(tdelta.d).any():
            print(f'Found nan in delta.r or delta.d')
        assert not torch.isnan(tdelta.r).any(), "Found nan in delta.r"
        assert not torch.isnan(tdelta.d).any(), "Found nan in delta.d"

        # def test_solvev():
        #     deltac = delta.clone()
        #     deltac.d = deltac.d*0.0
        #     M, D, N = deltac.d.shape
        #     for i in range(D):
        #         deltac.d[:, i, i] = 1.0
        #     deltaJ = aqr.solvev(deltac).square().sum()
        #     for i in range(D):
        #         dx = 1.0e-8 * max(torch.abs(deltac.r[0, i]), 1.0)
        #         deltacp = deltac.clone()
        #
        #         deltacp.r[0,i] = deltacp.r[0,i]+dx
        #         deltaJp = aqr.solvev(deltacp).square().sum()
        #         deltacm = deltac.clone()
        #         deltacm.r[0,i] = deltacm.r[0,i]-dx
        #         deltaJm = aqr.solvev(deltacm).square().sum()
        #         deltaJfd = (deltaJp-deltaJm)/(2*dx)
        #         print(f'deltaJ1={deltaJ.d[:,0,i]} deltaJfd={deltaJfd.r}')
        #         assert torch.allclose(deltaJ.d[:, 0, i], deltaJfd.r[:,0], atol=1.0e-2)
        #if __debug__:
        #    test_solvev()




        delta=tdelta.clone()

        #print(f'First delta = {delta}')
        #This creates a set of indexes to iterate over

        #Find the maximum number of steps across all samples
        mstep = torch.max(nsteps)
        yend = y.clone() #This ensures no side effects
        #There are three filters that get applied
        success = torch.BoolTensor(x.r.shape[0]).fill_(True).to(x.device)
        success_mask = torch.BoolTensor(x.r.shape[0]).fill_(True).to(x.device)
        maskif = torch.BoolTensor(x.r.shape[0]).fill_(True).to(x.device)
        for nn in range(1, mstep.int()):
            #Here nstep is of dimension [M]
            #Find all the indexes where nn is less than nsteps
            #This is essentially a filter

            masknn = nsteps >= torch.tensor(nn)
            ytempmasknn = TensorDual(ytemp.r[masknn], ytemp.d[masknn])
            deltamasnn = TensorDual(delta.r[masknn], delta.d[masknn])
            ytempres = ytempmasknn + deltamasnn
            ytemp.r[masknn] = ytempres.r
            ytemp.d[masknn] = ytempres.d
            #print(f'ytemp in batch for nn={nn}={ytemp}')
            xnew.r[masknn] += h.r[masknn]
            xnew.d[masknn] += h.d[masknn]
            xnewmasknn = TensorDual(xnew.r[masknn], xnew.d[masknn])
            ytempmasknn = TensorDual(ytemp.r[masknn], ytemp.d[masknn])
            paramsnn = TensorDual(params.r[masknn], params.d[masknn])
            yendres = dyns(xnewmasknn, ytempmasknn, paramsnn)
            # def test_dyns_inner():
            #     #Test the dynamics
            #     ytempmasknnc = ytempmasknn.clone()
            #     xnewmasknnc = xnewmasknn.clone()
            #     M, D, N = ytempmasknnc.d.shape
            #     ytempmasknnc.d = torch.zeros([M, D, D], dtype=ytempmasknn.r.dtype, device=ytempmasknn.r.device)
            #     for i in range(D):
            #         ytempmasknnc.d[:, i, i] = 1.0
            #     yendresc = dyns(xnewmasknnc, ytempmasknnc)
            #     Jc = yendresc.square().sum()
            #     for i in range(D):
            #         dx = 1.0e-8*max(1.0, ytempmasknnc.r[:,i])
            #         ytempmasknncp = ytempmasknnc.clone()
            #         ytempmasknncp.r[:,i] = ytempmasknnc.r[:,i]+dx
            #         yendrescp = dyns(xnewmasknnc, ytempmasknncp)
            #         Jcp = yendrescp.square().sum()
            #         ytempmasknncm = ytempmasknnc.clone()
            #         ytempmasknncm.r[:,i] = ytempmasknnc.r[:,i]-dx
            #         yendrescm = dyns(xnewmasknnc, ytempmasknncm)
            #         Jcm = yendrescm.square().sum()
            #         Jcfd = (Jcp-Jcm)/(2.0*dx)
            #         print(f'delta Jc dyns={Jc.d[:, 0, i]} Jcfd dyns={Jcfd.r[:,0]}')
            #         assert torch.allclose(Jc.d[:, 0, i], Jcfd.r[:,0], atol=1.0e-5)
            #if __debug__:
            #    test_dyns_inner()
            yend.r[masknn] = yendres.r
            yend.d[masknn] = yendres.d
            #print(f'yend in batch for nn={nn}={yend}')
            #print(f'masknn={masknn}')
            if nn == 1 & (k[masknn] <= 1).any():

                maskif = masknn & (nn == 1) & (k <= 1)
                #print(f'maskif={maskif}')
                deltamaskif = TensorDual(delta.r[maskif], delta.d[maskif])
                scalemaskif = TensorDual(scale.r[maskif], scale.d[maskif])
                del1 = (deltamaskif*scalemaskif.reciprocal()).square().sum().sqrt()
                # def test_del1():
                #     deltamaskifc = deltamaskif.clone()
                #     scalemaskifc = scalemaskif.clone()
                #     M, D, N = deltamaskif.d.shape
                #     N = 2*D
                #     deltamaskifc.d = torch.zeros([M, D, N], dtype=deltamaskif.r.dtype, device=deltamaskif.r.device)
                #     scalemaskifc.d = torch.zeros([M, D, N], dtype=scalemaskif.r.dtype, device=scalemaskif.r.device)
                #     for i in range(D):
                #         deltamaskifc.d[:, i, i]   = 1.0
                #         scalemaskifc.d[:, i, i+D] = 1.0
                #
                #     del1c = (deltamaskifc / scalemaskifc).square().sum().sqrt()
                #     for i in range(D):
                #         dx = 1.0e-8*max(deltamaskifc.r[:,i], 1.0)
                #         deltamaskifcp = deltamaskifc.clone()
                #         deltamaskifcp.r[:,i] = deltamaskifc.r[:,i]+dx
                #         del1cp = (deltamaskifcp / scalemaskifc).square().sum().sqrt()
                #         deltamaskifcm = deltamaskifc.clone()
                #         deltamaskifcm.r[:,i] = deltamaskifc.r[:,i]-dx
                #         del1cm = (deltamaskifcm / scalemaskifc).square().sum().sqrt()
                #         del1cfd = (del1cp-del1cm)/(2.0*dx)
                #         print(f'del1 duals={del1c.d[:, 0, i]} del1fd={del1cfd.r[:,0]}')
                #         assert torch.allclose(del1c.d[:, 0, i], del1cfd.r[:,0])
                #if __debug__:
                #    test_del1()
                if torch.isinf(del1.d).any():
                    print(f'del1.d has nan in it')
                #print(f'del1={del1}')
                xmaskif = TensorDual(x.r[maskif & masknn], x.d[maskif & masknn])
                hmaskif = TensorDual(h.r[maskif & masknn], h.d[maskif & masknn])
                ytempmaskif = TensorDual(ytemp.r[maskif & masknn], ytemp.d[maskif & masknn])
                paramsif = TensorDual(params.r[maskif & masknn], params.d[maskif & masknn])
                dytemp = dyns(xmaskif + hmaskif, ytempmaskif, paramsif)
                #print(f'dytemp in batch={dytemp}')
                deltam = TensorDual(delta.r[maskif & masknn], delta.d[maskif & masknn])
                dhm = TensorDual(h.r[maskif & masknn], h.d[maskif & masknn])
                dh = deltam*dhm.reciprocal()
                #print(f'dh in batch={dh}')
                delta = dytemp-dh
                mask = masknn & maskif
                tdeltaall = aqr.solvev(delta)
                tdelta.r[mask]=tdeltaall.r[mask]
                tdelta.d[mask]=tdeltaall.d[mask]
                # def test_solvef():
                #     deltac = delta.clone()
                #     deltac.d = deltac.d * 0.0
                #     M, D, N = deltac.d.shape
                #     for i in range(D):
                #         deltac.d[:, i, i] = 1.0
                #     deltaJ = aqr.solvef(masknn & maskif, deltac).square().sum()
                #     for i in range(D):
                #         dx = 1.0e-8 * max(torch.abs(deltac.r[0, i]), 1.0)
                #         deltacp = deltac.clone()
                #
                #         deltacp.r[0, i] = deltacp.r[0, i] + dx
                #         deltaJp = aqr.solvef(masknn & maskif, deltacp).square().sum()
                #         deltacm = deltac.clone()
                #         deltacm.r[0, i] = deltacm.r[0, i] - dx
                #         deltaJm = aqr.solvef(masknn & maskif, deltacm).square().sum()
                #         deltaJfd = (deltaJp - deltaJm) / (2 * dx)
                #         print(f'deltaJ1={deltaJ.d[:, 0, i]} deltaJfd={deltaJfd.r}')
                #         assert torch.allclose(deltaJ.d[:, 0, i], deltaJfd.r[:, 0], atol=1.0e-2)
                #if __debug__:
                #    test_solvef()
                delta = tdelta.clone()
                if torch.isnan(delta.d).any():
                    print(f'delta.d has nan in it')

                #print(f'Inner delta = {delta}')
                deltamaskif = TensorDual(delta.r[maskif], delta.d[maskif])
                scalemaskif = TensorDual(scale.r[maskif], scale.d[maskif])
                del2 = (deltamaskif*scalemaskif.reciprocal()).square().sum().sqrt()
                #print(f'del2={del2}')
                # def test_del2():
                #     deltamaskifc = deltamaskif.clone()
                #     scalemaskifc = scalemaskif.clone()
                #     M, D, N = deltamaskif.d.shape
                #     N = 2*D
                #     deltamaskifc.d = torch.zeros([M, D, N], dtype=deltamaskif.r.dtype, device=deltamaskif.r.device)
                #     scalemaskifc.d = torch.zeros([M, D, N], dtype=scalemaskif.r.dtype, device=scalemaskif.r.device)
                #     for i in range(D):
                #         deltamaskifc.d[:, i, i]   = 1.0
                #         scalemaskifc.d[:, i, i+D] = 1.0
                #
                #     del2c = (deltamaskifc / scalemaskifc).square().sum().sqrt()
                #     for i in range(D):
                #         dx = 1.0e-8*max(deltamaskifc.r[:,i], 1.0)
                #         deltamaskifcp = deltamaskifc.clone()
                #         deltamaskifcp.r[:,i] = deltamaskifc.r[:,i]+dx
                #         del2cp = (deltamaskifcp / scalemaskifc).square().sum().sqrt()
                #         deltamaskifcm = deltamaskifc.clone()
                #         deltamaskifcm.r[:,i] = deltamaskifc.r[:,i]-dx
                #         del2cm = (deltamaskifcm / scalemaskifc).square().sum().sqrt()
                #         del2cfd = (del2cp-del2cm)/(2.0*dx)
                #         print(f'del2 duals={del2c.d[:, 0, i]} del2fd={del2cfd.r[:,0]}')
                #         assert torch.allclose(del2c.d[:, 0, i], del2cfd.r[:,0], atol=1.0e-3)
                #if __debug__:
                #    test_del2()
                #del2 = torch.sqrt(torch.sum(torch.square(delta / scale)))

                #theta = del2 / torch.max(torch.ones_like(del1), del1)

                max_one_del1 = TensorDual.max_dual(TensorDual.ones_like(del1), del1) #
                thetares = del2*max_one_del1.reciprocal()


                # def test_thetares():
                #     del2c = del2.clone()
                #     del1c = del1.clone()
                #     M, D, N = del2c.d.shape
                #     del2c.d = torch.zeros([M, D, D+D], dtype=del2c.r.dtype, device=del2c.r.device)
                #     del1c.d = torch.zeros([M, D, D+D], dtype=del1c.r.dtype, device=del1c.r.device)
                #     for i in range(D):
                #         del1c.d[:,i, i]   = 1.0
                #         del2c.d[:,i, D+i] = 1.0
                #
                #     thetaresc = del2c / TensorDual.max_dual(TensorDual.ones_like(del1c), del1c)
                #     Jc = thetaresc.square().sum()
                #     for i in range(D):
                #         dx = 1.0e-8*max(1.0, del2c.r[:,i])
                #         del2cp = del2c.clone()
                #         del2cp.r[:,i] = del2c.r[:,i]+dx
                #         thetarescp = del2cp / TensorDual.max_dual(TensorDual.ones_like(del1c), del1c)
                #         Jcp = thetarescp.square().sum()
                #         del2cm = del2c.clone()
                #         del2cm.r[:,i] = del2c.r[:,i]-dx
                #         thetarescm = del2cm / TensorDual.max_dual(TensorDual.ones_like(del1c), del1c)
                #         Jcm = thetarescm.square().sum()
                #         Jfd = (Jcp-Jcm)/(2.0*dx)
                #         print(f'thetares duals={thetaresc.d[:, 0, i]} Jfd={Jfd.r[:,0]}')
                #         assert torch.allclose(thetaresc.d[:, 0, i], Jfd.r[:,0], atol=1.0e-2)
                #     for i in range(D):
                #         dx = 1.0e-8*max(1.0, del1c.r[:,i])
                #         del1cp = del1c.clone()
                #         del1cp.r[:,i] = del1c.r[:,i]+dx
                #         thetarescp = del2c / TensorDual.max_dual(TensorDual.ones_like(del1cp), del1cp)
                #         Jcp = thetarescp.square().sum()
                #         del1cm = del1c.clone()
                #         del1cm.r[:,i] = del1c.r[:,i]-dx
                #         thetarescm = del2c / TensorDual.max_dual(TensorDual.ones_like(del1cm), del1cm)
                #         Jcm = thetarescm.square().sum()
                #         Jfd = (Jcp-Jcm)/(2.0*dx)
                #         print(f'thetares duals={Jc.d[:, 0, D+i]} Jfd={Jfd.r[:,0]}')
                #         assert torch.allclose(Jc.d[:, 0, D+i], Jfd.r[:,0], atol=1.0e-5)
                #
                #         del1cm = del1c.clone()
                #         del1cm.r[:,i] = del1c.r[:,i]-dx
                #if __debug__:
                #    test_thetares()
                theta.r[maskif] = thetares.r
                theta.d[maskif] = thetares.d
                thetamaskif = TensorDual(theta.r[maskif], theta.d[maskif])
                if (thetamaskif > 1.0).any():
                    success_mask[maskif] = thetamaskif <= 1.0
                    success[:] = success_mask

            #print(f'yend before final LU={yend}')
            mall = masknn & maskif & success_mask

            tdeltares=aqr.solvev(yend)
            tdelta.r[mall]=tdeltares.r[mall]
            tdelta.d[mall]=tdeltares.d[mall]
            #ax = torch.einsum('mij,mj->mi', a[masknn,:,:], tdelta)
            # print(f'a*x={ax}')
            #print(f'Error in LU={torch.norm(ax - y[masknn,:])}')
            delta.r[mall] = tdelta.r[mall]
            delta.d[mall] = tdelta.d[mall]
            #print(f'End delta={delta}')
            yend.r[mall]=ytemp.r[mall] + delta.r[mall]
            yend.d[mall]=ytemp.d[mall] + delta.d[mall]
        #create an assertion that theta is not nan
        return success, theta, yend


    def extrapolate(self, kst, last, table, coeff):
        l = last.r.shape[1]
        ks = kst.tolist()
        for i in range(len(ks)):
            for j in range(ks[i] - 1, 0, -1):
                tableijl = TensorMatDual(table.r[i:i+1, j:j+1, :l], table.d[i:i+1, j:j+1, :l])
                coeffiksj = TensorMatDual(coeff.r[i:i+1, ks[i]:ks[i]+1, j:j+1], coeff.d[i:i+1, ks[i]:ks[i]+1, j:j+1])
                tableijm1l = TensorMatDual(table.r[i:i+1, j - 1:j, :l], table.d[i:i+1, j - 1:j, :l])
                tableres = tableijl + coeffiksj * (tableijl - tableijm1l)
                table.r[i, j - 1, :l] = tableres.r
                table.d[i, j - 1, :l] = tableres.d
        for i in range(len(ks)):
            #THere is one k per sample
            tablei0l = TensorDual(table.r[i:i+1, 0, :l], table.d[i:i+1, 0, :l])
            coeffiks0 = TensorDual(coeff.r[i:i+1, ks[i], 0:1], coeff.d[i:i+1, ks[i], 0:1])
            lastil = TensorDual(last.r[i:i+1, :l], last.d[i:i+1, :l])
            lastres = tablei0l + coeffiks0 * (tablei0l - lastil)
            last.r[i, :l] = lastres.r
            last.d[i, :l] = lastres.d

        return last, table
