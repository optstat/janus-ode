import torch
import copy
from LUBatch import LUBatch
from LU import LU
torch.set_printoptions(precision=16)
torch.set_num_threads(torch.get_num_threads())

class SeulexState:
    def __init__(self, y, x, atoll, rtoll, KMAXX, alpha, ft):
        self.M, self.D = y.shape
        if not isinstance(atoll, torch.Tensor):
            self.atol = torch.tensor(atoll, dtype=y.dtype, device=y.device)
        else:
            self.atol = atoll
        if not isinstance(rtoll, torch.Tensor):
            self.rtol = torch.tensor(rtoll, dtype=y.dtype, device=y.device)
        else:
            self.rtol = rtoll
        self.x = x.clone()
        self.y = y.clone()
        if not isinstance(alpha, torch.Tensor):
            self.alpha = torch.tensor(alpha, dtype=y.dtype, device=y.device)
        else:
            self.alpha = alpha
        if not isinstance(ft, torch.Tensor):
            self.ft = torch.tensor(ft, dtype=y.dtype, device=y.device)
        else:
            self.ft = ft
        self.KMAXX = KMAXX
        self.IMAXX = self.KMAXX + 1
        self.nseq = torch.zeros([self.M, self.IMAXX,], dtype=torch.int64, device=y.device)
        self.cost = torch.zeros([self.M, self.IMAXX,], dtype=y.dtype, device=y.device)
        self.table = torch.zeros((self.M, self.IMAXX, self.D), dtype=y.dtype, device=y.device)
        self.dfdy = torch.zeros((self.M, self.D, self.D), dtype=y.dtype, device=y.device)
        self.calcjac = torch.tensor(self.M*[False], device=y.device)
        self.a = torch.zeros((self.M, self.D, self.D), dtype=y.dtype, device=y.device)
        self.coeff = torch.zeros((self.M, self.IMAXX, self.IMAXX), dtype=y.dtype, device=y.device)
        self.costfunc = torch.ones((self.M), dtype=y.dtype, device=y.device)
        self.costjac = torch.ones((self.M), dtype=y.dtype, device=y.device)*5.0
        self.costlu = torch.ones((self.M), dtype=y.dtype, device=y.device)
        self.costsolv = torch.ones((self.M), dtype=y.dtype, device=y.device)
        self.EPS = torch.tensor(torch.finfo(y.dtype).eps).to(y.device)
        self.jac_redo =torch.ones((self.M), dtype=y.dtype, device=y.device)*torch.min(torch.tensor(1.0e-4), self.rtol)
        self.theta = 2.0*self.jac_redo
        self.one = torch.tensor(1, dtype=torch.int64, device=y.device)
        self.nseq[:, 0] = 2
        self.nseq[:, 1] = 3
        for i in range(2, self.IMAXX):
            self.nseq[:, i] = 2 * self.nseq[:, i - 2]
        #cost[0] = costjac + costlu + nseq[0] * (costfunc + costsolve);
        #for (Int k=0;k < KMAXX;k++)
        #    cost[k + 1] = cost[k] + (nseq[k + 1] - 1) * (costfunc + costsolve) + costlu;
        self.cost[:, 0] = self.costjac + self.costlu + self.nseq[:, 0] * (self.costfunc + self.costsolv)
        #vectorize this loop
        self.first_step= torch.tensor([True]*self.M, device=y.device)
        self.last_step = torch.tensor([False]*self.M, device=y.device)
        self.forward= torch.tensor([False]*self.M, device=y.device)
        self.reject = torch.tensor([False]*self.M, device=y.device)
        self.prev_reject = torch.tensor([False]*self.M, device=y.device)
        for k in range(self.KMAXX):
            kp1 = k+1
            self.cost[:, kp1] = self.cost[:, k] + (self.nseq[:,kp1]-1)*(self.costfunc + self.costsolv) + self.costlu
        self.hnext = -1.0e99*torch.ones((self.M), dtype=y.dtype, device=y.device)
        self.hdid = torch.zeros((self.M), dtype=y.dtype, device=y.device)

        self.logfact = -torch.log10(self.rtol + self.atol) * 0.6 + 0.5
        #k_targ depends only on the tolerances which are the same for all samples
        maxt = (torch.max(self.one, torch.min(torch.tensor(self.KMAXX - 1), torch.round(self.logfact))).floor().long())
        self.k_targ = torch.ones([self.M],   dtype=torch.int64, device = y.device)*maxt
        #print(f'setting k_targ to {self.k_targ} in constructor')

        for k in range (self.IMAXX):
            for l in range(k):
                ratio = self.nseq[:, k].to(y.dtype) / self.nseq[:, l]
                self.coeff[:, k, l] = 1.0/ (ratio-1.0)
        self.errold = torch.zeros((self.M), dtype=y.dtype, device=y.device)



        #Override setindex method
    def __setitem__(self, mask, newstate):
        self.y[mask] = newstate.y
        self.x[mask] = newstate.x
        self.x[mask] = newstate.x
        self.alpha   = newstate.alpha
        self.ft      = newstate.ft
        self.atol = newstate.atol
        self.rtol = newstate.rtol
        self.KMAXX = newstate.KMAXX
        self.IMAXX = newstate.IMAXX
        self.nseq[mask] = newstate.nseq
        self.cost[mask] = newstate.cost
        self.table[mask] = newstate.table
        self.dfdy[mask] = newstate.dfdy
        self.calcjac[mask] = newstate.calcjac
        self.a[mask] = newstate.a
        self.coeff[mask] = newstate.coeff
        self.costfunc[mask] = newstate.costfunc
        self.costjac[mask] = newstate.costjac
        self.costlu[mask] = newstate.costlu
        self.costsolv[mask] = newstate.costsolv
        self.EPS = newstate.EPS
        self.jac_redo[mask] = newstate.jac_redo
        self.theta[mask] = newstate.theta
        self.one = newstate.one
        self.first_step[mask] = newstate.first_step
        self.last_step[mask] = newstate.last_step
        self.forward[mask] = newstate.forward
        self.reject[mask] = newstate.reject
        self.prev_reject[mask] = newstate.prev_reject
        self.hnext[mask] = newstate.hnext
        self.hdid[mask] = newstate.hdid
        self.logfact = newstate.logfact
        self.k_targ[mask] = newstate.k_targ
        self.coeff[mask] = newstate.coeff
        self.errold[mask] = newstate.errold
        self.M = self.y.shape[0]


    def __getitem__(self, mask):
        newState = copy.deepcopy(self)
        newState.nseq = self.nseq[mask]
        newState.cost = self.cost[mask]
        newState.table = self.table[mask]
        newState.dfdy = self.dfdy[mask]
        newState.calcjac = self.calcjac[mask]
        newState.a = self.a[mask]
        newState.coeff = self.coeff[mask]
        newState.costfunc = self.costfunc[mask]
        newState.costjac = self.costjac[mask]
        newState.costlu = self.costlu[mask]
        newState.costsolv = self.costsolv[mask]
        newState.jac_redo = self.jac_redo[mask]
        newState.theta = self.theta[mask]
        newState.first_step = self.first_step[mask]
        newState.last_step = self.last_step[mask]
        newState.forward = self.forward[mask]
        newState.reject = self.reject[mask]
        newState.prev_reject = self.prev_reject[mask]
        newState.hnext = self.hnext[mask]
        newState.hdid = self.hdid[mask]
        newState.k_targ = self.k_targ[mask]
        newState.coeff = self.coeff[mask]
        newState.errold = self.errold[mask]
        newState.y = self.y[mask]
        newState.x = self.x[mask]
        newState.M = newState.y.size()[0]
        return newState


class SeulexT:
    def __init__(self):
        pass
    #htry is in [M,]
    #dyns is in [M, D]
    #sate is a SeulexState and is expected to be filters so that the stapsize is non zero

    def step(self, htry, dyns, state : SeulexState)->SeulexState:
        STEPFAC1 = torch.tensor(0.6, dtype=torch.float64, device=state.y.device)
        STEPFAC2 = torch.tensor(0.93, dtype=torch.float64, device=state.y.device)
        STEPFAC3 = torch.tensor(0.1, dtype=torch.float64, device=state.y.device)
        STEPFAC4 = torch.tensor(4.0, dtype=torch.float64, device=state.y.device)
        STEPFAC5 = torch.tensor(0.5, dtype=torch.float64, device=state.y.device)
        KFAC1 = torch.tensor(0.7, dtype=torch.float64, device=state.y.device)
        KFAC2 = torch.tensor(0.9, dtype=torch.float64, device=state.y.device)
        state.errold = torch.zeros((state.M,), dtype=state.y.dtype, device=state.y.device)
        firstk = torch.BoolTensor(state.x.size()).fill_(True).to(device=state.y.device)
        if not isinstance(htry, torch.Tensor):
            htry = torch.tensor(htry, dtype=state.y.dtype, device=state.y.device)

        hopt = torch.zeros((state.M, state.IMAXX), dtype=state.y.dtype, device=state.y.device)
        work = torch.zeros((state.M, state.IMAXX), dtype=state.y.dtype, device=state.y.device)
        fac  = torch.zeros((state.M), dtype=state.y.dtype, device=state.y.device)

        work[:, 0:1] = 1.0e30
        h = htry.clone()
        ysav = state.y.clone()

        state.forward[h > 0] = True
        state.forward[h <= 0] = False

        if ((h != state.hnext) & ~ state.first_step).any():
            mask = (h != state.hnext) & ~ state.first_step
            state.last_step[mask]   = True

        if state.reject.any():
            state.last_step[state.reject]   = False
            state.prev_reject[state.reject] = True
            state.theta[state.reject] = 2.0 * state.jac_redo[state.reject]

        scale = state.atol + state.rtol * torch.abs(state.y)
        state.reject = torch.tensor([False]*state.M).to(device=state.y.device)

        hnew = torch.abs(h)
        #if torch.allclose(hnew, torch.tensor([0.660544, 0.168978]), atol=1e-6):
        #    print(f'hnew is close to {hnew}')

        print(f'hnew = {hnew} set in step')
        #k is a counter used internally by compute_jac
        k=torch.zeros_like(htry).to(dtype=torch.int64)
        #Force calculation of the jacobian

        def compute_jac():
            print(f'compute_jac called')
            #This is a break mask.  If it is False it means that the loop should be broken
            #The state passed in ensures that the samples do not contain any zero step sizes
            bmask = torch.BoolTensor(state.y.shape[0]).fill_(True).to(device=state.y.device)
            while ((firstk | state.reject) ).any():

                m0 = (firstk | state.reject)
                print(f'm0 = {m0} set in compute_jac')
                m1 = (state.theta > state.jac_redo) & (~state.calcjac) & (torch.abs(h) > state.EPS) & m0
                print(f'm1 = {m1} set in compute_jac')
                if ~torch.any(m1):
                    print('m1 is all false')
                if torch.any(m1):
                    print(f'm1 = {m1} set in compute_jac')
                    #print(f'state.dfdy.shape = {state.dfdy.shape}')
                    #print(f'state.x.shape = {state.x.shape}')
                    #print(f'state.y.shape = {state.y.shape}')
                    state.dfdy[m1, :] = dyns.jacobian(state.x[m1], state.y[m1, :])
                    state.calcjac[m1] = torch.tensor(True).to(device=state.y.device)
                    print(f'setting calcjac to True for mask {m1}')

                m2 = (firstk | state.reject)  & bmask & m0



                firstk[m2]= False
                state.reject[m2]= False

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
                            print(f'm21 called with k={k[m21]} and k_targ={state.k_targ[m21]}')
                            #    def dy(state, y, x, htot, k, scale, theta, dfdy, dyns):

                            #if torch.allclose(state.x[m21], torch.tensor(9.9346237143, dtype=torch.float64), atol=1.0e-6):
                            #    print(f'x is close to {state.x}')
                            success, state.theta[m21], yseq = self.dy_batch(ysav[m21, :], state.x[m21], hnew[m21], k[m21], \
                                                                      state.nseq[m21, :],  scale[m21, :], \
                                                                      state.theta[m21], state.dfdy[m21, :,  :], dyns)


                            #success2, theta2, yseq2 = self.dy_serial(ysav[m21, :], state.x[m21], hc[m21], k[m21], \
                            #                                          state.nseq[m21, :],  scale[m21, :], \
                            #                                          state.theta[m21], state.dfdy[m21, :,  :], dyns)
                            #if not torch.allclose(success.cuda(), success2.cuda(), rtol=1e-15, atol=1e-15):
                            #    print(f'{success2} not close to {success}')
                            #if not torch.allclose(state.theta[m21], theta2, rtol=1e-15, atol=1e-15):
                            #    print(f'{state.theta[m21]} not close to {theta2}')
                            #    print(f'state.x={state.x[m21]}')

                            print(f'state.theta after dy={state.theta}')
                            print(f'yseq after dy={yseq}')
                            print(f'success shape={success.shape}')
                            print(f'k={k}')
                            print(f'k_targ={state.k_targ}')
                            #Here success is a break mask
                            m211 = m21 & bmask
                            m211[m21]=m211[m21] & ~success
                            if m211.any():
                                print(f'm211 called')
                                hnew[m211] = torch.abs(hnew[m211]) * STEPFAC5
                                print(f'hnew[m211] = {hnew[m211]} set in m211')
                                state.reject[m211] = True
                                bmask[m211] = False
                                print(f'bmask = {bmask} set in m211')
                            m212 = m21 & bmask & (k ==0)
                            if m212.any():
                                print(f'm212 called')
                                state.y[m212,:] = yseq[m212[m21], :]
                            m213 = m21 & bmask
                            m213[m21] = m213[m21] & (k[m21] != 0 & ~success)
                            if m213.any():
                                print(f'm213 called')
                                state.table[m213, k[m213] - 1, :] = yseq[m213[m21], :]  # This is a copy

                                state.y[m213, :], state.table[m213, :, :] = self.polyextr(k[m213], state.y[m213, :],
                                                                                          state.table[m213, :, :],
                                                                                          state.coeff[m213, :, :])
                                print(f'table: {state.table}')
                                scale[m213, :] = state.atol + state.rtol * torch.abs(ysav[m213, :])
                                err = torch.zeros((state.M), dtype=state.y.dtype, device=state.y.device)
                                #err = torch.sqrt(torch.sum(torch.square((state.y-state.table[0][:]) / scale))/state.n)
                                # err[mask] = torch.sqrt(torch.sum(torch.square((state.y[mask, :] - state.table[mask, 0, :]) / scale[mask, :]), dim=0) / state.D)
                                print(f'y={state.y[m213]}')

                                delta = (state.y[m213] - state.table[m213, 0, :])
                                print(f'delta={delta}')
                                sq_inpt = delta / scale[m213]
                                print(f'sq_inpt={sq_inpt}')
                                sq_term = torch.square(sq_inpt)
                                print(f'sq_term={sq_term}')
                                err[m213] = torch.sqrt(torch.sum(sq_term, dim=1) / state.D)
                                print(f'err={err}')

                                print(f'scale={scale[m213]}')
                                print(f'ysav={ysav[m213]}')
                                t1 = ((torch.ones_like(err[m213]) * 1.0 / state.EPS))
                                j2 = err[m213]
                                # m2131 = m213 & bmask
                                m2131 = m213 & bmask
                                m2131[m213] = m2131[m213] & (j2 > t1) | ((k[m213] > 1) & (err[m213] >= state.errold[m213]))
                                if m2131.any():
                                    # print(f'mask before state.reject = {mask}')
                                    state.reject[m2131] = True
                                    hnew[m2131] = torch.abs(hnew[m2131]) * STEPFAC5
                                    print(f'hnew[m2131] = {hnew[m2131]} set in m2131')
                                    bmask[m2131] = False
                                    #print(f'bmask set at the end of m2131 to {bmask}')
                                # bmask has been updated so also update m213
                                m213 = m213 & bmask
                                if m213.any():
                                    print(f'm213 called with m213={m213} and bmask={bmask}')
                                    state.errold[m213] = torch.max(4.0 * err[m213], state.one)
                                    expo = state.one / (k + 1).double()
                                    print(f'expo={expo}')
                                    facmin = torch.pow(STEPFAC3, expo)
                                    m2132 = m213 & bmask & (err == 0.0)
                                    if m2132.any():
                                        print(f'm2132 called')
                                        fac[m2132] = facmin[m2132].reciprocal()
                                    m2133 = ~m2132 & bmask
                                    if m2133.any():
                                        print(f'm2133 called')
                                        fac[m2133] = STEPFAC2 / torch.pow(err[m2133] / STEPFAC1, expo[m2133])
                                        print(f'fac first step: {fac}')
                                        fac[m2133] = torch.max(facmin[m2133] / STEPFAC4,
                                                               torch.min(facmin[m2133].reciprocal(), fac[m2133]))
                                        print(f'fac: {fac}')
                                    print(f'facmin: {facmin}')
                                    print(f'hnew: {hnew}')
                                    hopt[m213, k[m213]] = torch.abs(hnew[m213] * fac[m213])
                                    #if hopt[m213, k[m213]].shape[0] > 1:
                                    #    if torch.allclose(hopt[m213, k[m213]][1], torch.tensor(0.734853), atol=1e-6):
                                    #        print(f'hopt is close to 0.734853]')

                                    print(f'hopt[{m213},{k[m213]} ]={hopt[m213, k[m213]]}')
                                    work[m213, k[m213]] = (state.cost[m213, k[m213]] / hopt[m213, k[m213]])
                                    print(f'work={work[m213]}')
                                    m2134 = m213 & bmask & (state.first_step | state.last_step) & (err <= 1.0)
                                    if m2134.any():
                                        print(f'm2134 called')
                                        bmask[m2134] = False
                                        #print(f'bmask set at m2134={bmask}')
                                    m2135 = m213 & bmask & (
                                                    (k == (state.k_targ - 1)) & ~ state.prev_reject & \
                                                    ~ state.first_step & ~ state.last_step)
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
                                            workkm1 = torch.squeeze(work.gather(1, km1indcs))
                                            kindcs = torch.unsqueeze(k,1)
                                            kindcs[kindcs < 0] = 0
                                            workk = torch.squeeze(work.gather(1, kindcs))
                                            print(f'setting k_targ in m21352 = {state.k_targ[m21352]}')
                                            m213521 =m21352 & bmask & (state.k_targ > 1) & (
                                                    workkm1 < KFAC1 * workk)
                                            if m213521.any():
                                                print(f'm213521 called')
                                                state.k_targ[m213521] = state.k_targ[m213521] - 1
                                                #print(f'setting k_targ in m213521 = {state.k_targ[m213521]}')
                                                hnew[m21352] = hopt[m21352, state.k_targ[m21352]]
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
                                            temp = torch.squeeze(state.nseq.gather(1,nseq_indcs) * 2.0)
                                            m21362 = m2136 & bmask & (err > temp)
                                            if m21362.any():
                                                print(f'm21362 called')
                                                state.reject[m21362] = True

                                                km1indcs = torch.unsqueeze(k-1, 1)
                                                km1indcs[km1indcs<0]=0 #This won't matter since the negative indices are filters out
                                                print(f'km1indcs={km1indcs}')
                                                kindcs = torch.unsqueeze(k, 1)
                                                kindcs[kindcs<0]=0 #This won't matter since the negative indices are filters out
                                                print(f'kindcs={kindcs}')
                                                wm1 = torch.squeeze(work.gather(1, km1indcs))
                                                w = torch.squeeze(work.gather(1, kindcs))
                                                #  if self.k_targ > 1 and work[k-1] < KFAC1*work[k]:
                                                m213621= m21362 & bmask & (state.k_targ > 1) & (wm1 < (KFAC1 * w))
                                                if m213621.any():
                                                    print(f'm213621 called')
                                                    state.k_targ[m213621] = state.k_targ[m213621] - 1
                                                    #print(f'setting k_targ in m213621 = {state.k_targ[m213621]}')
                                                hnew[m21362] = hopt[m21362, state.k_targ[m21362]]
                                                print(f'setting hnew in m21362 = {hnew[m21362]}')
                                                if m21362.any():
                                                    #print(f'm21362 called')
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
                                            state.reject[m21371] = True
                                            temp = torch.unsqueeze(state.k_targ,1)
                                            #if self.k_targ > 1 and work[self.k_targ - 1] < KFAC1 * work[self.k_targ]:
                                            wm1 = torch.squeeze(work.gather(1, temp - 1))
                                            w = torch.squeeze(work.gather(1, temp))
                                            temps = torch.squeeze(temp)
                                            m213711 = m21371 & bmask & (temps > 1) & (wm1 < (KFAC1 * w))
                                            if m213711.any():
                                                print(f'm213711 called')
                                                state.k_targ[m213711] = state.k_targ[m213711] - 1
                                                #print(f'setting k_targ in m213711 = {state.k_targ[m213711]}')
                                            hnew[m21371] = hopt[m21371, state.k_targ[m21371]]
                                            print(f'setting hnew in m21371 = {hnew[m21371]}')

                                            if torch.allclose(hnew, torch.tensor(0.0009137).double(), atol=1e-6):
                                                print(f'hnew = {hnew}')
                                        bmask = bmask & ~m2137
                                        print(f'bmask set at end of m2137 = {bmask}')
                                        print(f'state.reject = {state.reject}')
                            m21 = m21 & bmask #Apply the break mask in case it has changed
                            if m21.any():
                                #print(f'bmask = {bmask}')
                                #print(f'm21 called')
                                #print(f'k = {k}')
                                #print(f'k_targ = {state.k_targ}')
                                k[m21]=k[m21]+1
                                #print(f'Incremented k = {k} with mask {bmask}')

                #Reset the flags
                m3 = state.reject #This has dimension M

                if m3.any():
                    print(f'm3 called')
                    bmask[m3]= True #Allow the step to be retried
                    state.prev_reject[m3] = True
                    m31 = ~state.calcjac & m3 #This has dimension M
                    if m31.any():
                        print(f'm31 called')
                        state.theta[m31] = state.theta[m31] * state.jac_redo[m31]

        #print(f'Leaving compute_jac with k = {k}')
        compute_jac()
        state.calcjac = torch.BoolTensor(state.x.size()).fill_(False).to(device=state.y.device)
        print(f'setting calcjac to {state.calcjac} after compute_jac')
        state.xold = state.x
        state.x = state.x + h
        state.hdid = h.clone() #Need to clone because h is a tensor
        print(f'hdid={state.hdid} after compute_jac')
        state.first_step = torch.BoolTensor(state.x.size()).fill_(False).to(device=state.y.device)
        ms1 = (k == 1)
        kopt = k.clone() #Need to be careful here as
        if ms1.any():
            print(f'ms1')
            kopt[ms1] = 2
            print(f'kopt = {kopt} in ms1')
        ms2 = k <= state.k_targ
        if ms2.any():
            print(f'ms2')
            kopt[ms2] = k[ms2]
            kms2 = k[ms2]
            ms21 = ms2.clone()
            ms21[ms2] =  (work[ms2, k[ms2] - 1] < (work[ms2, kms2]*KFAC1))

            if ms21.any():
                print(f'ms21')
                kopt[ms21] = k[ms21] - 1
            t1 = work[ms2, k[ms2]]
            t2 = (KFAC2 * work[ms2, k[ms2] - 1])
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
            ms31[ms3] = ms31[ms3] & (k[ms3] > 2) & (work[ms3, k[ms3] - 2] < KFAC1 * work[ms3, k[ms3] - 1])
            #k > 2 and work[k - 2] < KFAC1 * work[k - 1]:
            if ms31.any():
                print(f'ms31')
                kopt[ms31]=k[ms31] - 2
            ms32 = ms3.clone()
            ms32[ms3] = ms32[ms3] & (work[ms3, k[ms3]] < KFAC2 * work[ms3, kopt[ms3]])
            #work[k] < KFAC2 * work[kopt]:
            if ms32.any():
                print(f'ms32')
                kopt[ms32] = torch.min(k[ms32], torch.ones_like(k[ms32])*state.KMAXX)
        ms4 = state.prev_reject.clone() #Need to clone because the reference is changed later on
        if ms4.any():
            print(f'ms4')
            state.k_targ[ms4] = torch.min(kopt[ms4], k[ms4])
            print(f'setting k_targ in ms4 = {state.k_targ[ms4]}')
            print(f'kopt[ms4] = {kopt[ms4]}')
            print(f'k[ms4] = {k[ms4]}')
            hnew[ms4] = torch.min(torch.abs(h[ms4]), hopt[ms4, state.k_targ[ms4]])
            print(f'setting hnew in ms4 = {hnew[ms4]}')
            #if torch.allclose(hnew[ms4], torch.tensor(0.307131), atol=1e-6):
            #    print(f'hnew close to {hnew}')
            state.prev_reject[ms4] = False
        ms5 = ~ms4
        if ms5.any():
            print(f'ms5')
            ms51 = ~ms4 & (kopt <= k)
            if ms51.any():
                print(f'ms51')
                hnew[ms51] = hopt[ms51, kopt[ms51]]
                print(f'setting hnew in ms51 = {hnew[ms51]}')
            ms52 = ~ms4 & (kopt > k)
            if ms52.any():
                print(f'ms52 = {ms52}')
                t1 = (k < state.k_targ)
                ke = k.unsqueeze(1)
                km1e = (k - 1).unsqueeze(1)
                t2 = (work.gather(1, ke).squeeze() < KFAC2 * work.gather(1, km1e).squeeze())
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
                    hnew[ms521] = (hopt[ms521, k[ms521]] * state.cost[ms521, kopt[ms521] + 1]) / state.cost[ms521, k[ms521]]
                    print(f'setting hnew in ms521 = {hnew[ms521]}')
                ms522 = ms52 & ~(t1 & t2)
                if ms522.any():
                    print(f'ms522')
                    hnew[ms522] = hopt[ms522, k[ms522]] * state.cost[ms522, kopt[ms522]] / state.cost[ms522, k[ms522]]
                    print(f'setting hnew in ms522 = {hnew[ms522]}')
            if ms5.any():
                state.k_targ[ms5] = kopt[ms5]
                print(f'setting k_targ in ms5 = {state.k_targ[ms5]}')
        forwards = state.forward
        if forwards.any():
            state.hnext[forwards] = hnew[forwards]
        if ~forwards.any():
            state.hnext[~forwards] = -hnew[~forwards]

        return state






    # Here y in [M, D]
    # htot in [M]
    # k is a scalar
    # scale in [M, D]
    def dy_batch(self, y, x, htot, k, nseq, scale, theta, dfdy, dyns):
        #print(f'k={k}')
        nsteps = nseq[torch.arange(nseq.shape[0], dtype=torch.int64), k] #This is of dimension [M]
        h = htot/nsteps
        a = -dfdy
        D = y.shape[1]
        identity_matrix = torch.eye(D, dtype=y.dtype, device=y.device)
        identity_matrix_expanded = identity_matrix.unsqueeze(0).expand(k.shape[0], D, D)
        diagonal_tensor = torch.einsum('m, mij->mij', h.reciprocal(), identity_matrix_expanded)
        a = a + diagonal_tensor


        alu = LUBatch(a)
        xnew = x + h
        ytemp = y.clone()
        delta = dyns(xnew, ytemp)
        tdelta=alu.solvev(delta)

        delta=tdelta
        #print(f'First delta = {delta}')
        #This creates a set of indexes to iterate over

        #Find the maximum number of steps across all samples
        mstep = torch.max(nsteps)
        yend = torch.zeros_like(y)
        #There are three filters that get applied
        success = torch.BoolTensor(x.size()).fill_(True).to(x.device)
        success_mask = torch.BoolTensor(x.size()).fill_(True).to(x.device)
        maskif = torch.BoolTensor(x.size()).fill_(True).to(x.device)
        for nn in range(1, mstep):
            #Here nstep is of dimension [M]
            #Find all the indexes where nn is less than nsteps
            #This is essentially a filter

            masknn = nsteps >= torch.tensor(nn)
            ytemp[masknn, :] = ytemp[masknn, :] + delta[masknn, :]
            #print(f'ytemp in batch for nn={nn}={ytemp}')
            xnew[masknn] += h[masknn]
            yend[masknn, :] = dyns(xnew[masknn], ytemp[masknn, :])
            #print(f'yend in batch for nn={nn}={yend}')
            #print(f'masknn={masknn}')
            if nn == 1 & (k[masknn] <= 1).any():


                maskif = (masknn & nn == 1) & ((k & masknn) <= 1)
                #print(f'maskif={maskif}')
                del1 = torch.sqrt(torch.sum(torch.square((delta[maskif, :] / scale[maskif, :]) ), dim=1))
                #print(f'del1={del1}')
                dytemp = dyns(x[maskif & masknn] + h[maskif & masknn], ytemp[maskif & masknn,:])
                #print(f'dytemp in batch={dytemp}')
                deltam = delta[maskif & masknn]
                dhm = h[maskif & masknn]
                dh = torch.einsum('mi, m-> mi', deltam, dhm.reciprocal())
                #print(f'dh in batch={dh}')
                delta = dytemp-dh
                tdelta = alu.solvef(masknn & maskif, delta)
                delta = tdelta
                #print(f'Inner delta = {delta}')

                del2 = torch.sqrt(torch.sum(torch.square(delta[maskif,:] / scale[maskif,:]), dim=1))
                #print(f'del2={del2}')

                theta[maskif] = del2 / torch.max(torch.ones_like(del1), del1)
                if (theta[maskif] > 1.0).any():
                    success_mask[maskif] = theta[maskif] <= 1.0
                    success[:] = success_mask

            #print(f'yend before final LU={yend}')
            mall = masknn & maskif & success_mask
            tdelta[mall, :]=alu.solvef(masknn & maskif & success_mask, yend)
            #ax = torch.einsum('mij,mj->mi', a[masknn,:,:], tdelta)
            # print(f'a*x={ax}')
            #print(f'Error in LU={torch.norm(ax - y[masknn,:])}')
            delta[mall,:] = tdelta[mall]
            #print(f'End delta={delta}')
            yend[mall]=ytemp[mall] + delta[mall]
        return success, theta, yend

    def dy_serial(self, y, x, htot, k, nseq,  scale, theta, dfdy, dyns):
        ks = k.tolist()
        res=torch.BoolTensor(x.size()).fill_(False)
        yend = torch.zeros_like(y)
        for i in range(len(ks)):
            nstep = nseq[i, ks[i]]
            h = htot[i] / nstep
            a = -dfdy[i]
            a.diagonal()[:] += torch.reciprocal(h)
            #print(f'a={a}')
            alu = LU(a)
            xnew = x[i] + h
            delta = torch.squeeze(dyns(torch.unsqueeze(xnew, 0), torch.unsqueeze(y[i], 0)))
            ytemp = y[i].clone()
            tdelta=alu.solvev(delta)

            #ax = torch.einsum('ij,j->i', a, tdelta)
            #print(f'a*x={ax}')
            #print(f'Error in LU={torch.norm(ax-delta)}')
            delta=tdelta
            #print(f'First delta serial ={delta}')
            res[i] = True
            for nn in range(1, nstep):
                ytemp[:]=ytemp[:]+delta[:]
                #print(f'ytemp in serial for nn={nn}={ytemp}')

                xnew += h
                yend[i] = torch.squeeze(dyns(torch.unsqueeze(xnew, 0), torch.unsqueeze(ytemp, 0)))
                #print(f'yend in serial for nn={nn}={yend}')

                if nn == 1 and ks[i] <= 1:
                    del1 = torch.sqrt(torch.sum(torch.square((delta / scale[i]) )))
                    dytemp = torch.squeeze(dyns(torch.unsqueeze(x[i] + h, 0), torch.unsqueeze(ytemp, 0)))
                    #print(f'dytemp in serial={dytemp}')
                    dh = delta / h
                    #print(f'dh in serial={dh}')
                    delta = dytemp- dh
                    #print(f'Input to delta serial={delta}')
                    tdelta = alu.solvev(delta)

                    delta = tdelta
                    #print(f'Inner delta serial={delta}')

                    del2 = torch.sqrt(torch.sum(torch.square(delta / scale[i])))

                    theta[i] = del2 / torch.max(torch.ones_like(del1), del1)
                    if theta[i] > 1.0:
                        res[i] = False
                        break
                #print(f'yend before final LU serial={yend}')
                tdelta=alu.solvev(yend[i])
                delta = tdelta
                #print(f'End delta serial={delta}')
            if res[i]:
                yend[i] = ytemp + delta
        return res, theta, yend



    #Here k is a tensor of integers of shape (M,). last is a tensor of shape (M, D)


    #vectorize the following function
    def polyextr_bak(self, k, last, table, coeff):
        l = last.shape[1]
        jss= torch.stack([torch.arange(k[i]-1, 0, -1) for i in range(k.shape[0])])
        if jss.shape[1] > 0:
            coeffs = torch.stack([coeff[i, k[i], jss[i]] for i in range(k.shape[0])] )
            tables = torch.stack([table[i, jss[i], :l] for i in range(k.shape[0])])
            tablesm1 = torch.stack([table[i, jss[i]-1, :l] for i in range(k.shape[0])])
            tabledelta = torch.einsum('mj,mjl->mjl', coeffs, (tables - tablesm1))
            # create a range tensor for the batch dimension
            range_tensor = torch.arange(table.shape[0]).view(-1, 1)
            # Extend range tensor to match the shape of y
            range_expanded = range_tensor.expand_as(jss)
            range_expanded_m1 = range_tensor.expand_as(jss-1) #This is the range tensor with the last index reduced by 1
            # Use advanced indexing to get the required elements
            table[range_expanded_m1, jss-1] = table[range_expanded, jss] + tabledelta

        ftable = table[:, 0, :l]
        cselect = coeff.index_select(1, k).select(-1, 0)
        last[:, :l] = ftable + torch.einsum('ij, jk->ik', cselect, (ftable- last[:, :l]))
        #for i in range(k.shape[0]):
        #    js = torch.arange(k[i] - 1, 0, -1)
        #    #for j in range(k - 1, 0, -1):
        #    #    self.table[j - 1, :l] = self.table[j, :l] + self.coeff[k, j] * (self.table[j, :l] - self.table[j - 1, :l])
        #    if js.shape[0] > 0:
        #        tabledelta = torch.einsum('j,jl->jl', coeff[i, k[i], js], (table[i, js, :l] - table[i, js - 1, :l]))
        #        table[i, js - 1, :l] = table[i, js, :l]+ tabledelta
        #    last[i, :l] = table[i][0][:l] + coeff[i][k[i]][0] * (table[i][0][:l] - last[i][:l])
        return last, table

    def polyextr(self, kst, last, table, coeff):
        l = last.shape[1]
        ks = kst.tolist()
        for i in range(len(ks)):
            for j in range(ks[i] - 1, 0, -1):
                table[i, j - 1, :l] = table[i, j, :l] + coeff[i, ks[i], j] * (table[i, j, :l] - table[i, j - 1, :l])
        for i in range(len(ks)):
            #THere is one k per sample
            last[i, :l] = table[i, 0, :l] + coeff[i, ks[i], 0]*(table[i, 0, :l] - last[i, :l])

        #for i in range(k.shape[0]):
        #    js = torch.arange(k[i] - 1, 0, -1)
        #    #for j in range(k - 1, 0, -1):
        #    #    self.table[j - 1, :l] = self.table[j, :l] + self.coeff[k, j] * (self.table[j, :l] - self.table[j - 1, :l])
        #    if js.shape[0] > 0:
        #        tabledelta = torch.einsum('j,jl->jl', coeff[i, k[i], js], (table[i, js, :l] - table[i, js - 1, :l]))
        #        table[i, js - 1, :l] = table[i, js, :l]+ tabledelta
        #    last[i, :l] = table[i][0][:l] + coeff[i][k[i]][0] * (table[i][0][:l] - last[i][:l])
        return last, table

