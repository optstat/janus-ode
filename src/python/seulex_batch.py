import torch
from qrte import QRBatch
torch.set_printoptions(precision=6)


class SeulexBatch:
    def __init__(self, y, x, atoll, rtoll):

        self.M, self.D = y.shape
        self.atol = torch.tensor(atoll, dtype=y.dtype, device=y.device)
        self.rtol = torch.tensor(rtoll, dtype=y.dtype, device=y.device)
        self.x = x.clone()
        self.y = y.clone()
        self.KMAXX = 12
        self.IMAXX = self.KMAXX + 1
        self.nseq = torch.zeros([self.M, self.IMAXX,], dtype=torch.int64, device=y.device)
        self.cost = torch.zeros([self.M, self.IMAXX,], dtype=y.dtype, device=y.device)
        self.table = torch.zeros((self.M, self.IMAXX, self.D), dtype=y.dtype, device=y.device)
        self.dfdy = torch.zeros((self.M, self.D, self.D), dtype=y.dtype, device=y.device)
        self.dfdx = torch.zeros((self.M, self.D), dtype=y.dtype, device=y.device)
        self.calcjac = torch.tensor(self.M*[False])
        self.a = torch.zeros((self.M, self.D, self.D), dtype=y.dtype, device=y.device)
        self.coeff = torch.zeros((self.M, self.IMAXX, self.IMAXX), dtype=y.dtype, device=y.device)
        self.fsave = torch.zeros((self.M, (self.IMAXX-1)*(self.IMAXX+1)//2+2,self.D), dtype=y.dtype, device=y.device)
        self.dens = torch.zeros((self.M, (self.IMAXX+2)*self.D), dtype=y.dtype, device=y.device)
        self.factrl = torch.zeros((self.M, self.IMAXX,), dtype=y.dtype, device=y.device)
        self.costfunc = torch.ones((self.M), dtype=y.dtype, device=y.device)
        self.costjac = torch.ones((self.M), dtype=y.dtype, device=y.device)*5.0
        self.costlu = torch.ones((self.M), dtype=y.dtype, device=y.device)
        self.costsolv = torch.ones((self.M), dtype=y.dtype, device=y.device)
        self.EPS = torch.finfo(y.dtype).eps
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
        self.first_step= torch.tensor([True]*self.M)
        self.last_step = torch.tensor([False]*self.M)
        self.forward= torch.tensor([False]*self.M)
        self.reject = torch.tensor([False]*self.M)
        self.prev_reject = torch.tensor([False]*self.M)
        for k in range(self.KMAXX):
            kp1 = k+1
            self.cost[:, kp1] = self.cost[:, k] + (self.nseq[:,kp1]-1)*(self.costfunc + self.costsolv) + self.costlu
        self.hnext = -1.0e99*torch.ones((self.M), dtype=y.dtype, device=y.device)

        self.logfact = -torch.log10(self.rtol + self.atol) * 0.6 + 0.5
        #k_targ depends only on the tolerances which are the same for all samples
        maxt = (torch.max(self.one, torch.min(torch.tensor(self.KMAXX - 1), torch.round(self.logfact))).floor().long())
        self.k_targ = torch.ones([self.M],   dtype=torch.int64, device = y.device)*maxt
        print(f'setting k_targ to {self.k_targ} in constructor')

        for k in range (self.IMAXX):
            for l in range(k):
                ratio = self.nseq[:, k].to(y.dtype) / self.nseq[:, l]
                self.coeff[:, k, l] = 1.0/ (ratio-1.0)
        self.factrl[:, 0] = 1.0;
        for k in range(k < self.IMAXX-1):
            self.factrl[:, k + 1:k+1] = (k + 1) * self.factrl[:, k:k+1]
        #print(f'factrl: {self.factrl}')
        self.errold = torch.zeros((self.M), dtype=y.dtype, device=y.device)


    #htry is in [M,]
    #dyns is in [M, D]
    def step(self, htry, dyns):
        #STEPFAC1 = 0.6, STEPFAC2 = 0.93, STEPFAC3 = 0.1, STEPFAC4 = 4.0,
        #STEPFAC5 = 0.5, KFAC1 = 0.7, KFAC2 = 0.9;
        STEPFAC1, STEPFAC2, STEPFAC3, STEPFAC4, STEPFAC5, KFAC1, KFAC2 = 0.6, 0.93, 0.1, 4.0, 0.5, 0.7, 0.9
        #static bool first_step = true, last_step = false;
        #static bool forward, reject = false, prev_reject = false;
        self.errold = torch.zeros((self.M,), dtype=self.y.dtype, device=self.y.device)
        firstk = torch.BoolTensor(self.x.size()).fill_(True)
        if not isinstance(htry, torch.Tensor):
            htry = torch.tensor(htry, dtype=self.y.dtype, device=self.y.device)
        hopt = torch.zeros((self.M, self.IMAXX), dtype=self.y.dtype, device=self.y.device)
        work = torch.zeros((self.M, self.IMAXX), dtype=self.y.dtype, device=self.y.device)
        fac  = torch.zeros((self.M), dtype=self.y.dtype, device=self.y.device)

        work[:, 0:1] = 1.e30
        h = htry
        #print(f'htry={htry}')
        mask = h >= 0.0

        self.forward = mask


        ysav = self.y.clone()
        if ((h != self.hnext) & ~ self.first_step).any():
            mask = (h != self.hnext) & ~ self.first_step
            self.last_step[mask]   = True


        if self.reject.any():
            self.last_step[self.reject]   = False
            self.prev_reject[self.reject] = True
            self.theta[self.reject] = 2.0 * self.jac_redo[self.reject]

        scale = self.atol + self.rtol * torch.abs(self.y)
        self.reject = torch.tensor([False]*self.M)

        hnew = torch.abs(h)
        print(f'hnew = {hnew} set in step')
        k=torch.zeros((self.M,), dtype=torch.int64, device=self.y.device)

        def compute_jac():
            print(f'compute_jac called')
            #if (theta > jac_redo & & !calcjac) {
            #derivs.jacobian(x, y, dfdx, dfdy);
            #calcjac=true;
            #}
            nonlocal  firstk, hnew, scale, mask
            nonlocal  work, hopt, k, ysav
            #if (theta > jac_redo & & !calcjac) {
            #derivs.jacobian(x, y, dfdx, dfdy);
            #calcjac=true;
            #}
            m1 = (self.theta > self.jac_redo) & ~ self.calcjac & (torch.abs(h) > self.EPS)
            if torch.any(m1):
              self.dfdx[m1, :], self.dfdy[m1, :] = dyns.jacobian(self.x[m1], self.y[m1,:])
              self.calcjac[m1] = True
            hc = hnew.clone()
            while (firstk | self.reject).any():
                bmask =  (torch.abs(h) > self.EPS)
                m2 = (firstk | self.reject) & bmask
                if ~m2.all():
                    break
                mf = m2 & self.forward
                if mf.any():
                    hc[mf] = hnew[mf]
                mr = m2 & ~(self.forward)
                if mr.any():
                    hc[mr] = -hnew[mr]

                firstk[m2]= False
                self.reject[m2]= False
                #if (torch.abs(h) <= torch.abs(self.x)*self.EPS).any():
                #    raise ValueError('Step size too small in seulex')
                k = torch.zeros([self.M], dtype=torch.int64, device=self.y.device)
                while (m2.any() & (k[m2]  <= (self.k_targ[m2]+1)).any()):
                    if (~bmask).all():
                        break #All the break conditions have been met so break out of the loop
                              #otherwise we will be stuck forever
                    #print(f'bmask={bmask}')
                    m21 = m2 & bmask
                    m21[m2] = (k[m2] < (self.k_targ[m2]+2))
                    if m21.any():
                        #    def dy(self, y, x, htot, k, scale, theta, dfdy, dyns):
                        if torch.allclose(self.x[0], torch.tensor(1.36), atol=1e-2):
                            print(f'self.x is close to 1.36')
                        success, self.theta[m21], yseq = self.dy(ysav[m21, :], self.x[m21], hc[m21], k[m21], self.nseq[m21, :],  \
                                                scale[m21, :], self.theta[m21], self.dfdy[m21, :,  :], dyns)
                        print(f'self.theta after dy={self.theta}')
                        print(f'yseq after dy={yseq}')
                        #print(f'k={k}')
                        #print(f'k_targ={self.k_targ}')
                        #if not success:
                        #    self.reject = True
                        #    hnew = torch.abs(h) * STEPFAC5
                        #    break
                        #if k == 0:
                        #    self.y = yseq
                        #else:
                       #     self.table[k-1][:] = yseq.clone()
                        #Here success is a break mask
                        m211 = m21 & bmask
                        m211[m21] = (~success)
                        if m211.any():
                            hnew[m211] = torch.abs(hc[m211]) * STEPFAC5
                            print(f'hnew[m211] = {hnew[m211]} set in m211')
                            self.reject[m211] = True

                            bmask[m211] = ~m211[m211]
                        m212 = m21 & bmask
                        m212[m21] = (k[m21] ==0) & bmask[m21]
                        if m212.any():

                            self.y[m212,:] = yseq[m212[m21],:]

                        m213 = m21 & bmask
                        m213[m21] = (k[m21] != 0) & bmask[m21]
                        if m213.any():
                            try:
                                self.table[m213, k[m213]-1, :] = yseq[m213[m21]] #This is a copy
                            except (RuntimeError, IndexError) as e:
                                 junk = 'True'
                            ytemp = self.y.clone()
                            self.y[m213,:], self.table[m213, :, :] = self.polyextr(k[m213], self.y[m213,:], self.table[m213, :, :], self.coeff[m213, :, :])
                            print(f'table: {self.table}')
                            scale[m213, :] = self.atol + self.rtol * torch.abs(ysav[m213, :])
                            err = torch.zeros((self.M), dtype=self.y.dtype, device=self.y.device)
                            #err = torch.sqrt(torch.sum(torch.square((self.y-self.table[0][:]) / scale))/self.n)
                            #err[mask] = torch.sqrt(torch.sum(torch.square((self.y[mask, :] - self.table[mask, 0, :]) / scale[mask, :]), dim=0) / self.D)
                            print(f'y={self.y[m213]}')
                            delta = (self.y[m213] - self.table[m213,0])
                            print(f'delta={delta}')
                            sq_inpt = delta/scale[m213]
                            print(f'sq_inpt={sq_inpt}')
                            sq_term = torch.square(sq_inpt)
                            print(f'sq_term={sq_term}')
                            err[m213] = torch.sqrt(torch.sum(sq_term, dim=1) / self.D)


                            print(f'err={err}')

                            print(f'scale={scale[m213]}')
                            print(f'ysav={ysav[m213]}')
                            t1 = ((torch.ones_like(err[m213])*1.0 / self.EPS))
                            j2 = err[m213]
                            #m2131 = m213 & bmask
                            m2131 = m213 & bmask
                            #err > 1.0/self.EPS or ( k>1 and err >= self.errold)
                            m2131[m213] = (j2 > t1) | ((k[m213] > 1) & (err[m213] >= self.errold[m213])) & bmask[m213]
                            if m2131.any():
                                #print(f'mask before self.reject = {mask}')
                                self.reject[m2131] = True
                                hnew[m2131] = torch.abs(hc[m2131]) * STEPFAC5
                                print(f'hnew[m2131] = {hnew[m2131]} set in m2131')
                                bmask[m2131] = bmask[m2131] & ~m2131[m2131]

                            m213 = m21 & bmask
                            if m213.any():
                                self.errold[m213] = torch.max(4.0 * err[m213], self.one)
                                expo = self.one / (k + 1)
                                print(f'expo={expo}')
                                facmin = torch.pow(STEPFAC3, expo)
                                m2132 = m213 & bmask
                                m2132[m213] = (err[m213] == 0.0)
                                if m2132.any():
                                    fac[m2132] =  facmin[m2132].reciprocal()
                                m2133 = ~m2132 & bmask
                                if m2133.any():
                                    #fac = STEPFAC2 / torch.pow(err / STEPFAC1, expo)
                                    #fac = torch.max(facmin / STEPFAC4, torch.min(facmin.reciprocal(), fac))
                                    fac[m2133] = STEPFAC2 / torch.pow(err[m2133] / STEPFAC1, expo[m2133])
                                    fac[m2133] = torch.max(facmin[m2133] / STEPFAC4, torch.min(facmin[m2133].reciprocal(), fac[m2133]))
                                    print(f'fac: {fac}')
                                    print(f'facmin: {facmin}')
                                    if torch.allclose(facmin, torch.tensor(0.630), atol=1e-3):
                                        print(f'facmin is close to 0.630')
                                hopt[m213, k[m213]] = torch.abs(hc[m213] * fac[m213])
                                work[m213, k[m213]] = (self.cost[m213, k[m213]] / hopt[m213, k[m213]])
                                m2134 = m213 & bmask
                                m2134[m213] = (self.first_step[m213] | self.last_step[m213]) & (err[m213] <= 1.0)
                                if m2134.any():
                                    bmask[m2134] = bmask[m2134] & ~m2134[m2134]
                                m2135 = m213 & bmask
                                m2135[m213] = ((k[m213] == (self.k_targ[m213] - 1)) & ~ self.prev_reject[m213] &\
                                             ~ self.first_step[m213] & ~ self.last_step[m213])
                                if m2135.any():
                                    m21351 = m2135 & (err[m2135] <=1.0) & bmask[m2135]
                                    if m21351.any():
                                        bmask[m21351] = bmask[m21351] & ~m21351[m21351]
                                    #elif err > self.nseq[self.k_targ]*self.nseq[self.k_targ+1]*4.0:
                                    m21352 = m2135 & bmask[m2135]
                                    if m21352.any():
                                        m21352[m2135] = ((err[m2135]) > (self.nseq[m2135, self.k_targ[m2135]]*self.nseq[m2135, self.k_targ[m2135]+1]*4.0) )
                                    if m21352.any():
                                        self.reject[m21352] = True
                                        self.k_targ[m21352] = k[m21352]
                                        print(f'setting k_targ in m21352 = {self.k_targ[m21352]}')
                                        m213521 = (self.k_targ[m21352] > 1) & (work[m21352, k[m21352] - 1] < KFAC1 * work[m21352, k[m21352]]) & bmask
                                        if m213521.any():
                                            self.k_targ[m213521] = self.k_targ[m213521] - 1
                                            print(f'setting k_targ in m213521 = {self.k_targ[m213521]}')
                                        hnew[m21352] = hopt[m21352, self.k_targ[m21352]]
                                        print(f'setting hnew in m21352 = {hnew[m21352]}')
                                        bmask[m21352] = bmask[m21352] & ~m21352[m21352]
                                m2136 = m213 & bmask
                                m2136[m213] = (k[m213] == self.k_targ[m213])
                                if m2136.any():
                                    m21361 = (err <=1.0) & bmask
                                    if m21361.any():
                                        bmask[m21361] = bmask[m21361] & ~m21361[m21361]
                                    #There is a break statement here
                                    #We need to update m2136
                                    m2136 = m2136 & bmask
                                    if m2136.any():
                                        temp = self.nseq[m2136, k[m2136] + 1] * 2.0
                                        m21362 = m2136 & bmask
                                        m21362[m2136] = (err[m2136] > temp)
                                        if m21362.any():
                                            self.reject[m21362] = True
                                            m213621 = m21362 & bmask
                                            m213621[m21362]= (self.k_targ[m21362] > 1) & (work[m21362, k[m21362]-1] < KFAC1*work[m21362, k[m21362]])
                                            if m213621.any():
                                                self.k_targ[m213621] = self.k_targ[m213621] - 1
                                                print(f'setting k_targ in m213621 = {self.k_targ[m213621]}')
                                            hnew[m21362] = hopt[m21362, self.k_targ[m21362]]
                                            print(f'setting hnew in m21362 = {hnew[m21362]}')
                                            if m21362.any():
                                                bmask[m21362] = bmask[m21362] & ~m21362[m21362]
                                            #print(f'hnew shape = {hnew.shape}')
                            m213 = m2 & m21 & m213 & bmask
                            if m213.any():
                                m2137 = m213 & bmask
                                m2137[m213] = k[m213] == self.k_targ[m213] + 1
                                if m2137.any():
                                    m21371 =  m2137 & bmask
                                    m21371[m2137] = (err[m2137] > 1.0)
                                    if m21371.any():
                                        self.reject[m21371] = True
                                        #if self.k_targ > 1 and work[self.k_targ - 1] < KFAC1 * work[self.k_targ]:
                                        temp  = self.k_targ[m21371]
                                        m213711 = m21371 & bmask
                                        m213711[m21371] = (temp > 1) & (work[m21371, temp - 1] < KFAC1 * work[m21371, temp])
                                        if m213711.any():
                                            m213711 = m2 & m21 & m213 & m2137 & m21371 & m213711 & bmask
                                            self.k_targ[m213711] = self.k_targ[m213711] - 1
                                            print(f'setting k_targ in m213711 = {self.k_targ[m213711]}')
                                        hnew[m21371] = hopt[m21371, self.k_targ[m21371]]
                                        print(f'setting hnew in m21371 = {hnew[m21371]}')
                                        #print(f'hnew shape = {hnew.shape}')
                                    bmask[m2137] = bmask[m2137] & ~m2137[m2137]
                    m21 = m21 & bmask #Apply the break mask
                    if m21.any():
                        print(f'k = {k}')
                        print(f'k_targ = {self.k_targ}')
                        k[m21]=k[m21]+1
                    m2 = m2 & bmask #This is important
                m3 = self.reject
                if m3.any():
                    self.prev_reject[m3] = True
                    m31 = ~self.calcjac
                    if m31.any():
                        self.theta[m31] = self.theta[m31] * self.jac_redo[m31]
                        compute_jac()
        compute_jac()
        self.calcjac = torch.BoolTensor(self.x.size()).fill_(False)
        self.xold = self.x
        self.x = self.x + h
        self.hdid = h.clone() #Need to clone because h is a tensor
        print(f'hdid={self.hdid} after compute_jac')

        self.first_step = torch.BoolTensor(self.x.size()).fill_(False)
        ms1 = (k == 1)
        kopt = k.clone() #Need to be careful here as
        if ms1.any():
            print(f'ms1')
            kopt[ms1] = 2
        ms2 = k <= self.k_targ
        if ms2.any():
            print(f'ms2')
            kopt[ms2] = k[ms2]
            kms2 = k[ms2]
            ms21 = ms2.clone()
            ms21[ms2] =  (work[ms2, k[ms2] - 1] < (work[ms2, kms2]*KFAC1))
            if ms21.any():
                print(f'ms21')
                mask = ms2 & ms21
                kopt[mask] = k[mask] - 1
            t1 = work[ms2, k[ms2]]
            t2 = (KFAC2 * work[ms2, k[ms2] - 1])
            ms22 = ms2.clone()
            ms22[ms2] = ( t1 < t2)
            if ms22.any():
                print(f'ms22')
                kopt[ms22] = torch.min(k[ms22] + 1, torch.tensor(self.KMAXX))
        ms3 = (~ ms1) & (~ ms2)
        if ms3.any():
            print(f'ms3')
            kopt[ms3] = k[ms3] - 1
            ms31 = ms3.clone()
            ms31[ms3] =  (k[ms3] > 2) & (work[ms3, k[ms3] - 2] < KFAC1 * work[ms3, k[ms3] - 1])
            if ms31.any():
                print(f'ms31')
                kopt[ms31]=k[ms31] - 2
            ms32 = ms3.clone()
            ms32[ms3] = (work[ms3, k[ms3]] < KFAC2 * work[ms3, kopt[ms3]])
            if ms32.any():
                print(f'ms32')
                mask = ms3 & ms32
                kopt[mask] = torch.min(k[mask], torch.ones_like(k[mask])*self.KMAXX)
        ms4 = self.prev_reject.clone()
        if ms4.any():
            print(f'ms4')
            self.k_targ[ms4] = torch.min(kopt[ms4], k[ms4])
            print(f'setting k_targ in ms4 = {self.k_targ[ms4]}')
            print(f'kopt[ms4] = {kopt[ms4]}')
            print(f'k[ms4] = {k[ms4]}')
            hnew[ms4] = torch.min(torch.abs(h[ms4]), hopt[ms4, self.k_targ[ms4]])
            print(f'setting hnew in ms4 = {hnew[ms4]}')
            #print(f'hnew shape = {hnew.shape}')
            self.prev_reject[ms4] = False
        ms5 = ~ms4
        if ms5.any():
            print(f'ms5')
            ms51 = ms5 & (kopt <= k)
            if ms51.any():
                print(f'ms51')
                hnew[ms51] = hopt[ms51, kopt[ms51]]
                print(f'setting hnew in ms51 = {hnew[ms51]}')
            ms52 = ms5 & (kopt > k)
            if ms52.any():
                print(f'ms52')
                # if k < self.k_targ and work[k] < KFAC2 * work[k - 1]:
                t1 = (k[ms52] < self.k_targ[ms52])
                t2 = (work[ms52, k[ms52]] < KFAC2 * work[ms52, k[ms52] - 1])
                ms521 = ms52.clone()
                ms521[ms52]= ms52[ms52] & t1 & t2
                if ms521.any():
                    print(f'ms521')
                    hnew[ms521] = hopt[ms521, k[ms521]] * self.cost[ms521, kopt[ms521] + 1] / self.cost[ms521, k[ms521]]
                    print(f'setting hnew in ms521 = {hnew[ms521]}')
                ms522 = ~ms521
                if ms522.any():
                    print(f'ms522')
                    hnew[ms522] = hopt[ms522, k[ms522]] * self.cost[ms522, kopt[ms522]] / self.cost[ms522, k[ms522]]
                    print(f'setting hnew in ms522 = {hnew[ms522]}')
            if ms5.any():
                self.k_targ[ms5] = kopt[ms5]
                print(f'setting k_targ in ms5 = {self.k_targ[ms5]}')
        forwards = self.forward
        if forwards.any():
            self.hnext[forwards] = hnew[forwards]
        if ~forwards.any():
            self.hnext[~forwards] = -hnew[~forwards]






    # Here y in [M, D]
    # htot in [M]
    # k is a scalar
    # scale in [M, D]
    def dy_bak(self, y, x, htot, k, nseq, scale, theta, dfdy, dyns):
        #print(f'k={k}')
        nsteps = nseq[torch.arange(nseq.shape[0]), k] #This is of dimension [M]
        h = htot*nsteps.reciprocal()
        a = -dfdy
        identity_matrix = torch.eye(self.D)
        identity_matrix_expanded = identity_matrix.unsqueeze(0).expand(k.shape[0], self.D, self.D)
        try:
            diagonal_tensor = torch.einsum('m, mij->mij', h.reciprocal(), identity_matrix_expanded)
        except RuntimeError:
            print(f'k={k}')
            print(f'h={h}')
            print(f'h.reciprocal()={h.reciprocal()}')
            print(f'identity_matrix_expanded={identity_matrix_expanded}')
            raise RuntimeError
        a = a + diagonal_tensor


        alu = QRBatch(a)
        xnew = x + h
        ytemp = y
        delta = dyns(xnew, ytemp)
        tdelta=alu.solvev(delta)

        #ax = torch.einsum('mij,mj->mi', a, sdelta)
        #print(f'a*x={ax}')
        #print(f'Error in LU={torch.norm(ax-delta)}')
        delta=tdelta
        #This creates a set of indexes to iterate over

        #Find the maximum number of steps across all samples
        mstep = torch.max(nsteps)
        yend = torch.zeros_like(y)
        for nn in range(1, mstep):
            #Here nstep is of dimension [M]
            #Find all the indexes where nn is less than nsteps
            #This is essentially a filter

            masknn = nsteps > torch.tensor(nn)
            ytemp[masknn,:] = ytemp[masknn,:] + delta[masknn,:]
            xnew[masknn] += h[masknn]
            yend[masknn, :] = dyns(xnew[masknn], ytemp[masknn, :])
            #print(f'masknn={masknn}')
            if nn == 1 & (k[masknn] <= 1).any():
                maskif = (masknn & nn == 1) & ((k & masknn) <= 1)
                #print(f'maskif={maskif}')
                del1 = torch.sqrt(torch.sum(torch.square((delta[maskif, :] / scale[maskif, :]) ), dim=1))
                #print(f'del1={del1}')
                dytemp = dyns(x[maskif] + h[maskif], ytemp[maskif,:])
                dh = torch.einsum('mi, m->mi', delta[maskif,:], h[maskif].reciprocal())
                delta = dytemp[maskif,:]-dh
                tdelta = alu.solvev(delta)
                ax = torch.einsum('mij,mj->mi', a, tdelta)
                # print(f'a*x={ax}')
                #print(f'Error in LU={torch.norm(ax - delta)}')
                delta = tdelta

                del2 = torch.sqrt(torch.sum(torch.square(delta[maskif,:] / scale[maskif,:]), dim=1))
                #print(f'del2={del2}')

                theta[maskif] = del2 / torch.max(torch.ones_like(del1), del1)
                if (theta[maskif] > 1.0).all():
                    return torch.BoolTensor(x.size()).fill_(False), theta, yend
            tdelta[masknn]=alu.solvef(masknn, yend)
            #ax = torch.einsum('mij,mj->mi', a[masknn,:,:], tdelta)
            # print(f'a*x={ax}')
            #print(f'Error in LU={torch.norm(ax - y[masknn,:])}')
            delta[masknn,:] = tdelta[masknn]

        if (~ torch.isfinite(yend)).any():
            junk = True
        yend=ytemp + delta
        return torch.BoolTensor(self.x.size()).fill_(True), theta, yend

    def dy(self, y, x, htot, k, nseq,  scale, theta, dfdy, dyns):
        ks = k.tolist()
        res=torch.BoolTensor(x.size()).fill_(False)
        yend = torch.zeros_like(y)
        for i in range(len(ks)):
            nstep = nseq[i, k[i]]
            h = htot[i] / nstep
            a = -dfdy[i]
            a.diagonal()[:] += torch.reciprocal(h)

            aqr = QRBatch(a)
            xnew = x[i] + h
            delta = torch.squeeze(dyns(torch.unsqueeze(xnew, 0), torch.unsqueeze(y[i], 0)))
            ytemp = y[i].clone()
            tdelta=aqr.solvev(delta)

            #ax = torch.einsum('ij,j->i', a, tdelta)
            #print(f'a*x={ax}')
            #print(f'Error in LU={torch.norm(ax-delta)}')
            delta=tdelta
            res[i] = True
            for nn in range(1, nstep):
                ytemp[:]=ytemp[:]+delta[:]
                xnew += h
                yend[i] = torch.squeeze(dyns(torch.unsqueeze(xnew, 0), torch.unsqueeze(ytemp, 0)))
                if nn == 1 and ks[i] <= 1:
                    del1 = torch.sqrt(torch.sum(torch.square((delta / scale[i]) )))
                    dytemp = torch.squeeze(dyns(torch.unsqueeze(x[i] + h, 0), torch.unsqueeze(ytemp, 0)))
                    delta = dytemp- delta / h
                    tdelta = aqr.solvev(delta)

                    #ax = torch.einsum('ij,j->i', a, tdelta)
                    # print(f'a*x={ax}')
                    #print(f'Error in LU={torch.norm(ax - delta)}')
                    delta = tdelta

                    del2 = torch.sqrt(torch.sum(torch.square(delta / scale[i])))

                    theta[i] = del2 / torch.max(torch.ones_like(del1), del1)
                    if theta[i] > 1.0:
                        res[i] = False
                        break

                tdelta=aqr.solvev(yend[i])
                #ax = torch.einsum('ij,j->i', a, tdelta)
                # print(f'a*x={ax}')
                #print(f'Error in LU={torch.norm(ax - yend)}')
                delta = tdelta
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

