import torch
from qr import QR
torch.set_printoptions(precision=16)
class Seulex:
    #The inputs are the initial conditions y, the absolute tolerance atoll, and the relative tolerance rtoll
    def __init__(self, y, x, atoll, rtoll):
        if isinstance(atoll, torch.Tensor):
            self.atol = atoll
        else:
            self.atol = torch.tensor(atoll, dtype = y.dtype, device=y.device)
        if isinstance(rtoll, torch.Tensor):
            self.rtol = rtoll
        else:
            self.rtol = torch.tensor(rtoll, dtype = y.dtype, device=y.device)
        self.x = x.clone()
        self.y = y.clone()
        self.KMAXX = 12
        self.IMAXX = self.KMAXX + 1
        self.nseq = torch.zeros([self.IMAXX,], dtype=torch.int64, device=y.device)
        self.cost = torch.zeros([self.IMAXX,], dtype=y.dtype, device=y.device)
        self.n = y.shape[0]
        self.table = torch.zeros((self.IMAXX, self.n), dtype=y.dtype, device=y.device)
        self.dfdy = torch.zeros((self.n, self.n), dtype=y.dtype, device=y.device)
        self.dfdx = torch.zeros((self.n), dtype=y.dtype, device=y.device)
        self.calcjac = False
        self.a = torch.zeros((self.n, self.n), dtype=y.dtype, device=y.device)
        self.coeff = torch.zeros((self.IMAXX, self.IMAXX), dtype=y.dtype, device=y.device)
        self.fsave = torch.zeros(((self.IMAXX-1)*(self.IMAXX+1)//2+2,self.n), dtype=y.dtype, device=y.device)
        self.dens = torch.zeros(((self.IMAXX+2)*self.n), dtype=y.dtype, device=y.device)
        self.factrl = torch.zeros((self.IMAXX,), dtype=y.dtype, device=y.device)
        self.costfunc = torch.tensor(1.0, dtype=y.dtype, device=y.device)
        self.costjac = torch.tensor(5.0, dtype=y.dtype, device=y.device)
        self.costlu = torch.tensor(1.0, dtype=y.dtype, device=y.device)
        self.costsolv = torch.tensor(1.0, dtype=y.dtype, device=y.device)
        self.EPS = torch.finfo(y.dtype).eps
        self.jac_redo =torch.min(torch.tensor(1.0e-4, dtype=y.dtype, device=y.device), self.rtol)
        self.theta = 2.0*self.jac_redo
        self.one = torch.tensor(1, dtype=torch.int64, device=y.device)
        self.nseq[0] = 2
        self.nseq[1] = 3
        for i in range(2, self.IMAXX):
            self.nseq[i] = 2 * self.nseq[i - 2]
        self.first_step, self.last_step = True, False
        self.forward, self.reject, self.prev_reject = False, False, False
        #cost[0] = costjac + costlu + nseq[0] * (costfunc + costsolve);
        #for (Int k=0;k < KMAXX;k++)
        #    cost[k + 1] = cost[k] + (nseq[k + 1] - 1) * (costfunc + costsolve) + costlu;
        self.cost[0] = self.costjac + self.costlu + self.nseq[0] * (self.costfunc + self.costsolv)
        #vectorize this loop
        for k in range(self.KMAXX):
            self.cost[k + 1] = self.cost[k] + (self.nseq[k + 1] - 1) * (self.costfunc + self.costsolv) + self.costlu

        self.hnext = torch.tensor(-1.0e99, dtype=y.dtype, device=y.device)

        self.logfact = -torch.log10(self.rtol + self.atol) * 0.6 + 0.5
        self.k_targ = torch.max(self.one, torch.min(torch.tensor(self.KMAXX - 1), torch.round(self.logfact))).floor().long()
        for k in range (self.IMAXX):
                ratios = self.nseq[k].to(y.dtype) / self.nseq[:k]
                self.coeff[k, :k] = 1.0/ (ratios-1.0)

        self.factrl[0] = 1.0
        self.errold = 0.0

        k_values = torch.arange(1, self.IMAXX, dtype=self.factrl.dtype, device=self.factrl.device)
        self.factrl[1:self.IMAXX] = torch.cumprod(k_values, dim=0)
        #print(f'factrl: {self.factrl}')

    def step(self, htry, dyns):
        #STEPFAC1 = 0.6, STEPFAC2 = 0.93, STEPFAC3 = 0.1, STEPFAC4 = 4.0,
        #STEPFAC5 = 0.5, KFAC1 = 0.7, KFAC2 = 0.9;
        #STEPFAC1, STEPFAC2, STEPFAC3, STEPFAC4, STEPFAC5, KFAC1, KFAC2 = 0.6, 0.93, 0.1, 4.0, 0.5, 0.7, 0.9
        #static bool first_step = true, last_step = false;
        #static bool forward, reject = false, prev_reject = false;
        STEPFAC1 = torch.tensor(0.6, dtype=torch.float64, device=self.y.device)
        STEPFAC2 = torch.tensor(0.93, dtype=torch.float64, device=self.y.device)
        STEPFAC3 = torch.tensor(0.1, dtype=torch.float64, device=self.y.device)
        STEPFAC4 = torch.tensor(4.0, dtype=torch.float64, device=self.y.device)
        STEPFAC5 = torch.tensor(0.5, dtype=torch.float64, device=self.y.device)
        KFAC1 = torch.tensor(0.7, dtype=torch.float64, device=self.y.device)
        KFAC2 = torch.tensor(0.9, dtype=torch.float64, device=self.y.device)
        self.errold = torch.zeros((1,), dtype=self.y.dtype, device=self.y.device)

        firstk = True
        if not isinstance(htry, torch.Tensor):
            htry = torch.tensor(htry, dtype=self.y.dtype, device=self.y.device)
        hopt = torch.zeros((self.IMAXX), dtype=self.y.dtype, device=self.y.device)
        work = torch.zeros((self.IMAXX), dtype=self.y.dtype, device=self.y.device)
        scale = torch.zeros((self.n), dtype=self.y.dtype, device=self.y.device)
        work[0] = torch.tensor(1.0e30, dtype=torch.float64, device=self.y.device)
        h = htry.clone()
        if h > 0:
            self.forward = True
        else:
            self.forward = False
        ysav = self.y.clone()
        if h != self.hnext and not self.first_step:
            self.last_step = True
        if self.reject:
            self.prev_reject = True
            self.last_step = False
            self.theta = 2.0 * self.jac_redo
        scale = self.atol + self.rtol * torch.abs(self.y)
        self.reject = False
        firstk = True
        hnew = torch.abs(h)
        #if torch.allclose(hnew, torch.tensor(0.168978), atol=1e-6):
        #    print(f'hnew is close to {hnew}')
        print(f'hnew set in step = {hnew} set in step')
        k = 0
        self.errold = 0.0
        def compute_jac():
            print(f'compute_jac called')
            #if (theta > jac_redo & & !calcjac) {
            #derivs.jacobian(x, y, dfdx, dfdy);
            #calcjac=true;
            #}
            nonlocal  firstk, hnew, scale
            nonlocal  work, hopt, k, ysav
            #if (theta > jac_redo & & !calcjac) {
            #derivs.jacobian(x, y, dfdx, dfdy);
            #calcjac=true;
            #}
            while firstk or self.reject:

                if self.theta > self.jac_redo and not self.calcjac:
                    # m1
                    print(f'm1 called')
                    self.dfdx, self.dfdy = dyns.jacobian(self.x, self.y)
                    self.calcjac = True

                breakmask1 = ~(self.x == self.x) #Local to this method
                #m2
                print(f'm2 called')
                if (self.forward):
                    h = hnew.clone()
                else:
                    h = -hnew.clone()
                firstk = False
                self.reject = False
                if torch.abs(h) <= torch.abs(self.x)*self.EPS:
                    print('Step size too small in seulex')
                k = k*0
                while k <= (self.k_targ+1):

                    breakmask11 = ~(self.x == self.x) #Local to this method
                    #m21
                    print(f'm21 called')
                    #dy(self, x, y, htot, k, nseq, dfdy, scale, theta, dyns):
                    #if torch.allclose(self.x, torch.tensor(6.563035), atol=1e-6):
                    #    print(f'self.x is close to 6.563035')
                    success, self.theta, yseq = self.dy(self.x, ysav, h, k, self.nseq, self.dfdy, scale, self.theta, dyns)
                    #if torch.allclose(self.theta, torch.tensor(0.073011), atol=1e-6):
                    #    print(f'theta is close to {self.theta}')
                    if torch.allclose(yseq, torch.tensor([0.6523669780372592, 0.7705372945176566]).double(), rtol=1e-6,
                                      atol=1e-6):
                        print(f'{yseq} close to {torch.tensor([0.6491064438078820, 0.7650675874840187])}')

                    print(f'self.theta after dy={self.theta}')
                    print(f'yseq after dy={yseq}')

                    #print(f'k={k}')
                    #print(f'k_targ={self.k_targ}')
                    if not success:
                        #m211
                        print(f'm211 called')
                        self.reject = True
                        hnew = torch.abs(h) * STEPFAC5
                        print(f'hnew={hnew} set in m211')
                        break
                    if k == 0:
                        #m212
                        print(f'm212 called')
                        self.y = yseq

                    else:
                        #m213
                        print(f'm213 called')
                        self.table[k-1][:] = yseq
                        #if torch.allclose(yseq[1], torch.tensor(0.252), atol=1e-3):
                        #    print(f'yseq[0][0] is close to 1.21')
                        self.y, self.table=self.polyextr(k, self.y, self.table, self.coeff)
                        #print(f'coeff: {self.coeff}')
                        print(f'table: {self.table}')
                        scale = self.atol + self.rtol * torch.abs(ysav)
                        print(f'y={self.y}')
                        delta = (self.y - self.table[0][:])
                        print(f'delta={delta}')
                        sq_inpt= delta/ scale
                        print(f'sq_inpt={sq_inpt}')
                        sq_term = torch.square(sq_inpt)
                        print(f'sq_term={sq_term}')
                        err = torch.sqrt(torch.sum(sq_term)/self.n)
                        print(f'err={err}')
                        #if torch.allclose(self.y[0], torch.tensor(0.896298), atol=1e-6):
                        #    print(f'y[0][0] is close to 0.896298')
                        print(f'scale={scale}')
                        print(f'ysav={ysav}')
                        if err > 1.0/self.EPS or ( k>1 and err >= self.errold):
                            #m2131
                            print(f'm2131 called')
                            self.reject = True
                            hnew = torch.abs(h) * STEPFAC5
                            print(f'hnew={hnew} set in m2131')
                            if torch.allclose(torch.tensor(hnew), torch.tensor(0.0009137).double(), atol=1e-6):
                                print(f'hnew = {hnew}')
                            break
                        self.errold = torch.max(4.0*err, self.one)
                        expo = self.one/ torch.tensor((k+1), dtype=torch.float64, device=self.y.device)
                        print(f'expo={expo}')
                        facmin = torch.pow(STEPFAC3, expo)
                        #if (err == 0.0)
                        #    fac = 1.0 / facmin;
                        #else {
                        #fac=STEPFAC2/pow(err/STEPFAC1, expo);
                        #fac=MAX(facmin/STEPFAC4, MIN(1.0 /facmin, fac));
                        #}
                        if err == 0.0:
                            #m2132
                            print(f'm2132 called')
                            fac = facmin.reciprocal()
                        else:
                            # m2133
                            print(f'm2133 called')
                            fac = STEPFAC2 / torch.pow(err/STEPFAC1, expo)
                            fac = torch.max(facmin/STEPFAC4, torch.min(facmin.reciprocal(), fac))
                            print(f'fac: {fac}')
                            print(f'facmin: {facmin}')

                        hopt[k] = torch.abs(h*fac)
                        #if torch.allclose(hopt[k], torch.tensor(0.734853), atol=1e-6):
                        #    print(f'hopt is close to 0.734853]')

                        print(f'hopt[{k}]={hopt[k]}')
                        work[k] = self.cost[k] / hopt[k]
                        print(f'work={work}')
                        if (self.first_step or self.last_step) and err <= 1.0:
                            #m2134
                            print(f'm2134 called')
                            break
                        if k == self.k_targ - 1 and not self.prev_reject and not self.first_step and not self.last_step:
                            #m2135
                            print(f'm2135 called')
                            if err <= 1.0:
                                #m21351
                                print(f'm21351 called')
                                break
                            elif err > self.nseq[self.k_targ]*self.nseq[self.k_targ+1]*4.0:
                                #m21352
                                print(f'm21352 called')
                                self.reject=True
                                self.k_targ=k
                                print(f'setting k_targ to {k} in m21352')
                                if self.k_targ > 1 and work[k-1] < KFAC1*work[k]:
                                    #m213521
                                    print(f'm213521 called')
                                    self.k_targ = self.k_targ-1
                                hnew = hopt[self.k_targ]
                                print(f'hnew={hnew} set in m213521')
                                break
                        if k == self.k_targ:
                            #m2136
                            print(f'm2136 called')
                            if err <= 1.0:
                                #m21361
                                print(f'm21361 called')
                                break
                            elif err > self.nseq[k+1]*2.0:
                                #m21362
                                print(f'm21362 called')
                                self.reject = True
                                if self.k_targ > 1 and work[k-1] < KFAC1*work[k]:
                                    #m213621
                                    print(f'm213621 called')
                                    self.k_targ = self.k_targ-1
                                    print(f'setting k_targ to {self.k_targ} in m213621')
                                hnew = hopt[self.k_targ]
                                print(f'hnew={hnew} set in m213621')
                                break
                        if k == self.k_targ+1:
                            #m2137
                            print(f'm2137 called')
                            if err > 1.0:
                                #m21371
                                print(f'm21371 called')
                                self.reject = True
                                if self.k_targ > 1 and work[self.k_targ-1]< KFAC1*work[self.k_targ]:
                                    #m213711
                                    print(f'm213711 called')
                                    self.k_targ = self.k_targ-1
                                    print(f'setting k_targ to {self.k_targ} in m213711')
                                hnew = hopt[self.k_targ]
                                print(f'hnew={hnew} set in m213711')
                                if torch.allclose(hnew, torch.tensor(0.00091373).double(), atol=1e-8):
                                    print(f'hnew = {hnew}')
                            print(f'break called from m2137')
                            break
                    print(f'k={k}')
                    print(f'k_targ = {self.k_targ}')
                    k = k+1
                    print(f'Incremented k to {k}')
                if self.reject:
                    #m3
                    print(f'm3 called')
                    self.prev_reject = True
                    if not self.calcjac:
                        #m31
                        print(f'm31 called')
                        self.theta = self.theta * self.jac_redo
                        #compute_jac()
        compute_jac()
        print(f'Leaving compute_jac with k = {k}')
        self.calcjac = False
        self.xold = self.x
        self.x = self.x + h
        self.hdid = h
        print(f'hdid={self.hdid} after compute_jac')
        print(f'k={k} after compute_jac')
        #if torch.allclose(self.hdid, torch.tensor(0.499490), 1.0e-6):
        #    print(f'hdid close to {self.hdid}')
        self.first_step = False
        if k==0:
            print(f'k==0')
        if k == 1:
            #ms1
            print(f'ms1')
            kopt = 2
        elif k <= self.k_targ:
            print(f'ms2')
            #ms2
            kopt = k
            if work[k - 1] < KFAC1 * work[k]:
                #ms21
                print(f'ms21')
                kopt = k - 1
            elif work[k] < KFAC2 * work[k - 1]:
                #ms22
                print(f'ms22')
                kopt = min(k + 1, self.KMAXX - 1)
        else:
            #ms3
            print(f'ms3')
            kopt = k - 1
            if k > 2 and work[k - 2] < KFAC1 * work[k - 1]:
                #ms31
                print(f'ms31')
                kopt = k - 2
            if work[k] < KFAC2 * work[kopt]:
                #ms32
                print(f'ms32')
                kopt = min(k, self.KMAXX - 1)
        if self.prev_reject:
            #ms4
            print(f'ms4')
            self.k_targ = min(kopt, k)
            print(f'setting k_targ to {self.k_targ} in ms4')
            print(f'kopt in ms4 = {kopt}')
            print(f'k in ms4 = {k}')
            hnew = torch.min(torch.abs(h), hopt[self.k_targ])
            print(f'hnew: {hnew} set in ms4')
            #if torch.allclose(hnew, torch.tensor(0.307131), 1.0e-6):
            #    print(f'hnew close to {hnew}')
            self.prev_reject = False
        else:
            #ms5
            print(f'ms5')
            if kopt <= k:
                #ms51
                print(f'ms51')
                hnew = hopt[kopt]
                print(f'hnew: {hnew} set in ms51')
            else:
                #ms52
                print(f'ms52')
                if k < self.k_targ and work[k] < KFAC2 * work[k - 1]:
                    #ms521
                    print(f'ms521')
                    print(f'hopt={hopt}')
                    hnew = hopt[k] * self.cost[kopt + 1] / self.cost[k]
                    print(f'hnew: {hnew} set in ms521')
                else:
                    #ms522
                    print(f'ms522')
                    hnew = hopt[k] * self.cost[kopt] / self.cost[k]
                    print(f'hnew: {hnew} set in ms522')
            self.k_targ = kopt
            print(f'setting k_targ to {self.k_targ} in ms5')
        if self.forward:
            #ms6
            self.hnext = hnew
        else:
            #ms7
            self.hnext = -hnew

    def dy(self, x, y, htot, k, nseq, dfdy, scale, theta, dyns):
        nstep = nseq[k]
        h = htot / nstep
        a = -dfdy
        a.diagonal()[:] += torch.reciprocal(h)

        aqr = QR(a)
        xnew = x + h
        delta = dyns(xnew, y)
        ytemp = y.clone()
        tdelta=aqr.solvev(delta)

        #ax = torch.einsum('ij,j->i', a, tdelta)
        #print(f'a*x={ax}')
        #print(f'Error in LU={torch.norm(ax-delta)}')
        delta=tdelta
        for nn in range(1, nstep):
            ytemp[:]=ytemp[:]+delta[:]
            xnew += h
            yend = dyns(xnew, ytemp)
            if nn == 1 and k <= 1:
                del1 = torch.sqrt(torch.sum(torch.square((delta / scale) )))
                dytemp = dyns(x + h, ytemp)
                delta = dytemp- delta / h
                tdelta = aqr.solvev(delta)

                #ax = torch.einsum('ij,j->i', a, tdelta)
                # print(f'a*x={ax}')
                #print(f'Error in LU={torch.norm(ax - delta)}')
                delta = tdelta

                del2 = torch.sqrt(torch.sum(torch.square(delta / scale)))

                theta = del2 / torch.max(torch.ones_like(del1), del1)
                if theta > 1.0:
                    return False, theta, yend
            tdelta=aqr.solvev(yend)
            #ax = torch.einsum('ij,j->i', a, tdelta)
            # print(f'a*x={ax}')
            #print(f'Error in LU={torch.norm(ax - yend)}')
            delta = tdelta

        yend = ytemp + delta
        return True, theta, yend

    def polyextr(self, k, last, table, coeff):
        l = len(last)
        #js = torch.arange(k.item() - 1, 0, -1)
        #for j in range(k - 1, 0, -1):
        #    self.table[j - 1, :l] = self.table[j, :l] + self.coeff[k, j] * (self.table[j, :l] - self.table[j - 1, :l])
        #for (Int j=k-1; j > 0; j--)
        #    for (Int i=0; i < l; i++)
        #        table[j - 1][i] = table[j][i] + coeff[k][j] * (table[j][i] - table[j - 1][i]);
        #for (Int i=0; i < l; i++)
        #    last[i] = table[0][i] + coeff[k][0] * (table[0][i] - last[i]);

        for j in range(k - 1, 0, -1):
                table[j - 1, :l] = table[j, :l] + coeff[k, j] * (table[j, :l] - table[j - 1, :l])
            #tabledelta = torch.einsum('j,jl->jl', coeff[k, js], (table[js, :l] - table[js - 1, :l]))
            #table[js - 1, :l] = table[js, :l]+ tabledelta
        #for (Int i=0; i < l; i++)
        #    last[i] = table[0][i] + coeff[k][0] * (table[0][i] - last[i]);
        #for i in range(l):
        #    last[i] = table[0, i] + coeff[k, 0] * (table[0, i] - last[i])
        last[:l] = table[0, :l] + coeff[k, 0]*(table[0, :l] - last[:l])
        return last, table


#create main method
