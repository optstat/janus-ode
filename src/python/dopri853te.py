import torch
import numpy as np

class Dopri853Te:
    def __init__(self, yy, dydxx, xx, atoll, rtoll, dens, derivs):
        self.b1 = torch.tensor(5.42937341165687622380535766363e-2)
        self.b6 = torch.tensor(4.45031289275240888144113950566e0)
        self.b7 = torch.tensor(1.89151789931450038304281599044e0)
        self.b8 = torch.tensor(-5.8012039600105847814672114227e0)
        self.b9 = torch.tensor(3.1116436695781989440891606237e-1)
        self.b10 = torch.tensor(-1.52160949662516078556178806805e-1)
        self.b11 = torch.tensor(2.01365400804030348374776537501e-1)
        self.b12 = torch.tensor(4.47106157277725905176885569043e-2)

        self.c2 = torch.tensor(0.526001519587677318785587544488e-01)
        self.c3 = torch.tensor(0.789002279381515978178381316732e-01)
        self.c4 = torch.tensor(0.118350341907227396726757197510e+00)
        self.c5 = torch.tensor(0.281649658092772603273242802490e+00)
        self.c6 = torch.tensor(0.333333333333333333333333333333e+00)
        self.c7 = torch.tensor(0.25e+00)
        self.c8 = torch.tensor(0.307692307692307692307692307692e+00)
        self.c9 = torch.tensor(0.651282051282051282051282051282e+00)
        self.c10 = torch.tensor(0.6e+00)
        self.c11 = torch.tensor(0.857142857142857142857142857142e+00)
        self.c14 = torch.tensor(0.1e+00)
        self.c15 = torch.tensor(0.2e+00)
        self.c16 = torch.tensor(0.777777777777777777777777777778e+00)

        self.bhh1 = torch.tensor(0.244094488188976377952755905512e+00)
        self.bhh2 = torch.tensor(0.733846688281611857341361741547e+00)
        self.bhh3 = torch.tensor(0.220588235294117647058823529412e-01)

        self.er1 = torch.tensor(0.1312004499419488073250102996e-01)
        self.er6 = torch.tensor(-0.1225156446376204440720569753e+01)
        self.er7 = torch.tensor(-0.4957589496572501915214079952e+00)
        self.er8 = torch.tensor(0.1664377182454986536961530415e+01)
        self.er9 = torch.tensor(-0.3503288487499736816886487290e+00)
        self.er10 = torch.tensor(0.3341791187130174790297318841e+00)
        self.er11 = torch.tensor(0.8192320648511571246570742613e-01)
        self.er12 = torch.tensor(-0.2235530786388629525884427845e-01)
        # convert these to pytorch tensors

        self.a21 = torch.tensor(5.26001519587677318785587544488e-2)
        self.a31 = torch.tensor(1.97250569845378994544595329183e-2)
        self.a32 = torch.tensor(5.91751709536136983633785987549e-2)
        self.a41 = torch.tensor(2.95875854768068491816892993775e-2)
        self.a43 = torch.tensor(8.87627564304205475450678981324e-2)
        self.a51 = torch.tensor(2.41365134159266685502369798665e-1)
        self.a53 = torch.tensor(-8.84549479328286085344864962717e-1)
        self.a54 = torch.tensor(9.24834003261792003115737966543e-1)
        self.a61 = torch.tensor(3.7037037037037037037037037037e-2)
        self.a64 = torch.tensor(1.70828608729473871279604482173e-1)
        self.a65 = torch.tensor(1.25467687566822425016691814123e-1)
        self.a71 = torch.tensor(3.7109375e-2)
        self.a74 = torch.tensor(1.70252211019544039314978060272e-1)
        self.a75 = torch.tensor(6.02165389804559606850219397283e-2)
        self.a76 = torch.tensor(-1.7578125e-2)

        self.a81 = torch.tensor(3.70920001185047927108779319836e-2)
        self.a84 = torch.tensor(1.70383925712239993810214054705e-1)
        self.a85 = torch.tensor(1.07262030446373284651809199168e-1)
        self.a86 = torch.tensor(-1.53194377486244017527936158236e-2)
        self.a87 = torch.tensor(8.27378916381402288758473766002e-3)
        self.a91 = torch.tensor(6.24110958716075717114429577812e-1)
        self.a94 = torch.tensor(-3.36089262944694129406857109825e0)
        self.a95 = torch.tensor(-8.68219346841726006818189891453e-1)
        self.a96 = torch.tensor(2.75920996994467083049415600797e1)
        self.a97 = torch.tensor(2.01540675504778934086186788979e1)
        self.a98 = torch.tensor(-4.34898841810699588477366255144e1)
        self.a101 = torch.tensor(4.77662536438264365890433908527e-1)
        self.a104 = torch.tensor(-2.48811461997166764192642586468e0)
        self.a105 = torch.tensor(-5.90290826836842996371446475743e-1)
        self.a106 = torch.tensor(2.12300514481811942347288949897e1)
        self.a107 = torch.tensor(1.52792336328824235832596922938e1)
        self.a108 = torch.tensor(-3.32882109689848629194453265587e1)
        self.a109 = torch.tensor(-2.03312017085086261358222928593e-2)

        self.a111 = torch.tensor(-9.3714243008598732571704021658e-1)
        self.a114 = torch.tensor(5.18637242884406370830023853209e0)
        self.a115 = torch.tensor(1.09143734899672957818500254654e0)
        self.a116 = torch.tensor(-8.14978701074692612513997267357e0)
        self.a117 = torch.tensor(-1.85200656599969598641566180701e1)
        self.a118 = torch.tensor(2.27394870993505042818970056734e1)
        self.a119 = torch.tensor(2.49360555267965238987089396762e0)
        self.a1110 = torch.tensor(-3.0467644718982195003823669022e0)
        self.a121 = torch.tensor(2.27331014751653820792359768449e0)
        self.a124 = torch.tensor(-1.05344954667372501984066689879e1)
        self.a125 = torch.tensor(-2.00087205822486249909675718444e0)
        self.a126 = torch.tensor(-1.79589318631187989172765950534e1)
        self.a127 = torch.tensor(2.79488845294199600508499808837e1)
        self.a128 = torch.tensor(-2.85899827713502369474065508674e0)
        self.a129 = torch.tensor(-8.87285693353062954433549289258e0)
        self.a1210 = torch.tensor(1.23605671757943030647266201528e1)
        self.a1211 = torch.tensor(6.43392746015763530355970484046e-1)


        self.a141 = torch.tensor(5.61675022830479523392909219681e-2)
        self.a147 = torch.tensor(2.53500210216624811088794765333e-1)
        self.a148 = torch.tensor(-2.46239037470802489917441475441e-1)
        self.a149 = torch.tensor(-1.24191423263816360469010140626e-1)
        self.a1410 = torch.tensor(1.5329179827876569731206322685e-1)
        self.a1411 = torch.tensor(8.20105229563468988491666602057e-3)
        self.a1412 = torch.tensor(7.56789766054569976138603589584e-3)
        self.a1413 = torch.tensor(-8.298e-3)

        self.a151 = torch.tensor(3.18346481635021405060768473261e-2)
        self.a156 = torch.tensor(2.83009096723667755288322961402e-2)
        self.a157 = torch.tensor(5.35419883074385676223797384372e-2)
        self.a158 = torch.tensor(-5.49237485713909884646569340306e-2)
        self.a1511 = torch.tensor(-1.08347328697249322858509316994e-4)
        self.a1512 = torch.tensor(3.82571090835658412954920192323e-4)
        self.a1513 = torch.tensor(-3.40465008687404560802977114492e-4)
        self.a1514 = torch.tensor(1.41312443674632500278074618366e-1)
        self.a161 = torch.tensor(-4.28896301583791923408573538692e-1)
        self.a166 = torch.tensor(-4.69762141536116384314449447206e0)
        self.a167 = torch.tensor(7.68342119606259904184240953878e0)
        self.a168 = torch.tensor(4.06898981839711007970213554331e0)
        self.a169 = torch.tensor(3.56727187455281109270669543021e-1)
        self.a1613 = torch.tensor(-1.39902416515901462129418009734e-3)
        self.a1614 = torch.tensor(2.9475147891527723389556272149e0)
        self.a1615 = torch.tensor(-9.15095847217987001081870187138e0)


        self.d41 = torch.tensor(-0.84289382761090128651353491142e+01)
        self.d46 = torch.tensor(0.56671495351937776962531783590e+00)
        self.d47 = torch.tensor(-0.30689499459498916912797304727e+01)
        self.d48 = torch.tensor(0.23846676565120698287728149680e+01)
        self.d49 = torch.tensor(0.21170345824450282767155149946e+01)
        self.d410 = torch.tensor(-0.87139158377797299206789907490e+00)
        self.d411 = torch.tensor(0.22404374302607882758541771650e+01)
        self.d412 = torch.tensor(0.63157877876946881815570249290e+00)
        self.d413 = torch.tensor(-0.88990336451333310820698117400e-01)
        self.d414 = torch.tensor(0.18148505520854727256656404962e+02)
        self.d415 = torch.tensor(-0.91946323924783554000451984436e+01)
        self.d416 = torch.tensor(-0.44360363875948939664310572000e+01)


        self.d51 = torch.tensor(0.10427508642579134603413151009e+02)
        self.d56 = torch.tensor(0.24228349177525818288430175319e+03)
        self.d57 = torch.tensor(0.16520045171727028198505394887e+03)
        self.d58 = torch.tensor(-0.37454675472269020279518312152e+03)
        self.d59 = torch.tensor(-0.22113666853125306036270938578e+02)
        self.d510 = torch.tensor(0.77334326684722638389603898808e+01)
        self.d511 = torch.tensor(-0.30674084731089398182061213626e+02)
        self.d512 = torch.tensor(-0.93321305264302278729567221706e+01)
        self.d513 = torch.tensor(0.15697238121770843886131091075e+02)
        self.d514 = torch.tensor(-0.31139403219565177677282850411e+02)
        self.d515 = torch.tensor(-0.93529243588444783865713862664e+01)
        self.d516 = torch.tensor(0.35816841486394083752465898540e+02)


        self.d61 = torch.tensor(0.19985053242002433820987653617e+02)
        self.d66 = torch.tensor(-0.38703730874935176555105901742e+03)
        self.d67 = torch.tensor(-0.18917813819516756882830838328e+03)
        self.d68 = torch.tensor(0.52780815920542364900561016686e+03)
        self.d69 = torch.tensor(-0.11573902539959630126141871134e+02)
        self.d610 = torch.tensor(0.68812326946963000169666922661e+01)
        self.d611 = torch.tensor(-0.10006050966910838403183860980e+01)
        self.d612 = torch.tensor(0.77771377980534432092869265740e+00)
        self.d613 = torch.tensor(-0.27782057523535084065932004339e+01)
        self.d614 = torch.tensor(-0.60196695231264120758267380846e+02)
        self.d615 = torch.tensor(0.84320405506677161018159903784e+02)
        self.d616 = torch.tensor(0.11992291136182789328035130030e+02)


        self.d71 = torch.tensor(-0.25693933462703749003312586129e+02)
        self.d76 = torch.tensor(-0.15418974869023643374053993627e+03)
        self.d77 = torch.tensor(-0.23152937917604549567536039109e+03)
        self.d78 = torch.tensor(0.35763911791061412378285349910e+03)
        self.d79 = torch.tensor(0.93405324183624310003907691704e+02)
        self.d710 = torch.tensor(-0.37458323136451633156875139351e+02)
        self.d711 = torch.tensor(0.10409964950896230045147246184e+03)
        self.d712 = torch.tensor(0.29840293426660503123344363579e+02)
        self.d713 = torch.tensor(-0.43533456590011143754432175058e+02)
        self.d714 = torch.tensor(0.96324553959188282948394950600e+02)
        self.d715 = torch.tensor(-0.39177261675615439165231486172e+02)
        self.d716 = torch.tensor(-0.14972683625798562581422125276e+03)
        self.beta = torch.tensor(0.0).double()
        self.alpha = torch.tensor(1.0 / 8.0) - self.beta * torch.tensor(0.2)
        self.safe = torch.tensor(0.9).double()
        self.minscale = torch.tensor(1.0/3.0).double()
        self.maxscale = torch.tensor(6.0).double()


        self.derivs = derivs
        self.x    = xx #Initial conditions
        self.y    = yy #State
        self.yout = self.y.clone()
        self.dydx = dydxx #Derivatives
        self.atol = atoll #Absolute tolerance
        self.rtol = rtoll #Relative tolerance
        self.dense = dens #Dense output
        self.D = yy.shape[1] #Dimension of the system-this is number of samples * number of state spaace dimensions
        self.M = yy.shape[0]
        #EPS=numeric_limits<Doub>::epsilon();
        self.eps = torch.finfo(torch.float64).eps #Machine precision
        self.errold = torch.ones_like(self.x)*1.0e-4#Previous error
        self.hnext= torch.zeros_like(xx)
        self.yerr = self.y.clone()
        self.yerr2 = self.y.clone()



    def error(self, h, y, yout, yerr, yerr2):
        yabs = torch.abs(y) #Take the absolute value of the state
        youtabs = torch.abs(yout) #Take the absolute value of the output
        ymaxvals = torch.max(yabs, youtabs) #Take the maximum of the state and the output
        sk   = self.atol + self.rtol * ymaxvals
        yerrosk = torch.square(yerr / sk)
        yerr2osk = torch.square(yerr2 / sk)
        err2 = torch.sum(yerrosk, dim=1)
        err  = torch.sum(yerr2osk, dim=1)
        deno = err + 0.01 * err2
        if (deno <= 0.0).any():
            mask = deno <= 0.0
            deno[mask] = 1.0

        return torch.abs(h) * err * torch.sqrt(torch.tensor(1.0) / (self.D * self.M * deno))

    def dy(self, h, x, y, dydx):

        #ytemp[i] = y[i] + h * a21 * dydx[i];
        he = torch.unsqueeze(h, 1)
        ytemp = y + he* self.a21 * dydx
        #derivs(x+c2*h,ytemp,k2);
        k2 = self.derivs(x + self.c2 * h, ytemp)

        #ytemp[i]=y[i]+h*(a31*dydx[i]+a32*k2[i]);
        ytemp = y + he * (self.a31 * dydx + self.a32 * k2)

        #derivs(x+c3*h,ytemp,k3);
        k3 = self.derivs(x + self.c3 * h, ytemp)

        #ytemp[i] = y[i] + h * (a41 * dydx[i] + a43 * k3[i]);
        ytemp = y + he * (self.a41 * dydx + self.a43 * k3)
        #derivs(x + c4 * h, ytemp, k4);
        k4 = self.derivs(x + self.c4 * h, ytemp)
        #ytemp[i]=y[i]+h*(a51*dydx[i]+a53*k3[i]+a54*k4[i]);
        ytemp = y + he * (self.a51 * dydx + self.a53 * k3 + self.a54 * k4)
        #derivs(x+c5*h,ytemp,k5);
        k5 = self.derivs(x + self.c5 * h, ytemp)
        #ytemp[i]=y[i]+h*(a61*dydx[i]+a64*k4[i]+a65*k5[i]);
        ytemp = y + he * (self.a61 * dydx + self.a64 * k4 + self.a65 * k5)
        #derivs(x+c6*h,ytemp,k6);
        k6 = self.derivs(x + self.c6 * h, ytemp)

        #ytemp[i]=y[i]+h*(a71*dydx[i]+a74*k4[i]+a75*k5[i]+a76*k6[i]);
        ytemp = y + he * (self.a71 * dydx + self.a74 * k4 + self.a75 * k5 + self.a76 * k6)
        #derivs(x+c7*h,ytemp,k7);
        k7 = self.derivs(x + self.c7 * h, ytemp)
        #ytemp[i]=y[i]+h*(a81*dydx[i]+a84*k4[i]+a85*k5[i]+a86*k6[i]+a87*k7[i]);
        ytemp = y + he * (self.a81 * dydx + self.a84 * k4 + self.a85 * k5 + self.a86 * k6 + self.a87 * k7)
        #derivs(x+c8*h,ytemp,k8);
        k8 = self.derivs(x + self.c8 * h, ytemp)
        #ytemp[i]=y[i]+h*(a91*dydx[i]+a94*k4[i]+a95*k5[i]+a96*k6[i]+a97*k7[i]+a98*k8[i]);
        ytemp = y + he * (self.a91 * dydx + self.a94 * k4 + self.a95 * k5 + self.a96 * k6 + self.a97 * k7 + self.a98 * k8)
        #derivs(x+c9*h,ytemp,k9);
        k9 = self.derivs(x + self.c9 * h, ytemp)
        #ytemp[i]=y[i]+h*(a101*dydx[i]+a104*k4[i]+a105*k5[i]+a106*k6[i]+a107*k7[i]+a108*k8[i]+a109*k9[i]);
        ytemp = y + he * (self.a101 * dydx + self.a104 * k4 + self.a105 * k5 + self.a106 * k6 + self.a107 * k7 + \
                         self.a108 * k8 + self.a109 * k9)
        #derivs(x+c10*h,ytemp,k10);
        k10 = self.derivs(x + self.c10 * h, ytemp)

        #ytemp[i]=y[i]+h*(a111*dydx[i]+a114*k4[i]+a115*k5[i]+a116*k6[i]+a117*k7[i]+a118*k8[i]+a119*k9[i]+a1110*k10[i]);
        ytemp = y + he * (self.a111 * dydx + self.a114 * k4 + self.a115 * k5 + self.a116 * k6 + self.a117 * k7 +\
                         self.a118 * k8 + self.a119 * k9 + self.a1110 * k10)
        #derivs(x+c11*h,ytemp,k2);
        k2 = self.derivs(x + self.c11 * h, ytemp)

        xph = x + h
        #ytemp[i]=y[i]+h*(a121*dydx[i]+a124*k4[i]+a125*k5[i]+a126*k6[i]+a127*k7[i]+a128*k8[i]+a129*k9[i]+a1210*k10[i]+a1211*k2[i]);
        ytemp = y + he * (self.a121 * dydx + self.a124 * k4 + self.a125 * k5 + self.a126 * k6 + self.a127 * k7 + \
                         self.a128 * k8 + self.a129 * k9 + self.a1210 * k10 + self.a1211 * k2)
        #derivs(xph, ytemp, k3);
        k3 = self.derivs(xph, ytemp)
        #k4[i]=b1*dydx[i]+b6*k6[i]+b7*k7[i]+b8*k8[i]+b9*k9[i]+b10*k10[i]+b11*k2[i]+b12*k3[i];
        k4 = self.b1 * dydx + self.b6 * k6 + self.b7 * k7 + self.b8 * k8 + self.b9 * k9 + self.b10 * k10 +\
             self.b11 * k2 + self.b12 * k3

        #yout[i] = y[i] + h * k4[i];
        yout = y + he * k4

        #yerr[i] = k4[i] - bhh1 * dydx[i] - bhh2 * k9[i] - bhh3 * k3[i];
        yerr = k4-self.bhh1*dydx-self.bhh2*k9-self.bhh3*k3
        yerr2 = k5-self.bhh1*dydx-self.bhh2*k9-self.bhh3*k3
        return yout, yerr, yerr2


    def step(self, htry):
        h = htry
        reject = torch.ones((self.M,)).bool() & (h > self.eps)
        while (reject).any():
            #make sure h is larger than the machine precision
            mask = reject.clone()  #h could be set to zero if the stepsize is too small by the stepper
            if not mask.all():
                stopHere=True
            if mask.any():
                if not (mask.all() or (~mask).all()):
                    stopHere=True
                self.yout[mask], self.yerr[mask], self.yerr2[mask] = self.dy(h[mask], self.x[mask], \
                                                                     self.y[mask], self.dydx[mask])
                err = self.error(h[mask], self.y[mask], self.yout[mask], self.yerr[mask], self.yerr2[mask])
                #res, h, hnext, errold, reject
                reject[mask], h[mask], self.hnext[mask], self.errold[mask] = \
                    self.control(err, h[mask], self.hnext[mask], self.errold[mask])
                #if (torch.abs(h) <= torch.abs(self.x) * self.eps).any():
                #    raise RuntimeError("stepsize underflow in StepperDopr853")
        dydxnew = self.derivs(self.x + h, self.yout)
        self.dydx = dydxnew
        self.y    = self.yout #Only accept the output if error is acceptable.
        self.xold = self.x
        self.x    = self.x + h
        self.hdid = h


    def control(self, err, h, hnext, errold):
        scale = torch.zeros_like(h)
        mask = (err <= 1.0)
        reject = torch.ones_like(h).bool()
        if mask.any():
            maske = mask & (errold == 0.0)
            if maske.any():
                scale[maske] = self.maxscale
            maske = mask & (errold != 0.0)
            if maske.any():
                scale[maske] = self.safe * torch.pow(err[maske], -self.alpha) * torch.pow(errold[maske], self.beta)
                maskem = maske & (scale < self.minscale)
                if maskem.any():
                    scale[maskem] = self.minscale
                maskem = maske & (scale > self.maxscale)
                if maskem.any():
                    scale[maskem] = self.maxscale
            maskr = mask & reject
            if maskr.any():
                hnext[maskr] = h[maskr] * torch.min(scale[maskr], torch.tensor(1.0))
            masknr = mask & ~reject
            if masknr.any():
                hnext[masknr] = h[masknr] * scale[masknr]

            errold[mask] = torch.max(err[mask], torch.tensor(1.0e-4))
            reject[mask] = False

        mask = (err > 1.0)
        if mask.any():
            scale[mask] = torch.max(self.safe * torch.pow(err[mask], -self.alpha), self.minscale)
            h[mask] *= scale[mask]
            reject[mask] = True

        return reject, h, hnext, errold






def vdp(t, x):
    x1 = x[:,0]
    x2 = x[:,1]
    mus = x[:,2]
    dydx = x*0.0
    dydx[:,0] = x2
    dydx[:,1] = mus*(1 -x1 ** 2) * x2 - x1
    dydx[:,2] = x[:,2]*0.0
    return dydx


if __name__ == '__main__':
    M = 2
    D = 3
    y0 = torch.zeros((M,D)).double()
    y0[:,0] = torch.rand((M)).double()
    y0[:,1] = torch.rand((M)).double()
    y0[:,2] = torch.rand((M)).double()+1.0
    #y0[:,2] = y0[:,2]*0.0+1.0 #mu
    dydx = vdp(0.0, y0)
    t1 = 10.0
    atol = 1e-6
    rtol = 1e-2
    h = torch.ones((M,), dtype=torch.float64) * 1e-3
    x0 = torch.zeros((M,), dtype=torch.float64)
    y1s = y0[:, 0:1]
    y2s = y0[:, 1:2]
    t = x0.clone()
    nsteps = 0
    for i in range(M):
        dopri8 = Dopri853Te(y0, dydx, x0, atol, rtol, False, vdp)
        while (t < t1).any():
            if (t + h > t1).any():
                h = t1 - t
            dopri8.step(h)
            nsteps+=1
            print(f't = {t}, y = {dopri8.y}')
            #Concatenate the x1 and x2 values with the y outputs
            y1s = torch.cat((y1s, dopri8.y[:,0:1]), dim=1)
            y2s = torch.cat((y2s, dopri8.y[:,1:2]), dim=1)
            h = dopri8.hnext
            t = t+h
    print(f'Number of steps = {nsteps}')
    import matplotlib.pyplot as plt

    plt.rcParams['agg.path.chunksize'] = 1000
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i in range(M):
        color = np.random.rand(3)
        ax.scatter(y1s, y2s, color=color)
    plt.show()