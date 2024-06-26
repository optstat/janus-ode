import torch
import numpy as np
from tensordual import TensorDual
from dopri853te import Dopri853Te

#Change device to be cuda if you have a GPU and cpu if not
device = torch.device('cpu')






class Dopri853TeD:
    def __init__(self, yy : TensorDual, dydxx : TensorDual, xx : TensorDual,
                 atoll : torch.tensor, rtoll : torch.tensor, derivs : callable,\
                 theta: TensorDual=None, thetadim = None, ft : TensorDual=None):
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
        self.x      = xx #Initial conditions
        self.y      = yy #State
        self.dydx   = dydxx #Derivatives
        self.atol   = atoll #Absolute tolerance
        self.rtol   = rtoll #Relative tolerance
        self.D = yy.r.shape[1] #Dimension of the system-this is number of samples * number of state spaace dimensions
        self.M = yy.r.shape[0]
        #EPS=numeric_limits<Doub>::epsilon();
        self.eps = torch.finfo(torch.float64).eps #Machine precision
        self.errold = 1.0e-4 * TensorDual.ones_like(xx)
        self.reject = torch.squeeze(TensorDual.bool_like(xx).r) #This defaults to false
        self.hnext= TensorDual.zeros_like(self.x)
        self.yout = TensorDual.zeros_like(self.y)
        self.yerr = TensorDual.zeros_like(self.y)
        self.yerr2 = TensorDual.zeros_like(self.y)
        self.count = 1 #The derivative is already called once
        self.theta = theta
        self.thetadim = thetadim
        self.ft = ft
        self.dense = False
        self.x_dense = None
        self.y_dense = None

    def error(self, h, y, yout, yerr, yerr2):
        yabs = abs(y) #Take the absolute value of the state
        youtabs = abs(yout) #Take the absolute value of the output
        ymaxvals = TensorDual.max_dual(yabs, youtabs) #Take the maximum of the state and the output
        sk   = self.atol + self.rtol * ymaxvals
        yerrosk = (yerr / sk).square()
        yerr2osk = (yerr2 / sk).square()
        #Sum across the dimension of the state space maintaining the batch size
        err2 = TensorDual.sum(yerrosk)
        err  = TensorDual.sum(yerr2osk)
        deno = err + 0.01 * err2
        deno=TensorDual.where(deno <= TensorDual.zeros_like(deno), TensorDual.ones_like(deno), deno)

        res = abs(h) * err * (deno.reciprocal()/(self.D)).sqrt()
        return res #Return the new h



    def dy(self, h,  x, y, dydx):

        #ytemp[i] = y[i] + h * a21 * dydx[i];
        assert isinstance(h, TensorDual)

        #create the parameter segment if necessary
        if self.theta is not None:
            off = self.count * self.thetadim
            thetain = TensorDual(self.theta.r[:, off:off+self.thetadim], self.theta.d[:, off:off+self.thetadim])

        ytemp = y + h * self.a21 * dydx
        #derivs(x+c2*h,ytemp,k2);
        if self.theta is None:
            k2 = self.derivs(x + self.c2 * h, ytemp)
        else:
            junk = x + self.c2 * h
            k2 = self.derivs(x + self.c2 * h, ytemp, thetain)

        #ytemp[i]=y[i]+h*(a31*dydx[i]+a32*k2[i]);
        ytemp = y + h * (self.a31 * dydx + self.a32 * k2)

        #derivs(x+c3*h,ytemp,k3);
        if self.theta is None:
            k3 = self.derivs(x + self.c3 * h, ytemp)
        else:
            k3 = self.derivs(x + self.c3 * h, ytemp, thetain)

        #ytemp[i] = y[i] + h * (a41 * dydx[i] + a43 * k3[i]);
        ytemp = y + h * (self.a41 * dydx + self.a43 * k3)
        #derivs(x + c4 * h, ytemp, k4);
        if self.theta is None:
            k4 = self.derivs(x + self.c4 * h, ytemp)
        else:
            k4 = self.derivs(x + self.c4 * h, ytemp, thetain)
        #ytemp[i]=y[i]+h*(a51*dydx[i]+a53*k3[i]+a54*k4[i]);
        ytemp = y + h * (self.a51 * dydx + self.a53 * k3 + self.a54 * k4)
        #derivs(x+c5*h,ytemp,k5);
        if self.theta is None:
            k5 = self.derivs(x + self.c5 * h, ytemp)
        else:
            k5 = self.derivs(x + self.c5 * h, ytemp, thetain)
        #ytemp[i]=y[i]+h*(a61*dydx[i]+a64*k4[i]+a65*k5[i]);
        ytemp = y + h * (self.a61 * dydx + self.a64 * k4 + self.a65 * k5)
        #derivs(x+c6*h,ytemp,k6);
        if self.theta is None:
            k6 = self.derivs(x + self.c6 * h, ytemp)
        else:
            k6 = self.derivs(x + self.c6 * h, ytemp, thetain)

        #ytemp[i]=y[i]+h*(a71*dydx[i]+a74*k4[i]+a75*k5[i]+a76*k6[i]);
        ytemp = y + h * (self.a71 * dydx + self.a74 * k4 + self.a75 * k5 + self.a76 * k6)
        #derivs(x+c7*h,ytemp,k7);
        if self.theta is None:
            k7 = self.derivs(x + self.c7 * h, ytemp)
        else:
            k7 = self.derivs(x + self.c7 * h, ytemp, thetain)
        #ytemp[i]=y[i]+h*(a81*dydx[i]+a84*k4[i]+a85*k5[i]+a86*k6[i]+a87*k7[i]);
        ytemp = y + h * (self.a81 * dydx + self.a84 * k4 + self.a85 * k5 + self.a86 * k6 + self.a87 * k7)
        #derivs(x+c8*h,ytemp,k8);
        if self.theta is None:
            k8 = self.derivs(x + self.c8 * h, ytemp)
        else:
            k8 = self.derivs(x + self.c8 * h, ytemp, thetain)
        #ytemp[i]=y[i]+h*(a91*dydx[i]+a94*k4[i]+a95*k5[i]+a96*k6[i]+a97*k7[i]+a98*k8[i]);
        ytemp = y + h * (self.a91 * dydx + self.a94 * k4 + self.a95 * k5 + self.a96 * k6 + self.a97 * k7 + self.a98*k8)
        #derivs(x+c9*h,ytemp,k9);
        if self.theta is None:
            k9 = self.derivs(x + self.c9 * h, ytemp)
        else:
            k9 = self.derivs(x + self.c9 * h, ytemp, thetain)
        #ytemp[i]=y[i]+h*(a101*dydx[i]+a104*k4[i]+a105*k5[i]+a106*k6[i]+a107*k7[i]+a108*k8[i]+a109*k9[i]);
        ytemp = y + h * (self.a101 * dydx + self.a104*k4 + self.a105 * k5 + self.a106 * k6 + self.a107 * k7 +\
                         self.a108 * k8 + self.a109 * k9)
        #derivs(x+c10*h,ytemp,k10);
        if self.theta is None:
            k10 = self.derivs(x + self.c10 * h, ytemp)
        else:
            k10 = self.derivs(x + self.c10 * h, ytemp, thetain)

        #ytemp[i]=y[i]+h*(a111*dydx[i]+a114*k4[i]+a115*k5[i]+a116*k6[i]+a117*k7[i]+a118*k8[i]+a119*k9[i]+a1110*k10[i]);
        ytemp = y + h * (self.a111 * dydx + self.a114 * k4 + self.a115 * k5 + self.a116 * k6 + self.a117 * k7 + \
                         self.a118 * k8 + self.a119 * k9 + self.a1110 * k10)
        #derivs(x+c11*h,ytemp,k2);
        if self.theta is None:
            k2 = self.derivs(x + self.c11 * h, ytemp)
        else:
            k2 = self.derivs(x + self.c11 * h, ytemp, thetain)

        xph = x + h
        #ytemp[i]=y[i]+h*(a121*dydx[i]+a124*k4[i]+a125*k5[i]+a126*k6[i]+a127*k7[i]+a128*k8[i]+a129*k9[i]+a1210*k10[i]+a1211*k2[i]);
        ytemp = y + h * (self.a121 * dydx + self.a124 * k4 + self.a125 * k5 + self.a126 * k6 + self.a127 * k7 + \
                         self.a128 * k8 + self.a129 * k9 + self.a1210 * k10 + self.a1211 * k2)
        #derivs(xph, ytemp, k3);
        if self.theta is None:
            k3 = self.derivs(xph, ytemp)
        else:
            k3 = self.derivs(xph, ytemp, thetain)
        #k4[i]=b1*dydx[i]+b6*k6[i]+b7*k7[i]+b8*k8[i]+b9*k9[i]+b10*k10[i]+b11*k2[i]+b12*k3[i];
        k4 = self.b1 * dydx + self.b6 * k6 + self.b7 * k7 + self.b8 * k8 + self.b9 * k9 + self.b10 * k10 +\
             self.b11 * k2 + self.b12 * k3

        #yout[i] = y[i] + h * k4[i];
        yout = y + h * k4

        #yerr[i] = k4[i] - bhh1 * dydx[i] - bhh2 * k9[i] - bhh3 * k3[i];
        yerr = k4-self.bhh1*dydx-self.bhh2*k9-self.bhh3*k3
        yerr2 = k5-self.bhh1*dydx-self.bhh2*k9-self.bhh3*k3
        return yout, yerr, yerr2


    def control(self, err, h, hnext, errold):
        scale = TensorDual.zeros_like(h)
        mask = (err <= 1.0)
        reject = torch.squeeze(torch.ones_like(h.r).bool(), dim=1)
        if mask.any():
            maske = mask & (errold == 0.0)
            if maske.any():
                scale.r[maske] = self.maxscale
                scale.d[maske] *= 0.0
            maske = mask & (errold != 0.0)
            if maske.any():
                erroldm = TensorDual(errold.r[maske], errold.d[maske])
                errm = TensorDual(err.r[maske], err.d[maske])
                scalem = self.safe * pow(errm, -self.alpha) * pow(erroldm, self.beta)
                scale.r[maske] = scalem.r
                scale.d[maske] = scalem.d
                maskem = maske & (scale < self.minscale)
                if maskem.any():
                    #Set it to a constant
                    scale.r[maskem] = self.minscale
                    scale.d[maskem] *= 0.0
                maskem = maske & (scale > self.maxscale)
                if maskem.any():
                    scale.r[maskem] = self.maxscale
                    scale.d[maskem] *= 0.0
            maskr = mask & reject
            if maskr.any():
                hm = TensorDual(h.r[maskr], h.d[maskr])
                scalem = TensorDual(scale.r[maskr], scale.d[maskr])
                one = TensorDual.ones_like(hm)
                hnextm = hm * TensorDual.where(scalem < one, scalem, one)
                hnext.r[maskr] = hnextm.r
                hnext.d[maskr] = hnextm.d
            masknr = mask & ~reject
            if masknr.any():
                hnextm = TensorDual(hnext.r[masknr], hnext.d[masknr])
                hm = TensorDual(h.r[masknr], h.d[masknr])
                scalem = TensorDual(scale.r[masknr], scale.d[masknr])
                hnextm = hm * scalem
                hnext.r[masknr] = hnextm.r
                hnext.d[masknr] = hnextm.d
            errm = TensorDual(err.r[mask], err.d[mask])
            ct = TensorDual.ones_like(errm)
            erroldm = TensorDual.where(errm > ct, errm, ct)
            errold.r[mask] = erroldm.r
            errold.d[mask] = erroldm.d
            reject[mask] = False

        mask = (err > 1.0)
        if mask.any():
            errm = TensorDual(err.r[mask], err.d[mask])
            minsct = TensorDual.ones_like(errm) * self.minscale
            safepoe = self.safe * pow(errm, -self.alpha)
            scalem = TensorDual.where(safepoe > minsct, safepoe, minsct)
            scale.r[mask] = scalem.r
            scale.d[mask] = scalem.d
            hm = TensorDual(h.r[mask], h.d[mask])
            hm = hm*scalem
            h.r[mask] = hm.r
            h.d[mask] = hm.d
            reject[mask] = True

        return reject, h, hnext, errold




    def step(self, htry):
        if self.theta is not None:
            if self.count > self.theta.r.shape[1]/self.thetadim:
                raise ValueError("The number of steps exceeds the number of parameters.\
                                  You can increase the number of parameters or reduce accuracy")
        h = htry.clone()
        reject = torch.ones((self.M,)).bool() & (h > self.eps)
        while (reject).any():
            #make sure h is larger than the machine precision
            mask = reject.clone()

            if mask.any():
                hm = TensorDual(h.r[mask], h.d[mask])
                xm = TensorDual(self.x.r[mask], self.x.d[mask])
                ym = TensorDual(self.y.r[mask], self.y.d[mask])
                dydxm = TensorDual(self.dydx.r[mask], self.dydx.d[mask])
                youtm, yerrm, yerr2m = self.dy(hm, xm, ym, dydxm)
                self.yout.r[mask] = youtm.r
                self.yout.d[mask] = youtm.d
                self.yerr.r[mask] = yerrm.r
                self.yerr.d[mask] = yerrm.d
                self.yerr2.r[mask] = yerr2m.r
                self.yerr2.d[mask] = yerr2m.d

                err = self.error(hm, ym, youtm, yerrm, yerr2m)
                #res, h, hnext, errold, reject
                hnextm = TensorDual(self.hnext.r[mask], self.hnext.d[mask])
                erroldm = TensorDual(self.errold.r[mask], self.errold.d[mask])
                reject[mask], hm, hnextm, erroldm = \
                    self.control(err, hm, hnextm, erroldm)
                if (hm.r < self.eps).any():
                    stopHere = True
                h.r[mask] = hm.r
                h.d[mask] = hm.d
                self.hnext.r[mask] = hnextm.r
                self.hnext.d[mask] = hnextm.d
                self.errold.r[mask] = erroldm.r
        if self.theta is None:
            dydxnew = self.derivs(self.x + h, self.yout)
        else:
            if (self.count + 1) * self.thetadim > self.theta.r.shape[1]:
                raise Exception("Not enough parameters supplied for theta")
            thetain = TensorDual(self.theta.r[:, self.count*self.thetadim:(self.count+1)*self.thetadim],\
                                    self.theta.d[:, self.count*self.thetadim:(self.count+1)*self.thetadim])
            dydxnew = self.derivs(self.x + h, self.yout,  thetain)
        self.dydx = dydxnew
        self.y    = self.yout #Only accept the output if error is acceptable.
        self.xold = self.x
        self.x    = self.x + h
        self.hdid = h
        self.count+=1




def vdpd(t, x):
    x1 = TensorDual(x.r[:, 0], x.d[:, 0])
    x2 = TensorDual(x.r[:, 1], x.d[:, 1])
    mus = TensorDual(x.r[:, 2], x.d[:, 2])
    dydx = TensorDual.zeros_like(x)
    dydx1 = x2
    dydx2 = mus*(1 -x1 *x1) * x2 - x1
    dydx.r[:, 0] = dydx1.r
    dydx.d[:, 0] = dydx1.d
    dydx.r[:, 1] = dydx2.r
    dydx.d[:, 1] = dydx2.d
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


if __name__ == '__main__':
    M = 1
    ft = 6.0
    #The trick is to create a dual tensor that has the identity matrix as the derivative
    y0r = torch.zeros((M,3), dtype=torch.float64).to(device)
    y0r[:,0] = 2.0
    y0r[:,1] = 0.0
    y0r[:,2] = 1.0
    identity_list = [torch.eye(3, dtype=torch.float64) for _ in range(M)]
    y0d = torch.stack(identity_list).to(device)
    y0 = TensorDual(y0r, y0d)
    dydx = vdpd(0.0, y0) #Need to pass in the initial dynamics
    t = TensorDual(torch.zeros((M, 1), dtype=torch.float64), torch.zeros((M, 1, 3), dtype=torch.float64)) #Time has no dependency the initial conditions
    ts = t.r.clone()
    t1 = TensorDual(torch.ones(M, 1).double() * ft, torch.zeros(M, 1, 3).double())
    atol = 1e-6
    rtol = 1e-3
    h = TensorDual.ones_like(t) * 1.0e-2
    ys = torch.unsqueeze(y0r, dim=2)
    dydx0 = torch.unsqueeze(y0d, dim=3)
    dopri8 = Dopri853TeD(y0, dydx, t, atol, rtol, vdpd)


    while torch.any(torch.lt(t.r, t1.r)):

        h = TensorDual.where(t+h > t1, t1-t, h)
        print(f'h= {h}')
        print(f't= {t}')
        dopri8.step(h)
        t = t+dopri8.hdid


        ts = torch.cat((ts, t.r), dim=1)
        h = dopri8.hnext
        ys = torch.cat((ys, torch.unsqueeze(dopri8.yout.r, 2)), dim=2)
        dydx0 = torch.cat((dydx0, torch.unsqueeze(dopri8.yout.d, 3)), dim=3)
    print(f'yf = {dopri8.yout}')
    import matplotlib.pyplot as plt

    plt.rcParams['agg.path.chunksize'] = 1000
    markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']

    fig = plt.figure()
    #Add the first plot

    ax = fig.add_subplot(111)
    #make the plot bigger
    fig.set_size_inches(18.5, 10.5)
    #add a second plot
    #add space between plots
    for idx in range(M):
        ax.scatter(ts[idx,:].cpu().numpy(), ys[idx,0, :].cpu().numpy(), label=''+str(idx), marker=np.random.choice(markers))
    #put the legend outside the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #label the x axis
    ax.set_xlabel('t')
    #label the y axis
    ax.set_ylabel('x1')
    plt.show()
    plt.close(fig)

    fig = plt.figure()
    #Add the first plot

    ax2 = fig.add_subplot(111)
    #make the plot bigger
    fig.set_size_inches(18.5, 10.5)
    #add a second plot
    #add space between plots
    for idx in range(M):
        ax2.scatter(ts[idx,:].cpu().numpy(), ys[idx,1, :].cpu().numpy(), label=''+str(idx), marker=np.random.choice(markers))
    #put the legend outside the plot
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #label the x axis
    ax2.set_xlabel('t')
    #label the y axis
    ax2.set_ylabel('x2')
    plt.show()
    plt.close(fig)


    fig = plt.figure()
    #Add the first plot

    ax3 = fig.add_subplot(111)
    #make the plot bigger
    fig.set_size_inches(18.5, 10.5)
    #add a second plot
    #add space between plots
    for idx in range(M):
        ax3.scatter(ys[idx,0, :].cpu().numpy(), ys[idx,1, :].cpu().numpy(), label=''+str(idx), marker=np.random.choice(markers))
    #put the legend outside the plot
    ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #label the x axis
    ax3.set_xlabel('x1')
    #label the y axis
    ax3.set_ylabel('x2')
    plt.show()
    plt.close(fig)



    fig = plt.figure()
    # Add the first plot

    ax4 = fig.add_subplot(111)
    #make the plot bigger
    fig.set_size_inches(18.5, 10.5)
    #add a second plot
    #add space between plots
    for idx in range(M):
        ax4.scatter(ts[idx,:].cpu().numpy(), dydx0[idx,0, 0,:].cpu().numpy(), label=''+str(idx), marker=np.random.choice(markers))
    #put the legend outside the plot
    #label the x axis
    ax4.set_xlabel('t')
    #label the y axis
    ax4.set_ylabel('dx1dx10')
    plt.show()
    #close the figure
    plt.close(fig)


    fig = plt.figure()
    # Add the first plot

    ax = fig.add_subplot(111)
    #make the plot bigger
    fig.set_size_inches(18.5, 10.5)
    #add a second plot
    #add space between plots
    for idx in range(M):
        ax.scatter(ts[idx,:].cpu().numpy(), dydx0[idx,0, 1,:].cpu().numpy(), label=''+str(idx), marker=np.random.choice(markers))
    #put the legend outside the plot
    #label the x axis
    ax.set_xlabel('t')
    #label the y axis
    ax.set_ylabel('dx1dx20')
    plt.show()
    #close the figure
    plt.close(fig)


    fig = plt.figure()
    # Add the first plot
    ax = fig.add_subplot(111)
    #make the plot bigger
    fig.set_size_inches(18.5, 10.5)
    #add a second plot
    #add space between plots
    for idx in range(M):
        ax.scatter(ts[idx,:].cpu().numpy(), dydx0[idx,1, 0,:].cpu().numpy(), label=''+str(idx), marker=np.random.choice(markers))
    #put the legend outside the plot
    #label the x axis
    ax.set_xlabel('t')
    #label the y axis
    ax.set_ylabel('dx2dx10')
    plt.show()
    #close the figure
    plt.close(fig)


    fig = plt.figure()
    # Add the first plot
    ax = fig.add_subplot(111)
    #make the plot bigger
    fig.set_size_inches(18.5, 10.5)
    #add a second plot
    #add space between plots
    for idx in range(M):
        ax.scatter(ts[idx,:].cpu().numpy(), dydx0[idx,1, 1,:].cpu().numpy(), label=''+str(idx), marker=np.random.choice(markers))
    #put the legend outside the plot
    #label the x axis
    ax.set_xlabel('t')
    #label the y axis
    ax.set_ylabel('dx2dx20')
    plt.show()
    #close the figure
    plt.close(fig)
