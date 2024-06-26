#ifndef DOPRI853TED_HPP_INCLUDED
#define DOPRI853TED_HPP_INCLUDED
#include <type_traits>
#include <iostream>
#include <torch/torch.h>
#include <cassert>

//set the print precision

torch::Tensor filter(torch::Tensor x, torch::Tensor y) {
    // Create a tensor of zeros with the same size as x
    auto true_indices = torch::nonzero(x);

    auto expanded = torch::zeros_like(x);

    expanded.index_put_({true_indices}, y);

    auto filtered_x = x.to(torch::kBool) & expanded.to(torch::kBool);

    return filtered_x;
}


template <typename Dynamics>
//Create an interface that include method for dynamics and jacobian
class Dopri853TeD
{
public :
    const torch::Tensor b1 = torch::tensor(5.42937341165687622380535766363e-2);
    const torch::Tensor b6 = torch::tensor(4.45031289275240888144113950566e0);
    const torch::Tensor b7 = torch::tensor(1.89151789931450038304281599044e0);
    const torch::Tensor b8 = torch::tensor(-5.8012039600105847814672114227e0);
    const torch::Tensor b9 = torch::tensor(3.1116436695781989440891606237e-1);
    const torch::Tensor b10 = torch::tensor(-1.52160949662516078556178806805e-1);
    const torch::Tensor b11 = torch::tensor(2.01365400804030348374776537501e-1);
    const torch::Tensor b12 = torch::tensor(4.47106157277725905176885569043e-2);
    const torch::Tensor c2 = torch::tensor(0.526001519587677318785587544488e-01);
    const torch::Tensor c3 = torch::tensor(0.789002279381515978178381316732e-01);
    const torch::Tensor c4 = torch::tensor(0.118350341907227396726757197510e+00);
    const torch::Tensor c5 = torch::tensor(0.281649658092772603273242802490e+00);
    const torch::Tensor c6 = torch::tensor(0.333333333333333333333333333333e+00);
    const torch::Tensor c7 = torch::tensor(0.25e+00);
    const torch::Tensor c8 = torch::tensor(0.307692307692307692307692307692e+00);
    const torch::Tensor c9 = torch::tensor(0.651282051282051282051282051282e+00);
    const torch::Tensor c10 = torch::tensor(0.6e+00);
    const torch::Tensor c11 = torch::tensor(0.857142857142857142857142857142e+00);
    const torch::Tensor c14 = torch::tensor(0.1e+00);
    const torch::Tensor c15 = torch::tensor(0.2e+00);
    const torch::Tensor c16 = torch::tensor(0.777777777777777777777777777778e+00);
    const torch::Tensor bhh1 = torch::tensor(0.244094488188976377952755905512e+00);
    const torch::Tensor bhh2 = torch::tensor(0.733846688281611857341361741547e+00);
    const torch::Tensor bhh3 = torch::tensor(0.220588235294117647058823529412e-01);
    const torch::Tensor d1 = torch::tensor(0.1312004499419488073250102996e-01);
    const torch::Tensor d6 = torch::tensor(-0.1225156446376204440720569753e+01);
    const torch::Tensor d7 = torch::tensor(-0.4957589496572501915214079952e+00);
    const torch::Tensor d8 = torch::tensor(0.1664377182454986536961530415e+01);
    const torch::Tensor d9 = torch::tensor(-0.3503288487499736816886487290e+00);
    const torch::Tensor d10 = torch::tensor(0.3341791187130174790297318841e+00);
    const torch::Tensor d11 = torch::tensor(0.8192320648511571246570742613e-01);
    const torch::Tensor d12 = torch::tensor(-0.2235530786388629525884427845e-01);

    const torch::Tensor a21 = torch::tensor(5.26001519587677318785587544488e-2);
    const torch::Tensor a31 = torch::tensor(1.97250569845378994544595329183e-2);
    const torch::Tensor a32 = torch::tensor(5.91751709536136983633785987549e-2);
    const torch::Tensor a41 = torch::tensor(2.95875854768068491816892993775e-2);
    const torch::Tensor a43 = torch::tensor(8.87627564304205475450678981324e-2);
    const torch::Tensor a51 = torch::tensor(2.41365134159266685502369798665e-1);
    const torch::Tensor a53 = torch::tensor(-8.84549479328286085344864962717e-1);
    const torch::Tensor a54 = torch::tensor(9.24834003261792003115737966543e-1);
    const torch::Tensor a61 = torch::tensor(3.7037037037037037037037037037e-2);
    const torch::Tensor a64 = torch::tensor(1.70828608729473871279604482173e-1);
    const torch::Tensor a65 = torch::tensor(1.25467687566822425016691814123e-1);
    const torch::Tensor a71 = torch::tensor(3.7109375e-2);
    const torch::Tensor a74 = torch::tensor(1.70252211019544039314978060272e-1);
    const torch::Tensor a75 = torch::tensor(6.02165389804559606850219397283e-2);
    const torch::Tensor a76 = torch::tensor(-1.7578125e-2);

    const torch::Tensor a81 = torch::tensor(3.70920001185047927108779319836e-2);
    const torch::Tensor a84 = torch::tensor(1.70383925712239993810214054705e-1);
    const torch::Tensor a85 = torch::tensor(1.07262030446373284651809199168e-1);
    const torch::Tensor a86 = torch::tensor(-1.53194377486244017527936158236e-2);
    const torch::Tensor a87 = torch::tensor(8.27378916381402288758473766002e-3);
    const torch::Tensor a91 = torch::tensor(6.24110958716075717114429577812e-1);
    const torch::Tensor a94 = torch::tensor(-3.36089262944694129406857109825e0);
    const torch::Tensor a95 = torch::tensor(-8.68219346841726006818189891453e-1);
    const torch::Tensor a96 = torch::tensor(2.75920996994467083049415600797e1);
    const torch::Tensor a97 = torch::tensor(2.01540675504778934086186788979e1);
    const torch::Tensor a98 = torch::tensor(-4.34898841810699588477366255144e1);
    const torch::Tensor a101 = torch::tensor(4.77662536438264365890433908527e-1);
    const torch::Tensor a104 = torch::tensor(-2.48811461997166764192642586468e0);
    const torch::Tensor a105 = torch::tensor(-5.90290826836842996371446475743e-1);
    const torch::Tensor a106 = torch::tensor(2.12300514481811942347288949897e1);
    const torch::Tensor a107 = torch::tensor(1.52792336328824235832596922938e1);
    const torch::Tensor a108 = torch::tensor(-3.32882109689848629194453265587e1);
    const torch::Tensor a109 = torch::tensor(-2.03312017085086261358222928593e-2);

    const torch::Tensor a111 = torch::tensor(-9.3714243008598732571704021658e-1);
    const torch::Tensor a114 = torch::tensor(5.18637242884406370830023853209e0);
    const torch::Tensor a115 = torch::tensor(1.09143734899672957818500254654e0);
    const torch::Tensor a116 = torch::tensor(-8.14978701074692612513997267357e0);
    const torch::Tensor a117 = torch::tensor(-1.85200656599969598641566180701e1);
    const torch::Tensor a118 = torch::tensor(2.27394870993505042818970056734e1);
    const torch::Tensor a119 = torch::tensor(2.49360555267965238987089396762e0);
    const torch::Tensor a1110 = torch::tensor(-3.0467644718982195003823669022e0);
    const torch::Tensor a121 = torch::tensor(2.27331014751653820792359768449e0);
    const torch::Tensor a124 = torch::tensor(-1.05344954667372501984066689879e1);
    const torch::Tensor a125 = torch::tensor(-2.00087205822486249909675718444e0);
    const torch::Tensor a126 = torch::tensor(-1.79589318631187989172765950534e1);
    const torch::Tensor a127 = torch::tensor(2.79488845294199600508499808837e1);
    const torch::Tensor a128 = torch::tensor(-2.85899827713502369474065508674e0);
    const torch::Tensor a129 = torch::tensor(-8.87285693353062954433549289258e0);
    const torch::Tensor a1210 = torch::tensor(1.23605671757943030647266201528e1);
    const torch::Tensor a1211 = torch::tensor(6.43392746015763530355970484046e-1);

    const torch::Tensor a141 = torch::tensor(5.61675022830479523392909219681e-2);
    const torch::Tensor a147 = torch::tensor(2.53500210216624811088794765333e-1);
    const torch::Tensor a148 = torch::tensor(-2.46239037470802489917441475441e-1);
    const torch::Tensor a149 = torch::tensor(-1.24191423263816360469010140626e-1);
    const torch::Tensor a1410 = torch::tensor(1.5329179827876569731206322685e-1);
    const torch::Tensor a1411 = torch::tensor(8.20105229563468988491666602057e-3);
    const torch::Tensor a1412 = torch::tensor(7.56789766054569976138603589584e-3);
    const torch::Tensor a1413 = torch::tensor(-8.298e-3);

    const torch::Tensor a151 = torch::tensor(3.18346481635021405060768473261e-2);
    const torch::Tensor a156 = torch::tensor(2.83009096723667755288322961402e-2);
    const torch::Tensor a157 = torch::tensor(5.35419883074385676223797384372e-2);
    const torch::Tensor a158 = torch::tensor(-5.49237485713909884646569340306e-2);
    const torch::Tensor a1511 = torch::tensor(-1.08347328697249322858509316994e-4);
    const torch::Tensor a1512 = torch::tensor(3.82571090835658412954920192323e-4);
    const torch::Tensor a1513 = torch::tensor(-3.40465008687404560802977114492e-4);
    const torch::Tensor a1514 = torch::tensor(1.41312443674632500278074618366e-1);
    const torch::Tensor a161 = torch::tensor(-4.28896301583791923408573538692e-1);
    const torch::Tensor a166 = torch::tensor(-4.69762141536116384314449447206e0);
    const torch::Tensor a167 = torch::tensor(7.68342119606259904184240953878e0);
    const torch::Tensor a168 = torch::tensor(4.06898981839711007970213554331e0);
    const torch::Tensor a169 = torch::tensor(3.56727187455281109270669543021e-1);
    const torch::Tensor a1613 = torch::tensor(-1.39902416515901462129418009734e-3);
    const torch::Tensor a1614 = torch::tensor(2.9475147891527723389556272149e0);
    const torch::Tensor a1615 = torch::tensor(-9.15095847217987001081870187138e0);

    const torch::Tensor d41 = torch::tensor(-0.84289382761090128651353491142e+01);
    const torch::Tensor d46 = torch::tensor(0.56671495351937776962531783590e+00);
    const torch::Tensor d47 = torch::tensor(-0.30689499459498916912797304727e+01);
    const torch::Tensor d48 = torch::tensor(0.23846676565120698287728149680e+01);
    const torch::Tensor d49 = torch::tensor(0.21170345824450282767155149946e+01);
    const torch::Tensor d410 = torch::tensor(-0.87139158377797299206789907490e+00);
    const torch::Tensor d411 = torch::tensor(0.22404374302607882758541771650e+01);
    const torch::Tensor d412 = torch::tensor(0.63157877876946881815570249290e+00);
    const torch::Tensor d413 = torch::tensor(-0.88990336451333310820698117400e-01);
    const torch::Tensor d414 = torch::tensor(0.18148505520854727256656404962e+02);
    const torch::Tensor d415 = torch::tensor(-0.91946323924783554000451984436e+01);
    const torch::Tensor d416 = torch::tensor(-0.44360363875948939664310572000e+01);

    const torch::Tensor d51 = torch::tensor(0.10427508642579134603413151009e+02);
    const torch::Tensor d56 = torch::tensor(0.24228349177525818288430175319e+03);
    const torch::Tensor d57 = torch::tensor(0.16520045171727028198505394887e+03);
    const torch::Tensor d58 = torch::tensor(-0.37454675472269020279518312152e+03);
    const torch::Tensor d59 = torch::tensor(-0.22113666853125306036270938578e+02);
    const torch::Tensor d510 = torch::tensor(0.77334326684722638389603898808e+01);
    const torch::Tensor d511 = torch::tensor(-0.30674084731089398182061213626e+02);
    const torch::Tensor d512 = torch::tensor(-0.93321305264302278729567221706e+01);
    const torch::Tensor d513 = torch::tensor(0.15697238121770843886131091075e+02);
    const torch::Tensor d514 = torch::tensor(-0.31139403219565177677282850411e+02);
    const torch::Tensor d515 = torch::tensor(-0.93529243588444783865713862664e+01);
    const torch::Tensor d516 = torch::tensor(0.35816841486394083752465898540e+02);

    const torch::Tensor d61 = torch::tensor(0.19985053242002433820987653617e+02);
    const torch::Tensor d66 = torch::tensor(-0.38703730874935176555105901742e+03);
    const torch::Tensor d67 = torch::tensor(-0.18917813819516756882830838328e+03);
    const torch::Tensor d68 = torch::tensor(0.52780815920542364900561016686e+03);
    const torch::Tensor d69 = torch::tensor(-0.11573902539959630126141871134e+02);
    const torch::Tensor d610 = torch::tensor(0.68812326946963000169666922661e+01);
    const torch::Tensor d611 = torch::tensor(-0.10006050966910838403183860980e+01);
    const torch::Tensor d612 = torch::tensor(0.77771377980534432092869265740e+00);
    const torch::Tensor d613 = torch::tensor(-0.27782057523535084065932004339e+01);
    const torch::Tensor d614 = torch::tensor(-0.60196695231264120758267380846e+02);
    const torch::Tensor d615 = torch::tensor(0.84320405506677161018159903784e+02);
    const torch::Tensor d616 = torch::tensor(0.11992291136182789328035130030e+02);

    const torch::Tensor d71 = torch::tensor(-0.25693933462703749003312586129e+02);
    const torch::Tensor d76 = torch::tensor(-0.15418974869023643374053993627e+03);
    const torch::Tensor d77 = torch::tensor(-0.23152937917604549567536039109e+03);
    const torch::Tensor d78 = torch::tensor(0.35763911791061412378285349910e+03);
    const torch::Tensor d79 = torch::tensor(0.93405324183624310003907691704e+02);
    const torch::Tensor d710 = torch::tensor(-0.37458323136451633156875139351e+02);
    const torch::Tensor d711 = torch::tensor(0.10409964950896230045147246184e+03);
    const torch::Tensor d712 = torch::tensor(0.29840293426660503123344363579e+02);
    const torch::Tensor d713 = torch::tensor(-0.43533456590011143754432175058e+02);
    const torch::Tensor d714 = torch::tensor(0.96324553959188282948394950600e+02);
    const torch::Tensor d715 = torch::tensor(-0.39177261675615439165231486172e+02);
    const torch::Tensor d716 = torch::tensor(-0.14972683625798562581422125276e+03);
    const torch::Tensor beta = torch::tensor(0.0).to(torch::kFloat64);
    const torch::Tensor alpha = torch::tensor(1.0 / 8.0) - beta * torch::tensor(0.2);
    const torch::Tensor safe = torch::tensor(0.9).to(torch::kFloat64);
    const torch::Tensor minscale = torch::tensor(1.0/3.0).to(torch::kFloat64);
    const torch::Tensor maxscale = torch::tensor(6.0).to(torch::kFloat64);

    const torch::Tensor EPS = torch::tensor(std::numeric_limits<double>::epsilon()).to(torch::kFloat64);
    Dynamics dynamics;


    int D;
    int M;
    int N;
    TensorDual y, yerr, yerr2, yout;
    TensorDual x, xold;
    TensorDual dydx;
    TensorDual dfdx;
    TensorMatDual dfdy;
    TensorDual theta;
    TensorDual h, hnext, hdid, hold;
    bool dense;
    int thetadims;
    torch::Tensor reject;
    torch::Tensor first_step;
    TensorDual errold;
    torch::Tensor atol, rtol;
    int count;
    std::vector<TensorDual> y_dense, x_dense;



public:
    Dopri853TeD(TensorDual& x, TensorDual& y, TensorDual& dydx, TensorDual& h, Dynamics dyns, double atol, double rtol,
             bool dense=false, TensorDual& theta=defaultTensorDual, int thetadims=0) {
           //self.M = y.d.shape[0]
           M = y.r.size(0);
           //self.D = y.d.shape[1]
           D = y.r.size(1);
           //self.N = y.d.shape[2]
           N = y.d.size(2);
           //        self.y = y.clone()
           this->y = y;
           //self.dydx = dydx.clone()
           this->dydx = dydx;
           this->theta = theta;
           dynamics = dyns;
           this->x = x;
           this->h = h;
           this->atol = torch::tensor(atol, y.r.device());
           this->rtol = torch::tensor(rtol, y.r.device());
           this->dense = dense;
           this->thetadims = thetadims;
           torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat64).device(y.r.device());
           //        self.dfdx = TensorDual.createZero(torch.zeros((self.M, 1), dtype=torch.float64, device = y.device), self.N)
           this->dfdx = TensorDual::createZero(torch::zeros({this->M, 1}, options), this->N);
           this->dfdy = TensorMatDual::createZero(torch::zeros({this->M, this->D, this->D}, options), this->N);
           this->yout = TensorDual::createZero(torch::zeros({this->M, this->D}, options), this->N);
           this->yerr = TensorDual::createZero(torch::zeros({this->M, this->D}, options), this->N);
           this->yerr2 = TensorDual::createZero(torch::zeros({this->M, this->D}, options), this->N);
           //self.hnext = TensorDual.createZero(torch.zeros((self.M, 1), dtype=torch.float64, device = y.device), self.N)
           this->hnext = TensorDual::createZero(torch::zeros({this->M, 1}, options), this->N);
           //self.hdid = TensorDual.createZero(torch.zeros((self.M, 1), dtype=torch.float64, device = y.device), self.N)
           this->hdid = TensorDual::createZero(torch::zeros({this->M, 1}, options), this->N);
           //self.reject = torch.zeros(self.M, dtype=torch.float64, device = y.device).bool()
           this->reject = torch::zeros({this->M}, options).to(torch::kBool);
           //self.first_step = torch.ones(self.M, dtype=torch.float64, device = y.device).bool()
           this->first_step = torch::ones({this->M}, options).to(torch::kBool);
           //self.hold = TensorDual.createZero(torch.zeros((self.M, 1), dtype=torch.float64, device = y.device), self.N)
           this->hold = TensorDual::createZero(torch::zeros({this->M, 1}, options), this->N);
           //self.hnext = TensorDual.createZero(torch.zeros((self.M, 1), dtype=torch.float64, device = y.device), self.N)
           this->hnext = TensorDual::createZero(torch::zeros({this->M, 1}, options), this->N);
           //self.errold = TensorDual.createZero(torch.zeros((self.M, 1), dtype=torch.float64, device = y.device), self.N)
           this->errold = TensorDual::createZero(torch::zeros({this->M, 1}, options), this->N);
           this->count = 1;
           //self.y_dense = [self.y]
           this->y_dense = {this->y};
    }

    /**
     *Check for error and return the error only for those elements that have been rejected
     */
    TensorDual error() {
        TensorDual ym = y.index(reject);
        TensorDual youtm = yout.index(reject);
        TensorDual yerrm = yerr.index(reject);
        TensorDual yerr2m = yerrm.square();
        TensorDual hm = h.index(reject);


        TensorDual yabs = ym.abs();
        TensorDual youtabs = youtm.abs();
        TensorDual ymaxvals = max(yabs, youtabs);
        std::cerr << "ymaxvals = " << ymaxvals << std::endl;
        TensorDual sk = atol + rtol * ymaxvals;
        std::cerr << "sk = " << sk << std::endl;
        TensorDual yerr2osk = (yerrm / sk).square();
        TensorDual yerrosk = (yerr2m / sk).square();
        TensorDual err2 = TensorDual::sum(yerrosk);
        TensorDual err = TensorDual::sum(yerr2osk);
        TensorDual deno = err + 0.01 * err2;
        std::cerr << "deno = " << deno << std::endl;
        deno = TensorDual::where(deno <= TensorDual::zeros_like(deno), TensorDual::ones_like(deno), deno);
        TensorDual res = hm.abs() * err*(1.0/(deno*D)).sqrt();
        return res;
    }


    //PUsh forward the dynamics by a single step
    void dy() {
        //std::cerr << "dy called" << std::endl;
        //if self.theta is not None:
        //    off = self.count * self.thetadim
        //    thetain = TensorDual(self.theta.r[:, off:off+self.thetadim], self.theta.d[:, off:off+self.thetadim])
        if ( thetadims > 0 ) {
            int off = count * thetadims;
            TensorDual thetain = TensorDual(theta.r.index({Slice(), Slice(off, off+thetadims)}),
                                            theta.d.index({Slice(), Slice(off, off+thetadims)}));
        }
        //there is at least one rejected sample
        assert (reject.any().item<bool>());
        TensorDual ym = y.index(reject);
        TensorDual hm = h.index(reject);
        TensorDual dydxm = dydx.index(reject);
        //ytemp = y + h * self.a21 * dydx
        TensorDual ytemp = ym + hm * a21 * dydxm;
        TensorDual xm = x.index(reject);
        TensorDual k2;
        if ( thetadims == 0 ) {
            TensorDual temp = xm + c2 * hm;
            k2 = dynamics.dynamics(temp, ytemp);
        } else {
            TensorDual thetain = TensorDual(this->theta.r.slice(1, this->count*this->thetadims, (this->count+1)*this->thetadims), \
                                            this->theta.d.slice(1, this->count*this->thetadims, (this->count+1)*this->thetadims));
            TensorDual temp = xm + c2 * hm;
            k2 = dynamics.dynamics(temp, ytemp, thetain);
        }
        //std::cerr << "k2 computed" << std::endl;

        //        ytemp = y + h * (self.a31 * dydx + self.a32 * k2)
        ytemp = ym + hm * (a31 * dydxm + a32 * k2);
        TensorDual k3;
        if ( thetadims == 0 ) {
            TensorDual temp = xm + c3 * hm;
            k3 = dynamics.dynamics(temp, ytemp);
        } else {
            TensorDual thetain = TensorDual(this->theta.r.slice(1, this->count*this->thetadims, (this->count+1)*this->thetadims),
                                            this->theta.d.slice(1, this->count*this->thetadims, (this->count+1)*this->thetadims));
            TensorDual temp = x + c3 * h;
            k3 = dynamics.dynamics(temp, ytemp, thetain);
        }
        //std::cerr << "k3 computed" << std::endl;
        //ytemp = y + h * (self.a41 * dydx + self.a43 * k3)
        ytemp = ym + hm * (a41 * dydxm + a43 * k3);
        TensorDual k4;
        if ( thetadims == 0 ) {
            TensorDual temp = xm + c4 * hm;
            k4 = dynamics.dynamics(temp, ytemp);
        } else {
            TensorDual thetain = TensorDual(this->theta.r.slice(1, this->count*this->thetadims, (this->count+1)*this->thetadims),
                                            this->theta.d.slice(1, this->count*this->thetadims, (this->count+1)*this->thetadims));

            TensorDual temp = xm + c4 * hm;
            k4 = dynamics.dynamics(temp, ytemp, thetain);
        }
        //std::cerr << "k4 computed" << std::endl;
        //ytemp = y + h * (self.a51 * dydx + self.a53 * k3 + self.a54 * k4)
        ytemp = ym + hm * (a51 * dydxm + a53 * k3 + a54 * k4);
        /*
                if self.theta is None:
            k5 = self.derivs(x + self.c5 * h, ytemp)
        else:
            k5 = self.derivs(x + self.c5 * h, ytemp, thetain)*/
        TensorDual k5;
        if ( thetadims == 0 ) {
            TensorDual temp = xm + c5 * hm;
            k5 = dynamics.dynamics(temp, ytemp);
        } else {
            TensorDual thetain = TensorDual(this->theta.r.slice(1, this->count*this->thetadims, (this->count+1)*this->thetadims),
                                            this->theta.d.slice(1, this->count*this->thetadims, (this->count+1)*this->thetadims));
            TensorDual temp = xm + c5 * hm;
            k5 = dynamics.dynamics(temp, ytemp, thetain);
        }
        //std::cerr << "k5 computed" << std::endl;
        //ytemp = y + h * (self.a61 * dydx + self.a64 * k4 + self.a65 * k5)
        ytemp = ym + hm * (a61 * dydxm + a64 * k4 + a65 * k5);
        /*
                if self.theta is None:
            k6 = self.derivs(x + self.c6 * h, ytemp)
        else:
            k6 = self.derivs(x + self.c6 * h, ytemp, thetain)*/
        TensorDual k6;
        if ( thetadims == 0 ) {
            TensorDual temp = xm + c6 * hm;
            k6 = dynamics.dynamics(temp, ytemp);
        } else {
            TensorDual thetain = TensorDual(this->theta.r.slice(1, this->count*this->thetadims, (this->count+1)*this->thetadims),
                                            this->theta.d.slice(1, this->count*this->thetadims, (this->count+1)*this->thetadims));
            TensorDual temp = xm + c6 * hm;
            k6 = dynamics.dynamics(temp, ytemp, thetain);
        }
        //std::cerr << "k6 computed" << std::endl;
        //ytemp = y + h * (self.a71 * dydx + self.a74 * k4 + self.a75 * k5 + self.a76 * k6)
        ytemp = ym + hm * (a71 * dydxm + a74 * k4 + a75 * k5 + a76 * k6);
        /*
                if self.theta is None:
            k7 = self.derivs(x + self.c7 * h, ytemp)
        else:
            k7 = self.derivs(x + self.c7 * h, ytemp, thetain)*/
        TensorDual k7;
        if ( thetadims == 0 ) {
            TensorDual temp = xm + c7 * hm;
            k7 = dynamics.dynamics(temp, ytemp);
        } else {
            TensorDual thetain = TensorDual(this->theta.r.slice(1, this->count*this->thetadims, (this->count+1)*this->thetadims),
                                            this->theta.d.slice(1, this->count*this->thetadims, (this->count+1)*this->thetadims));
            TensorDual temp = x + c7 * h;
            k7 = dynamics.dynamics(temp, ytemp, thetain);
        }
        //std::cerr << "k7 computed" << std::endl;
        //ytemp = y + h * (self.a81 * dydx + self.a84 * k4 + self.a85 * k5 + self.a86 * k6 + self.a87 * k7)
        ytemp = ym + hm * (a81 * dydxm + a84 * k4 + a85 * k5 + a86 * k6 + a87 * k7);
        /*
                if self.theta is None:
            k8 = self.derivs(x + self.c8 * h, ytemp)
        else:
            k8 = self.derivs(x + self.c8 * h, ytemp, thetain)*/
        TensorDual k8;
        if ( thetadims == 0 ) {
            TensorDual temp = xm + c8 * hm;
            k8 = dynamics.dynamics(temp, ytemp);
        } else {
            TensorDual thetain = TensorDual(this->theta.r.slice(1, this->count*this->thetadims, (this->count+1)*this->thetadims),
                                            this->theta.d.slice(1, this->count*this->thetadims, (this->count+1)*this->thetadims));
            TensorDual temp = xm + c8 * hm;
            k8 = dynamics.dynamics(temp, ytemp, thetain);
        }
        //std::cerr << "k8 computed" << std::endl;
        //ytemp = y + h * (self.a91 * dydx + self.a94 * k4 + self.a95 * k5 + self.a96 * k6 + self.a97 * k7 + self.a98*k8)
        ytemp = ym + hm * (a91 * dydxm + a94 * k4 + a95 * k5 + a96 * k6 + a97 * k7 + a98*k8);
        /*
                if self.theta is None:
            k9 = self.derivs(x + self.c9 * h, ytemp)
        else:
            k9 = self.derivs(x + self.c9 * h, ytemp, thetain)*/
        TensorDual k9;
        if ( thetadims == 0 ) {
            TensorDual temp = xm + c9 * hm;
            k9 = dynamics.dynamics(temp, ytemp);
        } else {
            TensorDual thetain = TensorDual(this->theta.r.slice(1, this->count*this->thetadims, (this->count+1)*this->thetadims),
                                            this->theta.d.slice(1, this->count*this->thetadims, (this->count+1)*this->thetadims));
            TensorDual temp = x + c9 * h;
            k9 = dynamics.dynamics(temp, ytemp, thetain);
        }
        //std::cerr << "k9 computed" << std::endl;
        //ytemp = y + h * (self.a101 * dydx + self.a104*k4 + self.a105 * k5 + self.a106 * k6 + self.a107 * k7 +self.a108 * k8 + self.a109 * k9)
        ytemp = ym + hm * (a101 * dydxm + a104*k4 + a105 * k5 + a106 * k6 + a107 * k7 +a108 * k8 + a109 * k9);
        /*
                if self.theta is None:
            k10 = self.derivs(x + self.c10 * h, ytemp)
        else:
            k10 = self.derivs(x + self.c10 * h, ytemp, thetain)
        */
        TensorDual k10;
        if ( thetadims == 0 ) {
            TensorDual temp = xm + c10 * hm;
            k10 = dynamics.dynamics(temp, ytemp);
        } else {
            TensorDual thetain = TensorDual(this->theta.r.slice(1, this->count*this->thetadims, (this->count+1)*this->thetadims),
                                            this->theta.d.slice(1, this->count*this->thetadims, (this->count+1)*this->thetadims));
            TensorDual temp = xm + c10 * hm;
            k10 = dynamics.dynamics(temp, ytemp, thetain);
        }
        //std::cerr << "k10 computed" << std::endl;
        //ytemp = y + h * (self.a111 * dydx + self.a114 * k4 + self.a115 * k5 + self.a116 * k6 + self.a117 * k7 +self.a118 * k8 + self.a119 * k9 + self.a1110 * k10)
        ytemp = ym + hm * (a111 * dydxm + a114 * k4 + a115 * k5 + a116 * k6 + a117 * k7 +a118 * k8 + a119 * k9 + a1110 * k10);
        /*
                if self.theta is None:
            k2 = self.derivs(x + self.c11 * h, ytemp)
        else:
            k2 = self.derivs(x + self.c11 * h, ytemp, thetain)
        */
        if ( thetadims == 0 ) {
            TensorDual temp = xm + c11 * hm;
            k2 = dynamics.dynamics(temp, ytemp);
        } else {
            TensorDual thetain = TensorDual(this->theta.r.slice(1, this->count*this->thetadims, (this->count+1)*this->thetadims),
                                            this->theta.d.slice(1, this->count*this->thetadims, (this->count+1)*this->thetadims));
            TensorDual temp = xm + c11 * hm;
            k2 = dynamics.dynamics(temp, ytemp, thetain);
        }
        //std::cerr << "k2 computed" << std::endl;
        //xph = x + h
        TensorDual xph = xm + hm;
        //ytemp = y + h * (self.a121 * dydx + self.a124 * k4 + self.a125 * k5 + self.a126 * k6 + self.a127 * k7 + self.a128 * k8 + self.a129 * k9 + self.a1210 * k10 + self.a1211 * k2)
        ytemp = ym + hm * (a121 * dydx + a124 * k4 + a125 * k5 + a126 * k6 + a127 * k7 + a128 * k8 + a129 * k9 + a1210 * k10 + a1211 * k2);
        /*
                if self.theta is None:
            k3 = self.derivs(xph, ytemp)
        else:
            k3 = self.derivs(xph, ytemp, thetain)
        */
        if ( thetadims == 0 ) {
            TensorDual temp = xph;
            k3 = dynamics.dynamics(temp, ytemp);
        } else {
            TensorDual thetain = TensorDual(this->theta.r.slice(1, this->count*this->thetadims, (this->count+1)*this->thetadims),
                                            this->theta.d.slice(1, this->count*this->thetadims, (this->count+1)*this->thetadims));
            TensorDual temp = xph;
            k3 = dynamics.dynamics(temp, ytemp, thetain);
        }
        //std::cerr << "k3 computed" << std::endl;
        //k4 = self.b1 * dydx + self.b6 * k6 + self.b7 * k7 + self.b8 * k8 + self.b9 * k9 + self.b10 * k10 + self.b11 * k2 + self.b12 * k3
        k4 = b1 * dydxm + b6 * k6 + b7 * k7 + b8 * k8 + b9 * k9 + b10 * k10 + b11 * k2 + b12 * k3;
        //yout = y + h * k4
        //std::cerr << "k4 computed" << std::endl;
        TensorDual youtm = ym + hm * k4;
        yout.index_put_(reject, youtm);
        //std::cerr << "yout updated" << std::endl;
        //yerr = k4-self.bhh1*dydx-self.bhh2*k9-self.bhh3*k3
        TensorDual yerrm = k4-bhh1*dydxm-bhh2*k9-bhh3*k3;
        yerr.index_put_(reject, yerrm);
        //std::cerr << "yerr updated" << std::endl;
        //yerr2 = k5-self.bhh1*dydx-self.bhh2*k9-self.bhh3*k3
        TensorDual yerr2m = k5-bhh1*dydxm-bhh2*k9-bhh3*k3;
        yerr2.index_put_(reject, yerr2m);
        //std::cerr << "yerr2 updated" << std::endl;
        //std::cerr << "dy done" << std::endl;
    }

    //def control(self, err, h, hnext, errold):
    void control(TensorDual& err) {
        assert (reject.any().item<bool>());
        //std::cerr << "control called with reject=" << reject << std::endl;
        //std::cerr << "err=" << err << std::endl;
        TensorDual ym = y.index(reject);
        TensorDual hm = h.index(reject);
        TensorDual dydxm = dydx.index(reject);
        TensorDual errm = err.index(reject);
        TensorDual erroldm = errold.index(reject);
        TensorDual hnextm = hnext.index(reject);
        torch::Tensor rejectm = reject.index({reject});

        //scale = TensorDual.zeros_like(h)
        TensorDual scalem = TensorDual::zeros_like(hm);
        //mask = (err <= 1.0)
        torch::Tensor mask = (errm <= 1.0);
        //std::cerr << "erroldm=" << erroldm << std::endl;
        //std::cerr << "mask=" << mask << std::endl;
        //if mask.any():
        if (mask.any().item<bool>()) {
            //maske = mask & (errold == 0.0)
            torch::Tensor maske = mask & (erroldm == 0.0);
            //std::cerr << "maske=" << maske << std::endl;
            //            if maske.any():
            //    scale.r[maske] = self.maxscale
            //    scale.d[maske] *= 0.0
            if (maske.any().item<bool>()) {
                scalem.index_put_(maske, maxscale);
            }
            //maske = mask & (errold != 0.0)
            maske = mask & (erroldm != 0.0);
            //std::cerr << "maske errold<>0=" << maske << std::endl;
            if (maske.any().item<bool>()) {
                //erroldm = TensorDual(errold.r[maske], errold.d[maske])
                TensorDual erroldmm = erroldm.index(maske);
                //errm = TensorDual(err.r[maske], err.d[maske])
                TensorDual errmm = errm.index(maske);
                //scalem = self.safe * pow(errm, -self.alpha) * pow(erroldm, self.beta)
                TensorDual scalemm = safe * pow(errm, -alpha) * pow(erroldm, beta);
                scalem.index_put_(maske, scalemm);
                //maskem = maske & (scale < self.minscale)
                torch::Tensor maskem = maske & (scalem < minscale);
                //std::cerr << "maskem=" << maskem << std::endl;
                //std::cerr << "scalem=" << scalem << std::endl;
                //if maskem.any():
                if (maskem.any().item<bool>()) {
                    //scale.r[maskem] = self.minscale
                    scalem.index_put_(maskem, minscale);
                }
                //maskem = maske & (scale > self.maxscale)
                maskem = maske & (scalem > maxscale);
                //std::cerr << "Second maskem=" << maskem << std::endl;
                //std::cerr << "scalem=" << scalem << std::endl;
                //std::cerr << "maxscale=" << maxscale << std::endl;
                //if maskem.any():
                if (maskem.any().item<bool>()) {
                    //scale.r[maskem] = self.maxscale
                    scalem.index_put_(maskem, maxscale);
                }

            }
            //maskr = mask & reject
            torch::Tensor maskr = mask & rejectm;
            //if maskr.any():
            //std::cerr << "maskr=" << maskr << std::endl;
            if (maskr.any().item<bool>()) {
                //hm = TensorDual(h.r[maskr], h.d[maskr])
                TensorDual hmm = hm.index(maskr);
                //scalem = TensorDual(scale.r[maskr], scale.d[maskr])
                TensorDual scalemm = scalem.index(maskr);
                //one = TensorDual.ones_like(hm)
                TensorDual one = TensorDual::ones_like(hmm);
                //hnextm = hm * TensorDual.where(scalem < one, scalem, one)
                TensorDual hnextmm = hmm * TensorDual::where(scalemm < one, scalemm, one);
                hnextm.index_put_(maskr, hnextmm);
            }
            //masknr = mask & ~reject
            torch::Tensor masknr = mask & ~rejectm;
            //std::cerr << "masknr=" << masknr << std::endl;
            if (masknr.any().item<bool>()) {
                //hnextm = TensorDual(hnext.r[masknr], hnext.d[masknr])
                TensorDual hnextmm = hnextm.index(masknr);
                // hm = TensorDual(h.r[masknr], h.d[masknr])
                TensorDual hmm = hm.index(masknr);
                //scalem = TensorDual(scale.r[masknr], scale.d[masknr])
                TensorDual scalemm = scalem.index(masknr);
                //hnextm = hm * scalem
                hnextm = hmm * scalemm;
                hnextm.index_put_(masknr, hnextm);
            }
            //errm = TensorDual(err.r[mask], err.d[mask])
            TensorDual errmm = errm.index(mask);
            //ct = TensorDual.ones_like(errm)
            TensorDual ct = TensorDual::ones_like(errmm)*1.0e-4;
            //erroldm = TensorDual.where(errm > ct, errm, ct)
            TensorDual erroldmm = TensorDual::where(errmm > ct, errmm, ct);
            erroldm.index_put_(mask, erroldmm);
            // reject[mask] = False
            rejectm.index_put_({mask}, false);
        }
        //mask = (err > 1.0)
        mask = (err > 1.0);
        //std::cerr << "mask with err > 1.0 =" << mask << std::endl;
        //if mask.any():
        if (mask.any().item<bool>()) {
            //errm = TensorDual(err.r[mask], err.d[mask])
            TensorDual errmm = errm.index(mask);
            //minsct = TensorDual.ones_like(errm) * self.minscale
            TensorDual minsct = TensorDual::ones_like(errmm) * minscale;
            //safepoe = self.safe * pow(errm, -self.alpha)
            TensorDual safepoe = safe * pow(errmm, -alpha);
            //scalem = TensorDual.where(safepoe > minsct, safepoe, minsct)
            TensorDual scalemm = TensorDual::where(safepoe > minsct, safepoe, minsct);
            scalem.index_put_(mask, scalemm);
            //hm = TensorDual(h.r[mask], h.d[mask])
            TensorDual hmm = hm.index(mask);
            //hm = hm*scalem
            hmm = hmm*scalemm;
            hm.index_put_(mask, hmm);
            rejectm.index_put_({mask}, true);
        }
        //Finally update the values of the global variables
        h.index_put_(reject, hm);
        //std::cerr << "h set in control method=" << h << std::endl;
        hnext.index_put_(reject, hnextm);
        errold.index_put_(reject, erroldm);
        y.index_put_(reject, ym);
        dydx.index_put_(reject, dydxm);
        err.index_put_(reject, errm);
        errold.index_put_(reject, erroldm);
        //This should be last
        reject.index_put_({reject.clone()}, rejectm);
    }

    void step(TensorDual& htry) {
        /*if self.theta is not None:
            if self.count > self.theta.r.shape[1]/self.thetadim:
                raise ValueError("The number of steps exceeds the number of parameters.\
                                  You can increase the number of parameters or reduce accuracy")*/
        if ( thetadims > 0 ) {
            if ( count > theta.r.size(1)/thetadims ) {
                throw std::invalid_argument("The number of steps exceeds the number of parameters.  You can increase the number of parameters or reduce accuracy");
            }
        }
        //h = htry.clone()
        h = htry;
        std::cerr << "htry: " << htry << std::endl;
        std::cerr << "htry device" << htry.r.device() << std::endl;
        //reject = torch.ones((self.M,)).bool() & (h > self.eps)
        reject = torch::ones({M}, torch::TensorOptions().dtype(torch::kBool).device(h.r.device())) & (h > EPS);
        std::cerr << "reject: " << reject << std::endl;
        //while (reject).any():
        while (reject.any().item<bool>()) {
            //mask = reject.clone()
            torch::Tensor mask = reject.clone();
            if (mask.any().item<bool>()) {
                 dy();
                 TensorDual err = error();
                 std::cerr << "err: " << err << std::endl;
                 control(err);
            }
            std::cerr << "h=" << h << std::endl;

        }
        /*
                if self.theta is None:
            dydxnew = self.derivs(self.x + h, self.yout)
        else:
            if (self.count + 1) * self.thetadim > self.theta.r.shape[1]:
                raise Exception("Not enough parameters supplied for theta")
            thetain = TensorDual(self.theta.r[:, self.count*self.thetadim:(self.count+1)*self.thetadim],\
                                    self.theta.d[:, self.count*self.thetadim:(self.count+1)*self.thetadim])
            dydxnew = self.derivs(self.x + h, self.yout,  thetain)
        */
        TensorDual dydxnew;
        if ( thetadims == 0 ) {
            TensorDual temp = x + h;
            dydxnew = dynamics.dynamics(temp, yout);
        } else {
            if ( (count+1)*thetadims > theta.r.size(1) ) {
                throw std::invalid_argument("Not enough parameters supplied for theta");
            }
            TensorDual thetain = TensorDual(theta.r.slice(1, count*thetadims, (count+1)*thetadims),
                                            theta.d.slice(1, count*thetadims, (count+1)*thetadims));
            TensorDual temp = x + h;
            dydxnew = dynamics.dynamics(temp, yout, thetain);
        }
        //self.dydx = dydxnew
        dydx = dydxnew;
        //self.y    = self.yout
        y = yout;
        //self.xold = self.x
        xold = x;
        //self.x    = self.x + h
        x = x + h;
        //self.hdid = h
        hdid = h;
        //self.count+=1
        count += 1;
        if (dense) {
             y_dense.push_back(y.clone()); //Clone to make sure the data is not changed
             x_dense.push_back(x.clone()); //Clone to make sure the data is not changed
         }



    }




};
#endif