#include <gtest/gtest.h>
#include <torch/torch.h>
#include <random>
#include <janus/tensordual.hpp>
#include <janus/janus_util.hpp>
using namespace janus;

using TensorIndex = torch::indexing::TensorIndex;
using Slice = torch::indexing::Slice;

TEST(HamiltonianTest, dynsTest) 
{
    //Test for the construction of the dynamics for vdp oscillator
    //using the hamiltonian formulation and dual numbers
    auto vdp = [](const torch::Tensor& x) -> torch::Tensor
    {
        torch::Tensor x1 = x.index({Slice(), 0});
        torch::Tensor x2 = x.index({Slice(), 1});
        torch::Tensor x3 = x.index({Slice(), 2}); //This is mu
        torch::Tensor dxdt = torch::zeros_like(x);
        dxdt.index_put_({Slice(), 0}, x2);
        dxdt.index_put_({Slice(), 1}, x3*(-x1 + 0.5 * x2 * (1 - x1 * x1)));
        return dxdt;
    };

    int M = 1;
    int N = 3;
    auto H = [](const TensorDual &x, const TensorDual &p) -> TensorDual
    { 
        TensorDual p1 = p.index({Slice(), 0});
        TensorDual p2 = p.index({Slice(), 1});
        TensorDual p3 = p.index({Slice(), 2});
        TensorDual x1 = x.index({Slice(), 0});
        TensorDual x2 = x.index({Slice(), 1});
        TensorDual x3 = x.index({Slice(), 2});
        
        auto H=p1*x2+p2*(x3*(-x1 + 0.5 * x2 * (1 - x1 * x1)));
        return H;
    };
    //The Hamiltonian should yield the dynamics throught the dual gradients

    torch::Tensor p0 = torch::ones({M, N}, torch::kFloat64);

    torch::Tensor p0dual = torch::eye(3, dtype(torch::kFloat64)).repeat({M, 1, 1});
    auto p0d = TensorDual(p0, p0dual);
    torch::Tensor x0 = torch::ones({M, N}, torch::kFloat64);

    torch::Tensor x0dual = torch::zeros_like(p0dual);
    auto x0d = TensorDual(x0, x0dual);

    //Get the dynamics from the anaylitic function
    auto true_dynamics = vdp(x0).squeeze();
    //Get the dynamics from the hamiltonian
    auto hamiltonian_dynamics = H(x0d, p0d);
    auto dx1dt = true_dynamics.index({0});
    auto dx2dt = true_dynamics.index({1});
    auto dx3dt = true_dynamics.index({2});
    //The dynamics are contained in the dual part of the hamiltonian
    auto dHdp1 = hamiltonian_dynamics.d.index({Slice(), 0, 0}).squeeze();
    auto dHdp2 = hamiltonian_dynamics.d.index({Slice(), 0, 1}).squeeze();
    auto dHdp3 = hamiltonian_dynamics.d.index({Slice(), 0, 2}).squeeze();

    EXPECT_TRUE(torch::allclose(dx1dt, dHdp1));
    EXPECT_TRUE(torch::allclose(dx2dt, dHdp2));
    EXPECT_TRUE(torch::allclose(dx3dt, dHdp3));

}

template <typename T, typename std::enable_if<std::is_same<T, torch::Tensor>::value || std::is_same<T, TensorDual>::value, int>::type = 0>
T H(const T &x, const T &p, double W)
{ 
  T p1 = p.index({Slice(), 0});
  T p2 = p.index({Slice(), 1});
  T x1 = x.index({Slice(), 2});
  T x2 = x.index({Slice(), 3});
        
  auto H=p1*x2-(p2*(1-x1*x1)*x2-x1).pow(2)/W;
  return H;
};


/**
 * Explicit function for dynamics
 */
TensorDual vdpdyns(const TensorDual& y, double W)
{
  TensorDual p1 = y.index({Slice(), 0});
  TensorDual p2 = y.index({Slice(), 1});
  TensorDual x1 = y.index({Slice(), 2});
  TensorDual x2 = y.index({Slice(), 3});
  int M = y.d.size(0);
  int N = y.d.size(1);
  int D = y.d.size(2);
  TensorDual dyns = TensorDual(torch::zeros({M, N}, dtype(torch::kFloat64)),
                               torch::zeros({M, N, D}, dtype(torch::kFloat64)));
  dyns.index_put_({Slice(), 0}, x2);
  dyns.index_put_({Slice(), 1}, (x2*(1-x1*x1)-x1)*(-p2*(x2*(1-x1*x1)-x1)/W));
  dyns.index_put_({Slice(), 2}, p2*p2*(x2*(1-x1*x1)-x1)*(2*x1*x1+1)/W);
  dyns.index_put_({Slice(), 3}, p1-p2*p2*(x2*(1-x1*x1)-x1)*(1-x1*x1)/W);
  //std::cerr << "dyns=";
  //janus::print_dual(dyns);
  return dyns;
};


/**
 * Explicit function for jacobian
 */
torch::Tensor vdpjac(const torch::Tensor& y, 
                     double W)
{
  torch::Tensor p1 = y.index({Slice(), 0});
  torch::Tensor p2 = y.index({Slice(), 1});
  torch::Tensor x1 = y.index({Slice(), 2});
  torch::Tensor x2 = y.index({Slice(), 3});
  torch::Tensor dydt = y*0.0;
  torch::Tensor jac = torch::zeros({4,4}, torch::kFloat64); 
  //the dynamics are rows and the state/costate are columns
  jac.index_put_({Slice(), 2, 3}, 1.0);


  auto f_x1 = x2 * (1 - x1*x1) - x1;
  auto g_x1 = -p2 * (x2 * (1 - x1*x1) - x1) / W;
  jac.index_put_({Slice(), 3, 2}, (-2 * x2 * x1 + 1) * g_x1 + f_x1 * (-p2 / W * (2 * x2 * x1 - 1)));
  auto f_x2 = x2 * (1 - x1*x1) - x1;
  auto g_x2 = -p2 * (x2 * (1 - x1*x1) - x1) / W;
  jac.index_put_({Slice(), 3, 2}, (1 - x1*x1) * g_x2 + f_x2 * (-p2 / W * (1 - x1*x1)));
  jac.index_put_({Slice(), 3, 0}, (x2*(1-x1*x1)-x1)*(-(x2*(1-x1*x1)-x1)/W));

  
  jac.index_put_({Slice(), 0, 2}, (p2*p2 * ((x2 * (1 - x1*x1) - x1) * (4 * x1) + (2 * x1*x1 + 1) * (-2 * x2 * x1)))/W);
  jac.index_put_({Slice(), 0, 3}, (p2*p2 * ((1 - x1*x1) * (2 * x1*x1 + 1) + (x2 * (1 - x1*x1) - x1) * 4 * x1*x1))/W);
  jac.index_put_({Slice(), 0, 1},  (2 * p2 * (x2 * (1 - x1*x1) - x1) * (2 * x1*x1 + 1))/W);
  
  
  auto term1 = -2 * p2*p2 * x1 * (x2 * (1 - x1*x1) - x1);
  auto term2 = 2 * p2*p2 * (x2 * (1 - x1*x1) - x1) * (1 - 3 * x1*x1) * (1 - x1*x1);
  jac.index_put_({Slice(), 1, 2}, (term1 + term2) / W);
  jac.index_put_({Slice(), 1, 3}, -p2*p2 * (1 - x1*x1) * (1 - x1*x1) / W);
  jac.index_put_({Slice(), 1, 0}, 1.0);
  jac.index_put_({Slice(), 1, 1}, -2 * p2 * (x2 * (1 - x1*x1) - x1) * (1 - x1*x1) / W);
  return jac;
};


TensorDual rk4( const TensorDual &y, double W, double h)
{
    auto k1 = vdpdyns(y,  W);
    auto k2 = vdpdyns(y + h/2*k1, W);
    auto k3 = vdpdyns(y + h/2*k2, W);
    auto k4 = vdpdyns(y + h*k3, W);
    return y + h/6*(k1 + 2*k2 + 2*k3 + k4);
}


torch::Tensor pHpx(const torch::Tensor &x, const torch::Tensor &p, double W)
{
    // Create tensors with gradient tracking for x and no gradient tracking for p
    auto xt = x.clone().detach().requires_grad_(true);
    auto pt = p.clone().detach().requires_grad_(false);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the gradient of Hvalue with respect to xt
    auto grad_H_wrt_x = torch::autograd::grad({Hvalue}, {xt}, {torch::ones_like(Hvalue)})[0];

    // Return the gradient of H with respect to x
    return grad_H_wrt_x;
}

torch::Tensor pHpp(const torch::Tensor &x, const torch::Tensor &p, double W)
{
    // Create tensors with gradient tracking for p and no gradient tracking for x
    auto xt = x.clone().detach().requires_grad_(false);
    auto pt = p.clone().detach().requires_grad_(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the gradient of Hvalue with respect to pt
    auto grad_H_wrt_p = torch::autograd::grad({Hvalue}, {pt}, {torch::ones_like(Hvalue)})[0];

    // Return the gradient of H with respect to p
    return grad_H_wrt_p;
}

torch::Tensor ppHpxpx(const torch::Tensor &x, const torch::Tensor &p, double W)
{
    // Create tensors with gradient tracking for x and no gradient tracking for p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(false);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the first-order gradient with respect to xt
    auto first_order_grad = torch::autograd::grad({Hvalue}, {xt}, {torch::ones_like(Hvalue)}, true, true)[0];

    // Compute the second-order gradient with respect to xt
    auto second_order_grad = torch::autograd::grad({first_order_grad}, {xt}, {torch::ones_like(first_order_grad)})[0];

    // Return the second-order derivative
    return second_order_grad;
}
torch::Tensor ppHpppp(const torch::Tensor &x, const torch::Tensor &p, double W)
{
    // Create tensors with gradient tracking for p and no gradient tracking for x
    auto xt = x.clone().detach().set_requires_grad(false);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the first-order gradient with respect to pt
    auto first_order_grad = torch::autograd::grad({Hvalue}, {pt}, {torch::ones_like(Hvalue)}, true, true)[0];

    // Compute the second-order gradient with respect to pt
    auto second_order_grad = torch::autograd::grad({first_order_grad}, {pt}, {torch::ones_like(first_order_grad)})[0];

    // Return the second-order derivative
    return second_order_grad;
}

torch::Tensor ppHpxpp(const torch::Tensor &x, const torch::Tensor &p, double W)
{
    // Create tensors with gradient tracking for x and p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the first-order gradient with respect to xt
    auto grad_H_wrt_x = torch::autograd::grad({Hvalue}, {xt}, {torch::ones_like(Hvalue)}, true, true)[0];

    // Compute the second-order gradient with respect to pt
    auto second_order_grad = torch::autograd::grad({grad_H_wrt_x}, {pt}, {torch::ones_like(grad_H_wrt_x)})[0];

    // Return the mixed second-order derivative
    return second_order_grad;
}

torch::Tensor pppHpxpxpx(const torch::Tensor &x, const torch::Tensor &p, double W)
{
    // Create tensors with gradient tracking for x and no gradient tracking for p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(false);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the first-order gradient with respect to xt
    auto first_order_grad = torch::autograd::grad({Hvalue}, {xt}, {torch::ones_like(Hvalue)}, true, true)[0];

    // Compute the second-order gradient with respect to xt
    auto second_order_grad = torch::autograd::grad({first_order_grad}, {xt}, {torch::ones_like(first_order_grad)}, true, true)[0];

    // Compute the third-order gradient with respect to xt
    auto third_order_grad = torch::autograd::grad({second_order_grad}, {xt}, {torch::ones_like(second_order_grad)})[0];

    // Return the third-order derivative
    return third_order_grad;
}

torch::Tensor pppHpppppp(const torch::Tensor &x, const torch::Tensor &p, double W)
{
    // Create tensors with gradient tracking for p and no gradient tracking for x
    auto xt = x.clone().detach().set_requires_grad(false);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the first-order gradient with respect to pt
    auto first_order_grad = torch::autograd::grad({Hvalue}, {pt}, {torch::ones_like(Hvalue)}, true, true)[0];

    // Compute the second-order gradient with respect to pt
    auto second_order_grad = torch::autograd::grad({first_order_grad}, {pt}, {torch::ones_like(first_order_grad)}, true, true)[0];

    // Compute the third-order gradient with respect to pt
    auto third_order_grad = torch::autograd::grad({second_order_grad}, {pt}, {torch::ones_like(second_order_grad)})[0];

    // Return the third-order derivative
    return third_order_grad;
}

torch::Tensor pppHpppppx(const torch::Tensor &x, const torch::Tensor &p, double W)
{
    // Create tensors with gradient tracking for x and p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the first-order gradient with respect to xt
    auto first_order_grad_x = torch::autograd::grad({Hvalue}, {xt}, {torch::ones_like(Hvalue)}, true, true)[0];

    // Compute the second-order gradient with respect to xt
    auto second_order_grad_xx = torch::autograd::grad({first_order_grad_x}, {xt}, {torch::ones_like(first_order_grad_x)}, true, true)[0];

    // Compute the third-order mixed gradient with respect to xt and pt
    auto third_order_mixed_grad = torch::autograd::grad({second_order_grad_xx}, {pt}, {torch::ones_like(second_order_grad_xx)})[0];

    // Return the mixed third-order derivative
    return third_order_mixed_grad;
}




torch::Tensor pppHpppxpx(const torch::Tensor &x, const torch::Tensor &p, double W)
{
    // Create tensors with gradient tracking for p and x
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the first-order gradient with respect to pt
    auto first_order_grad_p = torch::autograd::grad({Hvalue}, {pt}, {torch::ones_like(Hvalue)}, true, true)[0];

    // Compute the second-order gradient with respect to pt
    auto second_order_grad_pp = torch::autograd::grad({first_order_grad_p}, {pt}, {torch::ones_like(first_order_grad_p)}, true, true)[0];

    // Compute the third-order mixed gradient with respect to pt and xt
    auto third_order_mixed_grad = torch::autograd::grad({second_order_grad_pp}, {xt}, {torch::ones_like(second_order_grad_pp)})[0];

    // Return the mixed third-order derivative
    return third_order_mixed_grad;
}


/**
 * Test for explicit and implicit methods to calculating 
 * the dual dynamics using an explicit method to calculate the dynamics 
 * using rk4 versus an implicit method using the hamiltonian and a combination
 * of dual numbers and backpropagation
 *
 * Using the implicit method the hamiltonian alone is used to calculate the dynamics
 * as well as the sensitivity of the dynamics wrt the initial costate conditions p0
 * after N steps of rk4 integration.
 * That in turn is compared with the implicit method of calculating the dynamics and the
 * dual sensitivity to initial conditions 
 * \dot{x}=\partial_x H + \epsilon d_{p0} \partial_x H
 * \dot{p}=-\partial_p H + \epsilon d_{x0} \partial_p H
 * Now using the chain rule
 * d_{p0} \partial_x H = \partial_x H \partial_p H x H d_{p0} p
 * where x is the tensor product
 * d_{x0} \partial_p H = \partial_p H \partial_x H x d_{x0} x
 * \partial_x H \partial_p H  and \partial_p H \partial_x H can be calcuated using backpropagation
 * whereas d_{p0} p and d_{p0} p are given by the dual part of the state and costate space respectively
 * calculated using dual numbers
 */
TEST(HamiltonianTest, DynsExplVsImplTest) 
{


    int M = 1;
    int N = 2;
    double W = 0.001;
    double h = 0.0001;





    //The Hamiltonian should yield the dynamics throught the dual gradients

    torch::Tensor y0 = torch::zeros({M, 2*N}, torch::kFloat64);
    y0.index_put_({Slice(), 0}, -1.0); //p1
    y0.index_put_({Slice(), 1}, -1.0); //p2
    y0.index_put_({Slice(), 2}, 2.0); //x1
    y0.index_put_({Slice(), 3}, 0.0); //x2

    torch::Tensor y0dual = torch::zeros({M, 2*N, N}, torch::kFloat64);
    y0dual.index_put_({Slice(), 0, 0}, 1.0);
    y0dual.index_put_({Slice(), 1, 1}, 1.0);
    auto y0ted = TensorDual(y0, y0dual);
    for ( int i=0; i < 1000; i++)
    {
        y0ted = rk4(y0ted, W, h);
        //std::cerr << "y0ted=";
        //janus::print_dual(y0ted);
    }
    //std::cerr << "Final y0 real=" << y0ted.r << std::endl;
    //std::cerr << "Final y0 dual=" << y0ted.d << std::endl;
    //Calculate d/dp0 \partial H/\partial x by applying the dynamics one more time
    auto dydt = vdpdyns(y0ted, W);
    //We need to check the dual part of the dynamics against the implicit method
    auto x = y0ted.r.index({Slice(), Slice(2, 4)});
    auto p = y0ted.r.index({Slice(), Slice(0, 2)});
    auto dxdp0 = y0ted.d.index({Slice(), Slice(2, 4), Slice(0, 2)});
    auto dpdp0 = y0ted.d.index({Slice(), Slice(0, 2), Slice(0, 2)});
    auto pHpxval = pHpx(x, p, W);
    EXPECT_TRUE(torch::allclose(dydt.r.index({Slice(), Slice(2, 4)}), pHpxval));
    auto pHppval = pHpp(x, p, W);
    EXPECT_TRUE(torch::allclose(dydt.r.index({Slice(), Slice(0, 2)}), pHppval));
    //Now for the second order sensitivities
    auto ppHpxpxval = ppHpxpx(x, p, W);
    auto ppHppppval = ppHpppp(x, p, W);
    auto ppHpxppval = ppHpxpp(x, p, W);
    auto dpHpxdp0 = torch::bmm(ppHpxpxval, dxdp0);
    auto dpHppdp0 = torch::bmm(ppHppppval, dpdp0);
    EXPECT_TRUE(torch::allclose(dpHpxdp0, dydt.d.index({Slice(), Slice(2, 4), Slice()})));
    EXPECT_TRUE(torch::allclose(dpHppdp0, dydt.d.index({Slice(), Slice(0, 2), Slice()})));
 

}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
