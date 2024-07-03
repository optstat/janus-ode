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
  T x1 = x.index({Slice(), 0});
  T x2 = x.index({Slice(), 1});
        
  auto H=p1*x2-(p2*((1-x1*x1)*x2-x1)).pow(2)/W;
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
  dyns.index_put_({Slice(), 0}, -2*p2*p2*((1-x1*x1)*x2-x1)*(-2*x1*x2-1)/W);
  dyns.index_put_({Slice(), 1}, p1-2*p2*p2*((1-x1*x1)*x2-x1)*((1-x1*x1))/W);
  dyns.index_put_({Slice(), 2}, x2);
  dyns.index_put_({Slice(), 3}, -2*p2*(x2*(1-x1*x1)-x1).pow(2)/W);

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
  jac.index_put_({Slice(), 0, 1}, -4*p2*(x2*(1-x1*x1)-x1)*(-2*x1*x2-1)/W);
  jac.index_put_({Slice(), 0, 2},  -2 * p2*p2 * ((1 - x1*x1) * x2 - x1) * (-2 * x1 * x2 - 1) / W);
  jac.index_put_({Slice(), 0, 3}, -2 * p2*p2 * ((1 - x1*x1) * x2 - x1) * (-2 * x1 * x2 - 1) / W);
  
  jac.index_put_({Slice(), 1, 0}, 1.0);
  jac.index_put_({Slice(), 1, 1}, -4*p2*((1 - x1*x1)*x2 - x1)*(1 - x1*x1)/W);
  jac.index_put_({Slice(), 1, 2}, -2*p2*p2*((-2*x1)*((1 - x1*x1)*x2 - x1) + (1 - x1*x1)*(-2*x1*x2 - 1))/W);
  jac.index_put_({Slice(), 1, 3}, -2*p2*p2*(1 - x1*x1).pow(2)/W);

  jac.index_put_({Slice(), 2, 3}, 1.0);

  jac.index_put_({Slice(), 3, 1}, -2*(x2*(1 - x1*x1) - x1).pow(2)/W);
  jac.index_put_({Slice(), 3, 2}, -4*p2*(x2*(1 - x1*x1) - x1)*(-2*x1*x2 - 1)/W);
  jac.index_put_({Slice(), 3, 3}, -4*p2*(x2*(1 - x1*x1) - x1)*(1 - x1*x1)/W);


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


torch::Tensor pxH(const torch::Tensor &x, const torch::Tensor &p, double W)
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

torch::Tensor ppH(const torch::Tensor &x, const torch::Tensor &p, double W)
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


torch::Tensor ppppH(const torch::Tensor &x, const torch::Tensor &p, double W)
{
    // Create tensors with gradient tracking for x and no gradient tracking for p
    auto xt = x.clone().detach().set_requires_grad(false);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the first-order gradient with respect to xt
    auto first_order_grad = torch::autograd::grad({Hvalue}, {pt}, {torch::ones_like(Hvalue)}, true, true, true)[0];

    // Initialize a tensor to hold the second-order gradients (Hessian)
    auto batch_size = x.size(0);
    auto x_dim = x.size(1);
    torch::Tensor hessian = torch::zeros({batch_size, x_dim, x_dim}, x.options());

    // Compute the second-order gradients for each dimension
    for (int i = 0; i < x_dim; ++i)
    {

        if (pt.grad().defined()) {
            pt.grad().zero_();
        }

        // Compute the gradient of the i-th component of the first-order gradient with respect to xt
        //This is the row of the original vector 
        //which is then used to populate the row of the hessian
        //\partial x_j (\partial_{x_i} H )
        auto grad_p_i = torch::autograd::grad({first_order_grad.index({Slice(), i})}, 
                                              {pt}, 
                                              {torch::ones_like(first_order_grad.index({Slice(), i}))}, 
                                              true, 
                                              false, true)[0];
        
        // Assign the gradient to the corresponding row of the Hessian matrix
        hessian.index_put_({Slice(), i, Slice()}, grad_p_i.clone());
    }

    // Return the Hessian
    return hessian;
}
torch::Tensor ppppppH(const torch::Tensor &x, const torch::Tensor &p, double W)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the first-order gradient of H with respect to p
    auto grad_H_p = torch::autograd::grad({Hvalue}, {pt}, {torch::ones_like(Hvalue)}, true, true, true)[0];

    // Initialize a tensor to hold the third-order gradients
    auto batch_size = p.size(0);
    auto p_dim = p.size(1);
    torch::Tensor third_order_derivative = torch::zeros({batch_size, p_dim, p_dim, p_dim}, p.options());

    // Compute the second-order and third-order gradients
    for (int i = 0; i < p_dim; ++i)
    {
        for (int j = 0; j < p_dim; ++j)
        {
            for (int k = 0; k < p_dim; ++k)
            {
                // Zero out the gradients of pt before computing the gradient for the next dimension
                if (pt.grad().defined()) {
                    pt.grad().zero_();
                }
                if (xt.grad().defined()) {
                    xt.grad().zero_();
                }

                // Compute the gradient of the i-th component of grad_H_p with respect to p
                auto grad_H_p_i = torch::autograd::grad({grad_H_p.select(1, i)}, 
                                                        {pt}, 
                                                        {torch::ones_like(grad_H_p.select(1, i))}, 
                                                        true, 
                                                        false, 
                                                        true)[0];
                
                // Compute the gradient of the j-th component of grad_H_p_i with respect to p
                auto grad_H_p_ij = torch::autograd::grad({grad_H_p_i.select(1, j)}, 
                                                        {pt}, 
                                                        {torch::ones_like(grad_H_p_i.select(1, j))}, 
                                                        true, 
                                                        false, 
                                                        true)[0];
                
                // Compute the gradient of the k-th component of grad_H_p_ij with respect to p
                auto grad_H_p_ijk = torch::autograd::grad({grad_H_p_ij.select(1, k)}, 
                                                        {pt}, 
                                                        {torch::ones_like(grad_H_p_ij.select(1, k))}, 
                                                        true, 
                                                        false, 
                                                        true)[0];

                // Assign the gradient to the corresponding slice of the third-order derivative tensor
                third_order_derivative.select(1, i).select(1, j).select(1, k).copy_(grad_H_p_ijk);
            }
        }
    }

    // Return the third-order derivative tensor
    return third_order_derivative;
}


torch::Tensor pppppxH(const torch::Tensor &x, const torch::Tensor &p, double W)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the first-order gradient of H with respect to x
    auto grad_H_x = torch::autograd::grad({Hvalue}, {xt}, {torch::ones_like(Hvalue)}, true, true, true)[0];

    // Initialize a tensor to hold the third-order mixed gradients
    auto batch_size = x.size(0);
    auto x_dim = x.size(1);
    auto p_dim = p.size(1);
    torch::Tensor third_order_mixed_hessian = torch::zeros({batch_size, x_dim, p_dim, p_dim}, x.options());

    // Compute the second-order and third-order gradients
    for (int i = 0; i < x_dim; ++i)
    {
        // Compute the gradient of the i-th component of grad_H_x with respect to p
        auto grad_H_x_i = torch::autograd::grad({grad_H_x.select(1, i)}, 
                                                {pt}, 
                                                {torch::ones_like(grad_H_x.select(1, i))}, 
                                                true, 
                                                false, 
                                                true)[0];

        for (int j = 0; j < p_dim; ++j)
        {
            // Zero out the gradients of xt and pt before computing the gradient for the next dimension
            if (xt.grad().defined()) {
                xt.grad().zero_();
            }
            if (pt.grad().defined()) {
                pt.grad().zero_();
            }

            // Compute the gradient of the j-th component of grad_H_x_i with respect to p
            auto grad_H_x_ij = torch::autograd::grad({grad_H_x_i.select(1, j)}, 
                                                     {pt}, 
                                                     {torch::ones_like(grad_H_x_i.select(1, j))}, 
                                                     true, 
                                                     false, 
                                                     true)[0];

            // Assign the gradient to the corresponding slice of the third-order mixed Hessian tensor
            for (int k = 0; k < p_dim; ++k)
            {
                third_order_mixed_hessian.select(1, i).select(1, j).select(1, k).copy_(grad_H_x_ij.select(1, k));
            }
        }
    }

    // Return the third-order mixed Hessian tensor
    return third_order_mixed_hessian;
}

torch::Tensor pxpxH(const torch::Tensor &x, const torch::Tensor &p, double W)
{
    // Create tensors with gradient tracking for x and no gradient tracking for p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(false);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the first-order gradient with respect to xt
    auto first_order_grad = torch::autograd::grad({Hvalue}, {xt}, {torch::ones_like(Hvalue)}, true, true, true)[0];

    // Initialize a tensor to hold the second-order gradients (Hessian)
    auto batch_size = x.size(0);
    auto x_dim = x.size(1);
    torch::Tensor hessian = torch::zeros({batch_size, x_dim, x_dim}, x.options());

    // Compute the second-order gradients for each dimension
    for (int i = 0; i < x_dim; ++i)
    {

        if (xt.grad().defined()) {
            xt.grad().zero_();
        }

        // Compute the gradient of the i-th component of the first-order gradient with respect to xt
        //This is the row of the original vector 
        //which is then used to populate the row of the hessian
        //\partial x_j (\partial_{x_i} H )
        auto grad_x_i = torch::autograd::grad({first_order_grad.index({Slice(), i})}, 
                                              {xt}, 
                                              {torch::ones_like(first_order_grad.index({Slice(), i}))}, 
                                              true, 
                                              false, true)[0];
        
        // Assign the gradient to the corresponding row of the Hessian matrix
        hessian.index_put_({Slice(), i, Slice()}, grad_x_i.clone());
    }

    // Return the Hessian
    return hessian;
}

torch::Tensor pxppH(const torch::Tensor &x, const torch::Tensor &p, double W)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the gradient of H with respect to x
    auto grad_H_p = torch::autograd::grad({Hvalue}, {pt}, {torch::ones_like(Hvalue)}, true, true, true)[0];

    // Initialize a tensor to hold the mixed second-order gradients
    auto batch_size = x.size(0);
    auto x_dim = x.size(1);
    auto p_dim = p.size(1);
    torch::Tensor mixed_hessian = torch::zeros({batch_size, p_dim, x_dim}, x.options());
    // Compute the second-order gradients for each dimension of x and p
    for (int i = 0; i < p_dim; ++i)
    {
        // Zero out the gradients of xt and pt before computing the gradient for the next dimension
        if (xt.grad().defined()) {
            xt.grad().zero_();
        }
        if (pt.grad().defined()) {
            pt.grad().zero_();  
        }


        // Compute the gradient of the i-th component of grad_H_x with respect to xt
        auto grad_H_p_i = torch::autograd::grad({grad_H_p.select(1, i)}, 
                                                {xt}, 
                                                {torch::ones_like(grad_H_p.select(1, i))}, 
                                                true, 
                                                false, 
                                                true)[0];
        
        // Assign the gradient to the corresponding slice of the mixed Hessian matrix
        mixed_hessian.select(1, i).copy_(grad_H_p_i);
    }

    // Return the mixed Hessian
    return mixed_hessian;
}


torch::Tensor pppxH(const torch::Tensor &x, const torch::Tensor &p, double W)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the gradient of H with respect to x
    auto grad_H_x = torch::autograd::grad({Hvalue}, {xt}, {torch::ones_like(Hvalue)}, true, true, true)[0];

    // Initialize a tensor to hold the mixed second-order gradients
    auto batch_size = x.size(0);
    auto x_dim = x.size(1);
    auto p_dim = p.size(1);
    torch::Tensor mixed_hessian = torch::zeros({batch_size, x_dim, p_dim}, x.options());
    // Zero out the gradients of pt

    // Compute the second-order gradients for each dimension of x and p
    for (int i = 0; i < x_dim; ++i)
    {
        if (pt.grad().defined()) {
            pt.grad().zero_();
        }
        if (xt.grad().defined()) {
            xt.grad().zero_();
        }


        // Compute the gradient of the i-th component of grad_H_x with respect to pt
        auto grad_H_x_i = torch::autograd::grad({grad_H_x.select(1, i)}, 
                                                {pt}, 
                                                {torch::ones_like(grad_H_x.select(1, i))}, 
                                                true, 
                                                false, 
                                                true)[0];
        
        // Assign the gradient to the corresponding slice of the mixed Hessian matrix
        mixed_hessian.select(1, i).copy_(grad_H_x_i);
    }

    // Return the mixed Hessian
    return mixed_hessian;
}





torch::Tensor pppxpxH(const torch::Tensor &x, const torch::Tensor &p, double W)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the first-order gradient of H with respect to x
    auto grad_H_x = torch::autograd::grad({Hvalue}, {xt}, {torch::ones_like(Hvalue)}, true, true, true)[0];

    // Initialize a tensor to hold the third-order mixed gradients
    auto batch_size = x.size(0);
    auto x_dim = x.size(1);
    auto p_dim = p.size(1);
    torch::Tensor third_order_mixed_hessian = torch::zeros({batch_size, p_dim, x_dim, x_dim}, x.options());

    // Compute the second-order and third-order gradients
    for (int i = 0; i < x_dim; ++i)
    {
        // Compute the gradient of the i-th component of grad_H_x with respect to x
        auto grad_H_x_i = torch::autograd::grad({grad_H_x.select(1, i)}, 
                                                {xt}, 
                                                {torch::ones_like(grad_H_x.select(1, i))}, 
                                                true, 
                                                false, 
                                                true)[0];

        for (int j = 0; j < x_dim; ++j)
        {
            // Zero out the gradients of xt and pt before computing the gradient for the next dimension
            if (xt.grad().defined()) {
                xt.grad().zero_();
            }
            if (pt.grad().defined()) {
                pt.grad().zero_();
            }

            // Compute the gradient of the j-th component of grad_H_x_i with respect to x
            auto grad_H_x_ij = torch::autograd::grad({grad_H_x_i.select(1, j)}, 
                                                     {pt}, 
                                                     {torch::ones_like(grad_H_x_i.select(1, j))}, 
                                                     true, 
                                                     false, 
                                                     true)[0];

            // Assign the gradient to the corresponding slice of the third-order mixed Hessian tensor
            for (int k = 0; k < p_dim; ++k)
            {
                third_order_mixed_hessian.select(1, k).select(1, i).select(1, j).copy_(grad_H_x_ij.select(1, k));
            }
        }
    }

    // Return the third-order mixed Hessian tensor
    return third_order_mixed_hessian;
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
    double h = 0.00001;





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
    std::cerr << "y0ted=";
    janus::print_dual(y0ted);
    //Update the state space using euler for one step
    auto yted =y0ted.clone();

    for ( int i=0; i < 1; i++)
    {
        yted = rk4(yted, W, h);
    }
    //yted = yted +h*vdpdyns(yted, W);

    //std::cerr << "Final y0 real=" << y0ted.r << std::endl;
    //std::cerr << "Final y0 dual=" << y0ted.d << std::endl;
    //Calculate d/dp0 \partial H/\partial x by applying the dynamics one more time
    auto dydt = vdpdyns(yted, W);//Calculate the dynamics at the new location
    //Do a dynamics check
    //We need to check the dual part of the dynamics against the implicit method
    auto x = yted.r.index({Slice(), Slice(2, 4)});
    auto p = yted.r.index({Slice(), Slice(0, 2)});
    auto pHppval = ppH(x, p, W);  //dot{x}
    EXPECT_TRUE(torch::allclose(dydt.r.index({Slice(), Slice(2, 4)}), pHppval));
    auto pHpxval = pxH(x, p, W); //dot{p}
    EXPECT_TRUE(torch::allclose(dydt.r.index({Slice(), Slice(0, 2)}), pHpxval));



    //Now for the dual part of the sensitivities
    auto dxdp0 = yted.d.index({Slice(), Slice(2, 4), Slice()});
    //std::cerr << "dxdp0=" << dxdp0 << std::endl;
    auto dpdp0 = yted.d.index({Slice(), Slice(0, 2), Slice()});
    //std::cerr << "dpdp0=" << dpdp0 << std::endl;
    //Now for the second order sensitivities
    auto ppppHval = ppppH(x, p, W);
    auto pxpxHval = pxpxH(x, p, W);
    auto pxppHval = pxppH(x, p, W);
    auto pppxHval = pppxH(x, p, W);
    std::cerr << "ppppHval=" << ppppHval << std::endl;
    std::cerr << "pxpxHval=" << pxpxHval << std::endl;
    //std::cerr << "pxppHval=" << pxppHval << std::endl;
    //std::cerr << "pppxHval=" << pppxHval << std::endl;
    EXPECT_TRUE(torch::allclose(pppxHval, pxppHval.transpose(1, 2)));

    
    auto dp0pxH = torch::einsum("mij, mjk->mik", {pxpxHval, dxdp0})+
                  //****Note here that the index wrt is i and the index wrt p is j****
                  torch::einsum("mij, mjk->mik", {pppxHval, dpdp0});  
    //std::cerr << "actual dp0pxH=" << dydt.d.index({Slice(), Slice(0,2)}) << std::endl;
    //std::cerr << "inferred dp0pxH=" << dp0pxH << std::endl;
    EXPECT_TRUE(torch::allclose(dp0pxH, dydt.d.index({Slice(), Slice(0, 2), Slice()})));
 

}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
