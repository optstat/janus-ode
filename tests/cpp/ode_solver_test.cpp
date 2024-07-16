#include <gtest/gtest.h>
#include <torch/torch.h>
#include <random>
#include <janus/tensordual.hpp>
#include <janus/janus_util.hpp>
#include "../../src/cpp/janus_ode_common.hpp"
#include "../../src/cpp/radauted.hpp"


using TensorIndex = torch::indexing::TensorIndex;
using Slice = torch::indexing::Slice;
using namespace janus;


torch::Tensor control(const torch::Tensor& x1, 
                      const torch::Tensor& x2,
                      const torch::Tensor& p1,
                      const torch::Tensor& p2, 
                      double W=1.0) {
    auto u = -p2 * ((1 - x1 * x1) * x2 - x1) / W;
    return u; // Return through copy elision
}

torch::Tensor hamiltonian(const torch::Tensor& x, 
                          const torch::Tensor& p, 
                          double W) {
    torch::Tensor p1 = p.index({Slice(), 0});  
    torch::Tensor p2 = p.index({Slice(), 1});  
    torch::Tensor x1 = x.index({Slice(), 0});  
    torch::Tensor x2 = x.index({Slice(), 1});  
    torch::Tensor u = control(x1, x2, p1, p2, W);

    auto H = p1*p1*p1 * x2 + p2*p2*p2 * u * ((1 - x1 * x1) * x2 - x1) + W * u * u / 2 + 1;
    return H;
}








TEST(HamiltonianTest, pxHTest)
{
    torch::Tensor x = torch::rand({{2, 2}}, dtype(torch::kFloat64)); // example tensor
    torch::Tensor p = torch::rand({{2, 2}}, dtype(torch::kFloat64)); // example tensor
    double W = 1.0; // example parameter

    auto result = janus::pxH<double>(x, p, W, hamiltonian);
    //We need to verify the result using finite differences
    auto result_fd = torch::zeros_like(result);
    for (int i = 0; i < x.size(1); i++)
    {
        auto m = x.index({Slice(), i}) < 1.0;
        auto h = 1e-8*x.index({Slice(), i}).abs();
        h.index_put_({m}, 1e-8);
        auto xp = x.clone();
        xp.index_put_({Slice(), i}, x.index({Slice(), i}) + h);
        auto Hp = hamiltonian(xp, p, W);
        auto xm = x.clone();
        xm.index_put_({Slice(), i}, x.index({Slice(), i}) - h);
        auto Hm = hamiltonian(xm, p, W);
        result_fd.index_put_({Slice(), i}, (Hp - Hm) / (2 * h));
    }
    EXPECT_TRUE(torch::allclose(result, result_fd));
}

TEST(HamiltonianTest, ppHTest)
{
    torch::Tensor x = torch::rand({{2, 2}}, dtype(torch::kFloat64)); // example tensor
    torch::Tensor p = torch::rand({{2, 2}}, dtype(torch::kFloat64)); // example tensor
    double W = 1.0; // example parameter

    auto result = ppH<double>(x, p, W, hamiltonian);
    //We need to verify the result using finite differences
    auto result_fd = torch::zeros_like(result);
    for (int i = 0; i < p.size(1); i++)
    {
        auto m = p.index({Slice(), i}) < 1.0;
        auto h = 1e-8*p.index({Slice(), i}).abs();
        h.index_put_({m}, 1e-8);
        auto pp = p.clone();
        pp.index_put_({Slice(), i}, p.index({Slice(), i}) + h);
        auto Hp = hamiltonian(x, pp, W);
        auto pm = p.clone();
        pm.index_put_({Slice(), i}, p.index({Slice(), i}) - h);
        auto Hm = hamiltonian(x, pm, W);
        result_fd.index_put_({Slice(), i}, (Hp - Hm) / (2 * h));
    }
    EXPECT_TRUE(torch::allclose(result, result_fd));
}





TEST(HamiltonianTest, ppppppHTest)
{
    torch::Tensor x = torch::rand({2, 4}); // example tensor
    torch::Tensor p = torch::rand({2, 4}); // example tensor
    double W = 1.0; // example parameter


    auto result = ppppppH<double>(x, p, W, hamiltonian);
    

}

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

torch::Tensor H(const torch::Tensor &x, const torch::Tensor &p, double W)
{ 
  torch::Tensor p1 = p.index({Slice(), 0});
  torch::Tensor p2 = p.index({Slice(), 1});
  torch::Tensor x1 = x.index({Slice(), 0});
  torch::Tensor x2 = x.index({Slice(), 1});
        
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
TensorMatDual vdpjac(const TensorDual& y, 
                     double W)
{
  TensorDual p1 = y.index({Slice(), 0});
  TensorDual p2 = y.index({Slice(), 1});
  TensorDual x1 = y.index({Slice(), 2});
  TensorDual x2 = y.index({Slice(), 3});
  int M = y.d.size(0);
  int N = y.d.size(1);
  int D = y.d.size(2);

  TensorMatDual jac = TensorMatDual(torch::zeros({M,N,N}, dtype(torch::kFloat64)),
                                    torch::zeros({M,N,N,D}, dtype(torch::kFloat64))); 
  //the dynamics are rows and the state/costate are columns
  //-2*p2*p2*((1-x1*x1)*x2-x1)*(-2*x1*x2-1)/W)
  jac.index_put_({Slice(), 0, 1}, -4*p2*((1-x1*x1)*x2-x1)*(-2*x1*x2-1)/W);
  jac.index_put_({Slice(), 0, 2}, (4 * p2 * p2 * x2 * (-x1 + x2 * (1 - x1 * x1))) / W -
                                  (2 * p2 * p2 * (-2 * x1 * x2 - 1) * (-2 * x1 * x2 - 1)) / W);
  jac.index_put_({Slice(), 0, 3}, (4 * p2 * p2 * x1 * (-x1 + x2 * (1 - x1 * x1))) / W-
                                  (2 * p2 * p2 * (1 - x1 * x1) * (-2 * x1 * x2 - 1)) / W);
  //p1-2*p2*p2*((1-x1*x1)*x2-x1)*((1-x1*x1))/W
  jac.index_put_({Slice(), 1, 0}, 1.0);
  jac.index_put_({Slice(), 1, 1}, (-4 * p2 * (1 - x1 * x1) * (-x1 + x2 * (1 - x1 * x1))) / W);
  jac.index_put_({Slice(), 1, 2}, (4 * p2 * p2 * x1 * (-x1 + x2 * (1 - x1 * x1))) / W-
                                  (2 * p2 * p2 * (1 - x1 * x1) * (-2 * x1 * x2 - 1)) / W);
  jac.index_put_({Slice(), 1, 3},  (-2 * p2 * p2 * (1 - x1 * x1) * (1 - x1 * x1)) / W);

  //x2
  jac.index_put_({Slice(), 2, 3}, 1.0);

  //-2*p2*(x2*(1-x1*x1)-x1).pow(2)/W
  jac.index_put_({Slice(), 3, 1},  -2*(x2*(1-x1*x1)-x1).pow(2)/W);
  jac.index_put_({Slice(), 3, 2}, 4*p2*(x2*(1-x1*x1)-x1)*(2*x1*x2+1) / W);
  jac.index_put_({Slice(), 3, 3}, -4*p2*(x2*(1-x1*x1)-x1)*(1-x1*x1)/W);


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
    double W = 0.1;
    double h = 0.000001;





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
    //std::cerr << "y0ted=";
    //janus::print_dual(y0ted);
    //Update the state space using euler for one step
    auto yted =y0ted.clone();

    for ( int i=0; i < 1000; i++)
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
    auto ppHval = ppH<double>(x, p, W, H);  //dot{x}
    EXPECT_TRUE(torch::allclose(dydt.r.index({Slice(), Slice(2, 4)}), ppHval));
    auto pxHval = janus::pxH<double>(x, p, W, H); //dot{p}
    EXPECT_TRUE(torch::allclose(dydt.r.index({Slice(), Slice(0, 2)}), pxHval));
    //Now compare with the APIs
    //Check for memory leak
    std::cerr << "Checking for memory leaks in explicit dynamics calculation" << std::endl;
    for ( int i=0; i < 100; i++)
    {
        auto implDyns = evalDynsDual<double>(yted, W, H);
        //std::cerr << "implDyns=";
        //janus::print_dual(implDyns);
        //std::cerr << "dydt=";
        //janus::print_dual(dydt);
        EXPECT_TRUE(torch::allclose(dydt.r, implDyns.r));
        EXPECT_TRUE(torch::allclose(dydt.d, implDyns.d));    
    }
}

TEST(HamiltonianTest, JacExplVsImplTest) 
{


    int M = 1;
    int N = 2;
    double W = 0.1;
    double h = 0.000001;





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
    //Update the state space using euler for one step
    auto yted =y0ted.clone();

    for ( int i=0; i < 1000; i++)
    {
        yted = rk4(yted, W, h);
    }
    auto jacExpl = vdpjac(yted, W);//Calculate the dynamics at the new location
    std::cerr << "Checking for memory leaks in explicit jacobian calculation" << std::endl;
    for (int i=0; i < 100; i++)
    {
        auto jacImpl = evalJacDual<double>(yted, W, H);
    }
    auto jacImpl = evalJacDual<double>(yted, W, H);
    EXPECT_TRUE(torch::allclose(jacExpl.r, jacImpl.r));
    //std::cerr << "yted=";
    //janus::print_dual(yted);
    EXPECT_TRUE(torch::allclose(jacExpl.d, jacImpl.d));
    //std::cerr << "jacExpl.d=";
    //janus::print_tensor(jacExpl.d);
    //std::cerr << "jacImpl.d=";
    //janus::print_tensor(jacImpl.d);

}



TensorDual vdpdyns_vdp(const TensorDual& t, const TensorDual& y, const TensorDual& params) {
  TensorDual ydot = TensorDual::zeros_like(y);
  auto yc = y.clone();
  TensorDual y1 = yc.index({Slice(), Slice(0,1)});  //Make sure the input is not modified
  TensorDual y2 = yc.index({Slice(), Slice(1,2)});  //Make sure the input is not modified
  TensorDual y3 = yc.index({Slice(), Slice(2,3)});  //Make sure the input is not modified
  ydot.index_put_({Slice(), Slice(0,1)}, y2);
  ydot.index_put_({Slice(), Slice(1,2)}, y3*(1 - y1 * y1) * y2 - y1);
  return ydot; //Return the through copy elision
}



TensorMatDual jac_vdp(const TensorDual& t, const TensorDual& y, 
                  const TensorDual& params) {
  int M = y.r.size(0);
  int D = y.r.size(1);
  int N = y.d.size(2);
  
  TensorMatDual jac = TensorMatDual(torch::zeros({M,D,D}, torch::kFloat64).to(y.device()),
                              torch::zeros({M,D,D,N}, torch::kFloat64).to(y.device()));
  jac.index_put_({Slice(), 0, 1}, 1.0);
  auto yc = y.clone();
  auto j10 = -2*yc.index({Slice(), Slice(2,3)})*yc.index({Slice(), Slice(0,1)})*yc.index({Slice(), Slice(1,2)})-1.0;
  jac.index_put_({Slice(), Slice(1,2), Slice(0,1)}, j10.unsqueeze(2));
  auto j11 = yc.index({Slice(), Slice(2,3)})*(1.0-yc.index({Slice(), Slice(0,1)}).square());
  jac.index_put_({Slice(), Slice(1,2), Slice(1,2)}, j11.unsqueeze(2));
  auto j12 = (1.0-yc.index({Slice(), Slice(0,1)}).square())*yc.index({Slice(), Slice(1,2)});
  jac.index_put_({Slice(), Slice(1,2), Slice(2,3)}, j12.unsqueeze(2));
  return jac; //Return the through copy elision
}



/**
 * Make sure the sensitivities including the 
 * sensitivities wrt the end state are correct
 * Use finite differences in double precision to verify
 * One important consideration here is terminal time so 
 * the dual number has an extra parameter
 */
TEST(RadauTedTest, SensitivityTest) 
{


  //set the device
  //torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  int M=1;
  int D=3;
  int N=D+1; //Number dual variables

  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual y = TensorDual(torch::zeros({M, D}, torch::kF64), torch::zeros({M, D, N}, torch::kF64));
  for (int i=0; i < M; i++) {
    y.r.index_put_({i, 0}, 2.0+0.0001*i);
    y.r.index_put_({i, 2}, 1.0+0.001*i); //This is mu
    y.d.index_put_({i, 0, 0}, 1.0);
    y.d.index_put_({i, 1, 1}, 1.0);
    y.d.index_put_({i, 2, 2}, 1.0);
  }
  y.r.index_put_({Slice(), 1}, 0.0);
 
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual tspan = TensorDual(torch::rand({M, 2}, torch::kFloat64), torch::zeros({M,2,N}, torch::kFloat64));
  tspan.r.index_put_({Slice(), 0}, 0.0);
  //tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
  tspan.r.index_put_({Slice(), 1}, 0.1);
  tspan.d.index_put_({Slice(), 0, 3}, 1.0); //Final time sensitivity
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  janus::OptionsTeD options = janus::OptionsTeD(); //Initialize with default options
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //*options.EventsFcn = vdpEvents;
  options.RelTol = torch::tensor({1e-13}, torch::kFloat64);
  options.AbsTol = torch::tensor({1e-16}, torch::kFloat64);
  //Create an instance of the Radau5 class
  TensorDual params = TensorDual(torch::empty({0,0}, torch::kFloat64), torch::zeros({M,2,N}, torch::kFloat64));
  janus::RadauTeD r(vdpdyns_vdp, jac_vdp, tspan, y, options, params);   // Pass the correct arguments to the constructor
  //Call the solve method of the Radau5 class
  r.solve();

  std::cout << "tout=";
  janus::print_tensor(r.tout.r);
  std::cout << "Number of points=" << r.nout << std::endl;
  std::cout << "Number of points=" << r.nout << std::endl;
  std::cout << "Final count=" << r.count << std::endl;
  //Test the sensitivities using dual numbers

}



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
