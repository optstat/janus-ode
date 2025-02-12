#define HEADER_ONLY
#include "../../src/cpp/radauted.hpp"
#include <janus/tensordual.hpp>
#include <janus/janus_util.hpp>
#include "matplotlibcpp.h"
#include <cmath>

namespace plt = matplotlibcpp;

 
/**
 * Radau example using the Van der Pol oscillator 
 * for odes in batch mode
*/
using Slice = torch::indexing::Slice;
torch::Device device(torch::kCPU);
int M = 1;
int D = 3;
int N = 3;


TensorDual vdpdyns(const TensorDual& t, const TensorDual& y, const TensorDual& params) {
  TensorDual ydot = TensorDual::zeros_like(y);
  auto yc = y.clone();
  TensorDual y1 = yc.index({Slice(), Slice(0,1)});  //Make sure the input is not modified
  TensorDual y2 = yc.index({Slice(), Slice(1,2)});  //Make sure the input is not modified
  TensorDual y3 = yc.index({Slice(), Slice(2,3)});  //Make sure the input is not modified
  ydot.index_put_({Slice(), Slice(0,1)}, y2);
  ydot.index_put_({Slice(), Slice(1,2)}, y3*(1 - y1 * y1) * y2 - y1);
  return ydot; //Return the through copy elision
}


torch::Tensor jac_r(const torch::Tensor&t,  //Has dimension [M,1]
                  const torch::Tensor& y, //Has dimension [M,D]
                  const torch::Tensor& params) {
  //Replicate y so that it produces a jacobian for each batch while retaining the computational graph
  //we want to produce a jacobian of size [M,D,D]
  auto jac  = torch::zeros({y.size(0), y.size(1), y.size(1)}, torch::kFloat64).to(y.device());
  jac.index_put_({Slice(), 0, 1}, 1.0);
  //There are only three real entries in the jacobian
  auto j10r = -2*y.index({Slice(), Slice(2,3)})*y.index({Slice(), Slice(0,1)})*y.index({Slice(), Slice(1,2)})-1.0;
  jac.index_put_({Slice(), Slice(1,2), Slice(0,1)}, j10r.unsqueeze(2));
  auto j11r = y.index({Slice(), Slice(2,3)})*(1.0-y.index({Slice(), Slice(0,1)}).square());
  jac.index_put_({Slice(), Slice(1,2), Slice(1,2)}, j11r.unsqueeze(2));
  auto j12r = (1.0-y.index({Slice(), Slice(0,1)}).square())*y.index({Slice(), Slice(1,2)});
  jac.index_put_({Slice(), Slice(1,2), Slice(2,3)}, j12r.unsqueeze(2));
  return jac; //Return the through copy elision

}



/**
 * Jacobian determination function
 */
TensorMatDual jac(const TensorDual& t, const TensorDual& y, 
                  const TensorDual& params) {
  int M = y.r.size(0);
  int D = y.r.size(1);
  int N = y.d.size(2);
  
  TensorMatDual jac = TensorMatDual(torch::zeros({M,D,D}, torch::kFloat64).to(y.device()),
                                    torch::zeros({M,D,D,N}, torch::kFloat64).to(y.device()));


    auto tr = t.r;
    auto yr = y.r;
    auto pr = params.r;
    jac.r = jac_r(t.r, y.r, params.r);   
    std::cerr << "jac.r=";
    janus::print_tensor(jac.r);
    //This is slow but is it safer
    for (int i=0; i < D; i++) {
      //Get the machine epsilon
      double epsilon = std::numeric_limits<double>::epsilon();
      auto h = sqrt(epsilon);
      auto yph = yr.clone();
      yph.index_put_({Slice(), i}, yph.index({Slice(), i}) + h);
      auto yp2h = yr.clone();
      yp2h.index_put_({Slice(), i}, yp2h.index({Slice(), i}) + 2.0*h);
      auto ymh = yr.clone();
      ymh.index_put_({Slice(), i}, ymh.index({Slice(), i}) - h);
      auto ym2h = yr.clone();
      ym2h.index_put_({Slice(), i}, ym2h.index({Slice(), i}) - 2.0*h);
      for ( int j=0; j < D; j++) {
        auto jacph = jac_r(t.r, yph, params.r);
        auto jacp2h = jac_r(t.r, yp2h, params.r);
        auto jacmh = jac_r(t.r, ymh, params.r);
        auto jacm2h = jac_r(t.r, ym2h, params.r);
        //Use a four point stencil to estimate the derivative
        auto dJ_dy = (-jacp2h + 8.0*jacph - 8.0*jacmh + jacm2h)/12.0/h;
        auto dJ_dp = torch::einsum("m,mk->mk", {dJ_dy.index({Slice(), i, j}), y.d.index({Slice(), i})});
        jac.d.index_put_({Slice(), i, j, Slice()}, dJ_dp);
      }
    }
  
  return jac; //Return the through copy elision
}




std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> vdpEvents(torch::Tensor& t, 
                                                                  torch::Tensor& y, 
                                                                  torch::Tensor& params) {
    int M = y.size(0);
    //return empty tensors
    torch::Tensor E = y.index({Slice(), 0});
    torch::Tensor Stop = torch::tensor({M, false}, torch::TensorOptions().dtype(torch::kBool));
    auto mask = (y.index({Slice(), 1}) == 0.0);
    Stop.index_put_({mask}, true);
    torch::Tensor Slope = torch::tensor({M, 1}, torch::TensorOptions().dtype(torch::kFloat64));
    return std::make_tuple(t, E, Stop, Slope);
}

  //% Initialisation of Dyn parameters
auto matD = TensorMatDual(torch::empty({M, D, D}, torch::kDouble).to(device), 
                          torch::empty({M, D, D, N}, torch::kDouble).to(device));
auto matOne = TensorMatDual(torch::empty({M, 1, 1}, torch::kDouble).to(device), 
                          torch::empty({M, 1, 1, N}, torch::kDouble).to(device));




//Create a main method for testing

int main(int argc, char *argv[])
{

  double mu = 10.0;

  void (*pt)(const torch::Tensor&) = janus::print_tensor;
  void (*pd)(const TensorDual&) = janus::print_dual;
  void (*pmd)(const TensorMatDual&) = janus::print_dual;

  //set the device
  //torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);


  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual y = TensorDual(torch::zeros({M, D}, torch::kF64).to(device), torch::eye(D).repeat({M,1,1}).to(torch::kF64).to(device));
  for (int i=0; i < M; i++) {
    y.r.index_put_({i, 0}, 2.0+0.0001*i);
    y.r.index_put_({i, 2}, mu+0.001*i);
  }
  y.r.index_put_({Slice(), 1}, 2.0);
 
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual tspan = TensorDual(torch::rand({M, 2}, torch::kFloat64).to(device), torch::zeros({M,2,N}, torch::kFloat64).to(device));
  tspan.r.index_put_({Slice(), 0}, 0.0);
  tspan.r.index_put_({Slice(), 1}, (3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}));
  //tspan.index_put_({Slice(), 1}, 10.0);
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  janus::OptionsTeD options = janus::OptionsTeD(); //Initialize with default options
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //*options.EventsFcn = vdpEvents;
  options.RelTol = torch::tensor({1e-8}, torch::kFloat64).to(device);
  options.AbsTol = torch::tensor({1e-8}, torch::kFloat64).to(device);
  //Create an instance of the Radau5 class
  TensorDual params = TensorDual(torch::empty({0,0}, torch::kFloat64).to(device), torch::zeros({M,2,N}, torch::kFloat64).to(device));
  //Run this multiple times to make sure there are no memory leaks
  for ( int i=0; i < 1; i++)
  {
    std::cerr << "Running iteration " << i << std::endl;
    janus::RadauTeD r(vdpdyns, jac, tspan, y, options, params);   // Pass the correct arguments to the constructor
    //Call the solve method of the Radau5 class
    r.solve();
  }
  janus::RadauTeD r(vdpdyns, jac, tspan, y, options, params);   // Pass the correct arguments to the constructor
  //Call the solve method of the Radau5 class
  r.solve();

  std::cout << "tout=";
  janus::print_tensor(r.tout.r);
  std::cout << "Number of points=" << r.nout << std::endl;
  std::cout << "Number of points=" << r.nout << std::endl;
  std::cout << "Final count=" << r.count << std::endl;
  std::vector<int> nouts(M);
  std::vector<std::vector<double>> x1s(M);
  std::vector<std::vector<double>> x2s(M);

  //Print out the dual of the final point
  std::cout << "Final y=";
  janus::print_dual(r.y);
  //Check the dual parts using finite differences
  for ( int j=0; j< 3; j++) {

    auto yp = TensorDual(torch::zeros({M, D}, torch::kF64).to(device), torch::eye(D).repeat({M,1,1}).to(torch::kF64).to(device));
    for (int i=0; i < M; i++) {
      yp.r.index_put_({i, 0}, 2.0+0.0001*i);
      yp.r.index_put_({i, 2}, mu+0.001*i);
    }
    yp.r.index_put_({Slice(), 1}, 2.0);
    auto hp = 1.0e-8;
    std::cout << "hp[" << j << "]=" << hp << std::endl;
    yp.r.index_put_({Slice(), j}, yp.r.index({Slice(), j}) + hp);
    //Integrate it forward
    janus::RadauTeD rp(vdpdyns, jac, tspan, yp, options, params);   // Pass the correct arguments to the constructor
    rp.solve();
    auto ym = TensorDual(torch::zeros({M, D}, torch::kF64).to(device), torch::eye(D).repeat({M,1,1}).to(torch::kF64).to(device));
    for (int i=0; i < M; i++) {
      ym.r.index_put_({i, 0}, 2.0+0.0001*i);
      ym.r.index_put_({i, 2}, mu+0.001*i);
    }
    ym.r.index_put_({Slice(), 1}, 2.0);
    ym.r.index_put_({Slice(), j}, ym.r.index({Slice(), j}) - hp);
    //Integrate it forward
    janus::RadauTeD rm(vdpdyns, jac, tspan, ym, options, params);   // Pass the correct arguments to the constructor
    rm.solve();
    //Compute the finite difference
    auto dy = (rp.y.r - rm.y.r)/(2.0*hp);
    std::cout << "dy[" << j << "]=";
    janus::print_tensor(dy.index({Slice(), j}));
    std::cout << "dyd[" << j << "]=";
    janus::print_tensor(r.y.d.index({Slice(), j}));
    
  }
  
  /*for ( int i=0; i < M; i++) {
    nouts[i] = r.nout.index({i}).item<int>();
    std::cout << "nouts[" << i << "]=" << nouts[i] << std::endl;
    x1s[i].resize(nouts[i]);
    x2s[i].resize(nouts[i]);
    for ( int j=0; j < nouts[i]; j++) {
      x1s[i][j] = r.yout.r.index({i, 0, j}).item<double>();
      x2s[i][j] = r.yout.r.index({i, 1, j}).item<double>();
    }
  }*/


  //Plot p1 vs p2
  /*plt::figure();
  for (int i=0; i < M; i++) {
    plt::plot(x1s[i], x2s[i]);
  }

  plt::xlabel("x1");
  plt::ylabel("x2");
  plt::title("x2 versus x1");
  plt::save("/tmp/x1x2.png");
  plt::close();  std::cout << "Final count=" << r.count << std::endl;*/
  //std::cout << "yout size=" << r.yout.r.sizes() << std::endl;
  //for ( int i=0; i < M; i++) {
  //  std::cout << "yout[" << i << "]=";
  //  for ( int j=0; j < nouts[i]; j++) {
  //    print_dual(r.yout.index({Slice(i, i+1), Slice(), Slice(j, j+1)}));
  //  }
 // }


  

  return 0;
}
