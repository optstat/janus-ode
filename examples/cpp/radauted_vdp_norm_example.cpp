#define HEADER_ONLY
#include "../../src/cpp/radauted.hpp"
#include <janus/tensordual.hpp>
#include <janus/janus_util.hpp>
#include "matplotlibcpp.h"

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
  //Here y3 is mu^2 not mu
  ydot.index_put_({Slice(), Slice(1,2)}, y3*((1 - y1 * y1) * y2 - y1));
  return ydot; //Return the through copy elision
}


TensorMatDual jac(const TensorDual& t, 
                   const TensorDual& y, 
                   const TensorDual& params) {
  auto jac = TensorMatDual(torch::zeros({y.r.size(0), y.r.size(1), y.r.size(1)}, torch::kFloat64), 
                           torch::zeros({y.r.size(0), y.r.size(1), y.r.size(1), y.d.size(2)}, torch::kFloat64));
  auto x1 = y.index({Slice(), 0});
  auto x2 = y.index({Slice(), 1});
  auto x3 = y.index({Slice(), 2});
  TensorDual one= TensorDual(torch::ones_like(x1.r), torch::zeros_like(x1.d));
  jac.index_put_({Slice(), 0, 1}, one);

  TensorDual zero = TensorDual(torch::zeros_like(x1.r), torch::zeros_like(x1.d));
  jac.index_put_({Slice(), 1, 0}, -x3*(2*x1*x2+1));
  jac.index_put_({Slice(), 1, 1}, x3*(1-x1*x1));
  jac.index_put_({Slice(), 1, 2}, (1-x1*x1)*x2);

  

  return jac;
}




 


  //% Initialisation of Dyn parameters
auto matD = TensorMatDual(torch::empty({M, D, D}, torch::kDouble).to(device), 
                          torch::empty({M, D, D, N}, torch::kDouble).to(device));
auto matOne = TensorMatDual(torch::empty({M, 1, 1}, torch::kDouble).to(device), 
                          torch::empty({M, 1, 1, N}, torch::kDouble).to(device));




//Create a main method for testing

int main(int argc, char *argv[])
{

  void (*pt)(const torch::Tensor&) = janus::print_tensor;
  void (*pd)(const TensorDual&) = janus::print_dual;
  void (*pmd)(const TensorMatDual&) = janus::print_dual;

  //set the device
  //torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);


  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual y = TensorDual(torch::zeros({M, D}, torch::kF64).to(device), torch::eye(D).repeat({M,1,1}).to(torch::kF64).to(device));
  for (int i=0; i < M; i++) {
    y.r.index_put_({i, 0}, 2.0+0.0001*i);
    y.r.index_put_({i, 2}, 1.0+0.001*i);
  }
  y.r.index_put_({Slice(), 1}, 0.0);
 
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual tspan = TensorDual(torch::rand({M, 2}, torch::kFloat64).to(device), torch::zeros({M,2,N}, torch::kFloat64).to(device));
  tspan.r.index_put_({Slice(), 0}, 0.0);
  //tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
  tspan.index_put_({Slice(), 1}, 10.0);
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  janus::OptionsTeD options = janus::OptionsTeD(); //Initialize with default options
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //*options.EventsFcn = vdpEvents;
  options.RelTol = torch::tensor({1e-13}, torch::kFloat64).to(device);
  options.AbsTol = torch::tensor({1e-16}, torch::kFloat64).to(device);
  //Create an instance of the Radau5 class
  TensorDual params = TensorDual(torch::empty({0,0}, torch::kFloat64).to(device), torch::zeros({M,2,N}, torch::kFloat64).to(device));
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
  for ( int i=0; i < M; i++) {
    nouts[i] = r.nout.index({i}).item<int>();
    x1s[i].resize(nouts[i]);
    x2s[i].resize(nouts[i]);
    for ( int j=0; j < nouts[i]; j++) {
      x1s[i][j] = r.yout.r.index({i, j, 0}).item<double>();
      x2s[i][j] = r.yout.r.index({i, j, 1}).item<double>();
    }
  }


  //Plot p1 vs p2
  plt::figure();
  for (int i=0; i < M; i++) {
    plt::plot(x1s[i], x2s[i]);
  }

  plt::xlabel("x1");
  plt::ylabel("x2");
  plt::title("x2 versus x1");
  plt::save("/tmp/x1x2.png");
  plt::close();  std::cout << "Final count=" << r.count << std::endl;
  std::cout << "yout real=";
  janus::print_tensor(r.yout.r);

  

  return 0;
}
