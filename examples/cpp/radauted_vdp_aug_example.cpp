#define HEADER_ONLY
#include "../../src/cpp/radauted.hpp"
#include <janus/tensordual.hpp>
#include <janus/janus_util.hpp>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

 
/**
 * Radau example using the Van der Pol oscillator 
 * using augmented dynamics 
 * for odes in batch mode
*/
using Slice = torch::indexing::Slice;
torch::Device device(torch::kCPU);
int M = 1;
int D = 4;
int N = 5;
double umin = 1.0;
double umax = 100.0;

TensorDual vdpdyns(const TensorDual& t, const TensorDual& y, const TensorDual& params) {
  TensorDual ydot = TensorDual::zeros_like(y);
  auto yc = y.clone();
  TensorDual p1 = yc.index({Slice(), Slice(0,1)});  
  TensorDual p2 = yc.index({Slice(), Slice(1,2)});  
  TensorDual x1 = yc.index({Slice(), Slice(2,3)});  
  TensorDual x2 = yc.index({Slice(), Slice(3,4)});
  auto m = p2*(1-x1*x1)*x2 < 0;
  auto ustar = TensorDual::zeros_like(p1);  
  ustar.r.index_put_({m}, umax);
  ustar.r.index_put_({~m}, umin);
  //p2 * u2star * (-2.0 * x1) * x2 - p2;
  ydot.index_put_({Slice(), Slice(0,1)}, 
                   p2 * ustar * (-2.0 * x1) * x2 - p2);
  //p1 + p2 * u2star * (1.0 - x1 * x1)
  ydot.index_put_({Slice(), Slice(1,2)}, 
                   p1 + p2 * ustar * (1.0 - x1 * x1));
  //x2
  ydot.index_put_({Slice(), Slice(2,3)}, x2);
  //u2star * (1.0 - x1 * x1) * x2 - x1
  ydot.index_put_({Slice(), Slice(3,4)}, 
                   ustar * (1.0 - x1 * x1) * x2 - x1);
  return ydot; //Return the through copy elision
}


TensorMatDual jac(const TensorDual& t, 
                   const TensorDual& y, 
                   const TensorDual& params) {
  auto jac = TensorMatDual(torch::zeros({y.r.size(0), 4, 4}, torch::kFloat64), 
                           torch::zeros({y.r.size(0), 4, 4, y.d.size(2)}, torch::kFloat64));
  auto p1 = y.index({Slice(), Slice(0,1)});
  auto p2 = y.index({Slice(), Slice(1,2)});
  auto x1 = y.index({Slice(), Slice(2,3)});
  auto x2 = y.index({Slice(), Slice(3,4)});
  TensorDual one= TensorDual::ones_like(p1);
  auto m = p2*(1-x1*x1)*x2 < 0;
  auto ustar = TensorDual::zeros_like(p1);  
  ustar.r.index_put_({m}, umax);
  ustar.r.index_put_({~m}, umin);
  //jac(1, 2) = u2star * (-2 * x1) * x2 - 1.0;
  jac.index_put_({Slice(), 0, 1}, ustar * (-2 * x1) * x2 - 1.0);
  //jac(1, 3) =  -p2 * u2star * 2 * x2;
  jac.index_put_({Slice(), 0, 2}, -p2 * ustar * 2 * x2);
  //jac(1, 4) =  -p2 * u2star * 2 * x1;
  jac.index_put_({Slice(), 0, 3}, -p2 * ustar * 2 * x1);

  //jac(2, 1) = 1.0;
  jac.index_put_({Slice(), 1, 0}, one);
  //jac(2, 2) = u2star * (1 - x1 * x1);
  jac.index_put_({Slice(), 1, 1}, ustar * (1 - x1 * x1));
  //jac(2, 3) = p2 * u2star * (-2 * x1);
  jac.index_put_({Slice(), 1, 2}, p2 * ustar * (-2 * x1));

  //jac(3, 4) = 1.0;
  jac.index_put_({Slice(), 2, 3}, one);

  //jac(4, 3) = u2star * (-2 * x1 * x2) - 1.0;
  jac.index_put_({Slice(), 3, 2}, ustar * (-2 * x1 * x2) - 1.0);

  //jac(3, 4) = u2star * ((1 - x1 * x1));
  jac.index_put_({Slice(), 3, 3}, ustar * (1 - x1 * x1));


  

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
  TensorDual y = TensorDual(torch::zeros({M, D}, torch::kF64).to(device), torch::zeros({M, D, N}, torch::kF64).to(device));
  y.r.index_put_({Slice(), 0}, -5); //p1
  y.r.index_put_({Slice(), 1}, -5); //p2
  y.r.index_put_({Slice(), 2}, 0); //x1
  y.r.index_put_({Slice(), 3}, 1.5); //x2
  y.d.index_put_({Slice(), 0, 0}, 1.0); //p1
  y.d.index_put_({Slice(), 1, 1}, 1.0); //p2
  y.d.index_put_({Slice(), 2, 2}, 1.0); //x1
  y.d.index_put_({Slice(), 3, 3}, 1.0); //x2
  
 
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual tspan = TensorDual(torch::rand({M, 2}, torch::kFloat64).to(device), torch::zeros({M,2,N}, torch::kFloat64).to(device));
  tspan.r.index_put_({Slice(), 0}, 0.0);
  //tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
  tspan.r.index_put_({Slice(), 1}, 10.0);
  tspan.d.index_put_({Slice(), 0, N-1}, 1.0);
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  janus::OptionsTeD options = janus::OptionsTeD(); //Initialize with default options
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //*options.EventsFcn = vdpEvents;
  options.RelTol = torch::tensor({1e-6}, torch::kFloat64).to(device);
  options.AbsTol = torch::tensor({1e-9}, torch::kFloat64).to(device);
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
  std::cout << "yout real=";
  janus::print_tensor(r.y.r);

  

  return 0;
}
