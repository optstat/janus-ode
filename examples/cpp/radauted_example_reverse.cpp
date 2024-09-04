#define HEADER_ONLY
#include "../../src/cpp/radauted.hpp"
#include "../../src/cpp/radaute.hpp"
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

torch::Tensor vdpdyns_real(const torch::Tensor& t, 
                           const torch::Tensor& y, 
                           const torch::Tensor& params) {
  torch::Tensor ydot = torch::zeros_like(y);
  auto yc = y.clone();
  auto y1 = yc.index({Slice(), Slice(0,1)});  //Make sure the input is not modified
  auto y2 = yc.index({Slice(), Slice(1,2)});  //Make sure the input is not modified
  auto y3 = yc.index({Slice(), Slice(2,3)});  //Make sure the input is not modified
  ydot.index_put_({Slice(), Slice(0,1)}, y2);
  ydot.index_put_({Slice(), Slice(1,2)}, y3*(1 - y1 * y1) * y2 - y1);
  return ydot; //Return the through copy elision
}



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


torch::Tensor jac_real(const torch::Tensor& t, 
                       const torch::Tensor& y, 
                       const torch::Tensor& params) {
  int M = y.size(0);
  int D = y.size(1);
  
  torch::Tensor jac = torch::zeros({M,D,D}, torch::kFloat64).to(y.device());
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



TensorMatDual jac(const TensorDual& t, const TensorDual& y, 
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

  void (*pt)(const torch::Tensor&) = janus::print_tensor;
  void (*pd)(const TensorDual&) = janus::print_dual;
  void (*pmd)(const TensorMatDual&) = janus::print_dual;

  //set the device
  //torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);


  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual y = TensorDual(torch::zeros({M, D}, torch::kF64).to(device), torch::eye(D).repeat({M,1,1}).to(torch::kF64).to(device));
  for (int i=0; i < M; i++) {
    y.r.index_put_({i, 0}, 2.0+0.0001*i);
    y.r.index_put_({i, 2}, 100.0+0.001*i);
  }
  y.r.index_put_({Slice(), 1}, 0.0);
  auto y0 = y.clone();
  auto ft = 10.0;
 
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  TensorDual tspan = TensorDual(torch::rand({M, 2}, torch::kFloat64).to(device), torch::zeros({M,2,N}, torch::kFloat64).to(device));
  tspan.r.index_put_({Slice(), 0}, 0.0);
  //tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
  tspan.r.index_put_({Slice(), 1}, ft);
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  janus::OptionsTeD options = janus::OptionsTeD(); //Initialize with default options
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //*options.EventsFcn = vdpEvents;
  options.RelTol = torch::tensor({1e-13}, torch::kFloat64).to(device);
  options.AbsTol = torch::tensor({1e-16}, torch::kFloat64).to(device);
  //Create an instance of the Radau5 class
  std::cout << "Initial point before dual integrating forward" << std::endl;
  pd(y);
  TensorDual params = TensorDual(torch::empty({0,0}, torch::kFloat64).to(device), torch::zeros({M,2,N}, torch::kFloat64).to(device));
  janus::RadauTeD r(vdpdyns, jac, tspan, y, options, params);   // Pass the correct arguments to the constructor
  //Call the solve method of the Radau5 class
  r.solve();
  auto yf = r.y.clone();
  std::cout << "Final point after dual integrating forward" << std::endl;
  pd(yf);
  //Now compare with the regular radau integrator
  janus::OptionsTe options_real = janus::OptionsTe(); //Initialize with default options
  options_real.RelTol = torch::tensor({1e-13}, torch::kFloat64).to(device);
  options_real.AbsTol = torch::tensor({1e-16}, torch::kFloat64).to(device);
  auto yr = y0.r.clone();
  auto params_real = params.r.clone();
  auto tspan_real = tspan.r.clone();
  janus::RadauTe r_real(vdpdyns_real, jac_real, tspan_real, yr, options_real, params_real);   // Pass the correct arguments to the constructor
  //Call the solve method of the Radau5 class
  r_real.solve();
  std::cout << "Final point after integrating forward using real numbers" << std::endl;
  pt(r_real.y);
  std::cout << "Difference in forward integration between real and dual numbers=";
  pt(r_real.y - yf.r);
  auto yf_real = r_real.y.clone();
  
  
  
  
  //tspan.r.index_put_({Slice(), 1}, 2*((3.0-2.0*std::log(2.0))*y.r.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
  tspan.index_put_({Slice(), 0}, ft);
  tspan.index_put_({Slice(), 1}, 0.0);
  y = yf.clone();
  //zero out the sensitivities
  y.d.zero_();
  janus::RadauTeD rr(vdpdyns, jac, tspan, y, options, params);   // Pass the correct arguments to the constructor
  //Call the solve method of the Radau5 class
  rr.solve();
  std::cout << "Final point after integrating backward with dual numbers" << std::endl;
  pd(rr.y);
  std::cout << "Error in dual reverse forward integration=";
  pd(rr.y - y0);

  //Now calculate the reverse using the real numbers
  tspan_real.index_put_({Slice(), 0}, ft);
  tspan_real.index_put_({Slice(), 1}, 0.0);
  yr = yf_real.clone();
  janus::RadauTe rr_real(vdpdyns_real, jac_real, tspan_real, yr, options_real, params_real);   // Pass the correct arguments to the constructor
  std::cerr << "Real reverse results" << std::endl;
  rr_real.solve();
  std::cout << "Final point after integrating backward using real numbers" << std::endl;
  pt(rr_real.y);
  std::cerr << "Difference between dual and real reverse results" << std::endl;
  pt(rr.y.r - rr_real.y);

  std::cout << "Error in reverse integration using real numbers only=";
  pt(rr_real.y - y0.r);

    

  return 0;
}
