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
  ydot.index_put_({Slice(), Slice(1,2)}, y3*(1 - y1 * y1) * y2 - y1);
  return ydot; //Return the through copy elision
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
    y.r.index_put_({i, 2}, 1000.0+0.001*i);
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
  //Run this multiple times to make sure there are no memory leaks
  for ( int i=0; i < 10000; i++)
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
  for ( int i=0; i < M; i++) {
    nouts[i] = r.nout.index({i}).item<int>();
    std::cout << "nouts[" << i << "]=" << nouts[i] << std::endl;
    x1s[i].resize(nouts[i]);
    x2s[i].resize(nouts[i]);
    for ( int j=0; j < nouts[i]; j++) {
      x1s[i][j] = r.yout.r.index({i, 0, j}).item<double>();
      x2s[i][j] = r.yout.r.index({i, 1, j}).item<double>();
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
