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
torch::Tensor vdpdyns(const torch::Tensor& t, const torch::Tensor& y, 
                      const torch::Tensor& params) {
  torch::Tensor ydot = torch::zeros_like(y).to(y.device());
  ydot.index_put_({Slice(), 0}, y.index({Slice(), 1}));
  ydot.index_put_({Slice(), 1}, y.index({Slice(), 2})*(1.0 - y.index({Slice(), 0}).square()) * y.index({Slice(), 1}) - y.index({Slice(), 0}));
  return ydot.clone(); 
}

TensorDual vdpdynsdual(TensorDual& t, TensorDual& y) {
  TensorDual ydot = TensorDual::zeros_like(y);
  TensorDual y1 = y.index(0);  //Make sure the input is not modified
  TensorDual y2 = y.index(1);  //Make sure the input is not modified
  TensorDual y3 = y.index(2);  //Make sure the input is not modified
  ydot.index_put_(0, y2);
  ydot.index_put_(1, y3*(1 - y1 * y1) * y2 - y1);
  return ydot.clone(); //Return the through copy elision
}


torch::Tensor jac(const torch::Tensor& t, const torch::Tensor& y, 
                  const torch::Tensor& params) {
  int M = y.size(0);
  torch::Tensor jac = torch::zeros({y.size(1), y.size(1)}, torch::kFloat64).repeat({M, 1, 1}).to(y.device());
  jac.index_put_({Slice(), 0, 1}, 1.0);
  jac.index_put_({Slice(), 1, 0}, -2*y.index({Slice(), 2})*y.index({Slice(), 0})*y.index({Slice(), 1})-1);
  jac.index_put_({Slice(), 1, 1}, y.index({Slice(), 2})*(1-y.index({Slice(), 0}).square())); 
  jac.index_put_({Slice(), 1, 2}, (1.0-y.index({Slice(), 0}).square())*y.index({Slice(), 1}));
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





//Create a main method for testing

int main(int argc, char *argv[])
{


  //set the device
  //torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  torch::Device device(torch::kCPU);
  int M = 1;
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  torch::Tensor y = torch::zeros({M, 3}, torch::kF64).to(device);
  for (int i=0; i < M; i++) {
    y.index_put_({i, 0}, 2.0);
    y.index_put_({i, 2}, 1.0+i*0.001);
  }
  y.index_put_({Slice(), 1}, 0.0);
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  torch::Tensor tspan = torch::rand({M, 2}, torch::kFloat64).to(device);
  tspan.index_put_({Slice(), 0}, 0.0);
  //tspan.index_put_({Slice(), 1}, 1.5*((3.0-2.0*std::log(2.0))*y.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
  tspan.index_put_({Slice(), 1}, 10.0);
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  janus::OptionsTe options = janus::OptionsTe(); //Initialize with default options
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //*options.EventsFcn = vdpEvents;
  options.RelTol = torch::tensor({1e-13}, torch::kFloat64).to(device);
  options.AbsTol = torch::tensor({1e-16}, torch::kFloat64).to(device);
  //Create an instance of the Radau5 class
  torch::Tensor params = torch::empty({0}, torch::kFloat64).to(device);
  //Check for memory leaks
  for ( int i=0; i < 1000; i++) {
    std::cerr << "running iteration " << i << "for memory leaks" << std::endl;
    janus::RadauTe r(vdpdyns, jac, tspan, y, options, params);   
    r.solve();
  }
  janus::RadauTe r(vdpdyns, jac, tspan, y, options, params);   // Pass the correct arguments to the constructor
  //Call the solve method of the Radau5 class
  r.solve();
  //std::cout << "tout=";
  //janus::print_tensor(r.tout);;
  //std::cout << "yout=";
  //janus::print_tensor(r.yout);
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
      x1s[i][j] = r.yout.index({i, j, 0}).item<double>();
      x2s[i][j] = r.yout.index({i, j, 1}).item<double>();
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
  plt::close();

  return 0;
}
