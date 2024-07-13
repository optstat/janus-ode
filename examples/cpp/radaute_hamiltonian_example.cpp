#include "../../src/cpp/radaute.hpp"
#include <janus/tensordual.hpp>
#include <janus/janus_util.hpp>
#include "../../src/cpp/janus_ode_common.hpp"
#include "matplotlibcpp.h"
using namespace janus;
namespace plt = matplotlibcpp;


/**
 * Radau example using the Van der Pol oscillator 
 * Using the Hamiltonian with dual number approach to calcuate the dynamics and
 * the Jacobian
*/
using Slice = torch::indexing::Slice;
double W = 0.01;

torch::Tensor control(const torch::Tensor& x1, 
                      const torch::Tensor& x2,
                      const torch::Tensor& p1,
                      const torch::Tensor& p2, 
                      double W=1.0) {
  auto u = -p2*((1-x1*x1)*x2)/W;
  return u; //Return the through copy elision
}

torch::Tensor hamiltonian(const torch::Tensor& x, const torch::Tensor& p, double W) {
  torch::Tensor p1 = p.index({Slice(), 0});  
  torch::Tensor p2 = p.index({Slice(), 1});  
  torch::Tensor x1 = x.index({Slice(), 0});  
  torch::Tensor x2 = x.index({Slice(), 1});  
  auto u = control(x1, x2, p1, p2, W);
  if ( (u < 0.0).any().item<bool>()) {
    u = u* 0.0+0.01;
  } 
  auto H = p1*x2+p2*(u*((1-x1*x1)*x2)-x1)+W*u*u/2; //Return the through copy elision
  return H; //Return the through copy elision
}

/**
 * Dynamics calculated according the hamiltonian method
 */
torch::Tensor vdpdyns_ham(const torch::Tensor& t, const torch::Tensor& y, const torch::Tensor& params) {
  return evalDyns<double>(y, W, hamiltonian);
}


torch::Tensor jac_ham(const torch::Tensor& t, const torch::Tensor& y, 
                  const torch::Tensor& params) {
  return evalJac<double>(y, W, hamiltonian);
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

    //Initialize the statistics
torch::Tensor  janus::StatsTe::FcnNbr=torch::empty({0}, torch::kInt64); // Add missing type specifier
torch::Tensor  janus::StatsTe::StepNbr = torch::empty({0}, torch::kInt64);
torch::Tensor  janus::StatsTe::JacNbr = torch::empty({0}, torch::kInt64);
torch::Tensor  janus::StatsTe::DecompNbr = torch::empty({0}, torch::kInt64);
torch::Tensor  janus::StatsTe::SolveNbr = torch::empty({0}, torch::kInt64);
torch::Tensor  janus::StatsTe::AccptNbr = torch::empty({0}, torch::kInt64);
torch::Tensor  janus::StatsTe::StepRejNbr = torch::empty({0}, torch::kInt64);
torch::Tensor  janus::StatsTe::NewtRejNbr = torch::empty({0}, torch::kInt64);
  //% Initialisation of Dyn parameters
torch::Tensor  janus::DynTe::Jac_t = torch::empty({0}, torch::kDouble);
torch::Tensor  janus::DynTe::Jac_Step = torch::empty({0}, torch::kDouble);
torch::Tensor  janus::DynTe::haccept_t = torch::empty({0}, torch::kDouble);
torch::Tensor  janus::DynTe::haccept_Step = torch::empty({0}, torch::kDouble);
torch::Tensor  janus::DynTe::haccept = torch::empty({0}, torch::kDouble);
torch::Tensor  janus::DynTe::hreject_t = torch::empty({0}, torch::kDouble);
torch::Tensor  janus::DynTe::hreject_Step = torch::empty({0}, torch::kDouble);
torch::Tensor  janus::DynTe::hreject = torch::empty({0}, torch::kDouble);
torch::Tensor  janus::DynTe::Newt_t = torch::empty({0}, torch::kDouble);
torch::Tensor  janus::DynTe::Newt_Step = torch::empty({0}, torch::kDouble);
torch::Tensor  janus::DynTe::NewtNbr = torch::empty({0}, torch::kInt64);
torch::Tensor  janus::DynTe::NbrStg_t = torch::empty({0}, torch::kInt64);
torch::Tensor  janus::DynTe::NbrStg_Step = torch::tensor({0}, torch::kInt64);
torch::Tensor  janus::DynTe::NbrStg = torch::tensor({0}, torch::kInt64);




//Create a main method for testing

int main(int argc, char *argv[])
{


  //set the device
  //torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  torch::Device device(torch::kCPU);
  int M = 1;
  int N = 2;
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  torch::Tensor y = torch::zeros({M, 2*N}, torch::kF64).to(device);
  for (int i=0; i < M; i++) {
    y.index_put_({i, 3}, 2.0+i*0.001);
  }
  y.index_put_({Slice(), 0}, 1.0); //p1
  y.index_put_({Slice(), 1}, 1.0); //p2
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  torch::Tensor tspan = torch::rand({M, 2}, torch::kFloat64).to(device);
  tspan.index_put_({Slice(), 0}, 0.0);
  //tspan.index_put_({Slice(), 1}, ((3.0-2.0*std::log(2.0))*y.index({Slice(), 2}) + 2.0*3.141592653589793/1000.0/3.0));
  tspan.index_put_({Slice(), 1}, 100.0);
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
  for ( int i=0; i < 100; i++) {
    janus::RadauTe r(vdpdyns_ham, jac_ham, tspan, y, options, params);   // Pass the correct
    r.solve();
  }
  janus::RadauTe r(vdpdyns_ham, jac_ham, tspan, y, options, params);   // Pass the correct arguments to the constructor
  //Call the solve method of the Radau5 class
  r.solve();
  std::cout << "tout=";
  janus::print_tensor(r.tout);;
  std::cout << "yout=";
  janus::print_tensor(r.yout);
  std::cout << "Number of points=" << r.nout << std::endl;
  std::cout << "Final count=" << r.count << std::endl;

  auto p1out = r.yout.index({0, Slice(), 0}).contiguous();
  std::cout << "p1out=";
  janus::print_tensor(p1out);
  auto p2out = r.yout.index({0, Slice(), 1}).contiguous();
  auto x1out = r.yout.index({0, Slice(), 2}).contiguous();
  auto x2out = r.yout.index({0, Slice(), 3}).contiguous();

  std::vector<double> p1(p1out.data_ptr<double>(), p1out.data_ptr<double>() + p1out.numel());
  std::cerr << "p1=";
  std::cerr << p1 << std::endl;
  std::vector<double> p2(p2out.data_ptr<double>(), p2out.data_ptr<double>() + p2out.numel());
  std::vector<double> x1(x1out.data_ptr<double>(), x1out.data_ptr<double>() + x1out.numel());
  std::vector<double> x2(x2out.data_ptr<double>(), x2out.data_ptr<double>() + x2out.numel());

  //Plot p1 vs p2
  plt::figure();
  plt::plot(p1, p2, "r-");
  plt::ylabel("p2");
  plt::xlabel("p1");
  plt::title("p2 versus p1");
  plt::save("/tmp/p1p2.png");
  plt::close();
  plt::figure();
  plt::plot(x1, x2, "r-");
  plt::xlabel("x1");
  plt::ylabel("x2");
  plt::title("x2 versus x1");
  plt::save("/tmp/x1x2.png");
  plt::close();

  return 0;
}
