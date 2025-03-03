#include "../../src/cpp/janus_ode_common.hpp"
#include "petsc_vdp_jv_example.hpp"
#include "u2d_sens.hpp"


// Generate a main function as a controller
int main()
{
  // First generate data based on the vdp system
  torch::Tensor x0 = torch::tensor({2.0, 0.0});
  torch::Tensor mu = torch::tensor({100.0});
  torch::Tensor ft = torch::tensor({3.0 * mu.item<double>()});
  int Nt =1000;
  auto res = u2d::vdp::gen_data(x0, mu, ft, Nt);
  //Extract the tensor
  auto t = std::get<0>(res);
  auto y = std::get<1>(res);
  // Now we have the data in t and y
  //Extract just the first three points to test the sensitivity analysis
  auto t0in = t.index({0});
  std::cerr << "t0in=";
  janus::print_tensor(t0in);
  auto tfin = t.index({2});
  std::cerr << "tfin=";
  janus::print_tensor(tfin);
  //Exclude the first point since it will be used as the initial condition
  auto xdata = y.index({Slice(1,3), Slice()});
  std::cerr << "xdata=";
  janus::print_tensor(xdata);
  torch::Tensor y0in = torch::zeros({6});
  //First guess the initial augmented system
  y0in.index_put_({Slice(3,5)}, xdata.index({0, Slice()}));
  y0in.index_put_({-1}, 1.0); //The time dimension.  This will not change
  //Also exclude the first time since it is assumed to be zero
  u2d::vdpforpmp::for_sens(y0in, t.index({Slice(1,3)}), xdata);
  

  return 0;
}