#include "radaut.hpp"
#include "tensordual.hpp"


torch::Tensor vdpdyns(torch::Tensor& t, torch::Tensor& y, 
                      torch::Tensor& params) {
  torch::Tensor ydot = torch::zeros_like(y).to(y.device());
  ydot[0] = y[1];
  ydot[1] = y[2]*(1 - y[0] * y[0]) * y[1] - y[0];
  return ydot;
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


torch::Tensor jac(torch::Tensor& t, torch::Tensor& y, 
                      torch::Tensor& params) {
  torch::Tensor jac = torch::zeros({y.size(0), y.size(0)}, torch::kFloat64).to(y.device());
  jac[0][1] = 1.0;
  jac[1][0] = -2*y[2]*y[0]*y[1]-1;
  jac[1][1] = y[2]*(1-y[0]*y[0]);
  jac[1][2] = (1.0-y[0]*y[0])*y[1];
  return jac.clone(); //Return the through copy elision
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> vdpEvents(torch::Tensor& t, 
                                                                  torch::Tensor& y, 
                                                                  torch::Tensor& params) {
    //return empty tensors
    torch::Tensor E = y[1];
    torch::Tensor Stop = torch::tensor({false}, torch::TensorOptions().dtype(torch::kBool));
    if ((y[1] == 0.0).item<bool>()) {
        Stop = torch::tensor({true}, torch::TensorOptions().dtype(torch::kBool));
    }
    torch::Tensor Slope = torch::tensor({1}, torch::TensorOptions().dtype(torch::kFloat64));
    return std::make_tuple(t, E, Stop, Slope);
}


//Create a main method for testing

int main(int argc, char *argv[])
{
  //set the device
  //torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  torch::Device device(torch::kCPU);
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  torch::Tensor y = torch::zeros({3}, torch::kF64).to(device);
  y[0] = 2.0;
  y[1] = 0.0;
  y[2] = 1.0; 
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  torch::Tensor tspan = torch::rand({2}, torch::kF64).to(device);
  tspan[0] = 0.0;
  //T ~ (3 – 2 log 2) μ + 2π/μ1/3
  tspan[1] = 2*((3.0-2.0*std::log(2.0))*y[2] + 2.0*3.141592653589793/1000.0/3);
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  janus::OptionsT options = janus::OptionsT(); //Initialize with default options
  //Create a tensor of size 2x2 filled with random numbers from a uniform distribution on the interval [0,1)
  //*options.EventsFcn = vdpEvents;
  options.RelTol = torch::tensor({1e-13}, torch::kFloat64).to(device);
  options.AbsTol = torch::tensor({1e-16}, torch::kFloat64).to(device);
  //Create an instance of the Radau5 class
  torch::Tensor params = torch::empty({0}, torch::kF64).to(device);
  janus::RadauT r(vdpdyns, jac, tspan, y, options, params);   // Pass the correct arguments to the constructor
  //Call the solve method of the Radau5 class
  r.solve();
  std::cout << "nout=" << r.nout << std::endl;
  /*
      static int FcnNbr;
    static int JacNbr;
    static int DecompNbr;
    static int SolveNbr;
    static int StepNbr;
    static int AccptNbr;
    static int StepRejNbr;
    static int NewtRejNbr;*/
  std::cout << "FcnNbr=" << janus::StatsT::FcnNbr << std::endl;
  std::cout << "JacNbr=" << janus::StatsT::JacNbr << std::endl;
  std::cout << "DecompNbr=" << janus::StatsT::DecompNbr << std::endl;
  std::cout << "SolveNbr=" << janus::StatsT::SolveNbr << std::endl;
  std::cout << "StepNbr=" << janus::StatsT::StepNbr << std::endl;
  std::cout << "AccptNbr=" << janus::StatsT::AccptNbr << std::endl;
  std::cout << "StepRejNbr=" << janus::StatsT::StepRejNbr << std::endl;
  std::cout << "NewtRejNbr=" << janus::StatsT::NewtRejNbr << std::endl;
  std::cout << "t=";
  janus::print_vector(r.tout.index({Slice(0, r.nout)}));
  std::cout << "y=";
  janus::print_vector(r.y.index({Slice(0, 3*r.nout)}));

  return 0;
}
