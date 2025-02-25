#include "../../src/cpp/janus_ode_common.hpp"
#include "vdp_cvodes_solver.hpp"
#include "u2d_pmp_solver.hpp"


// Generate a main function as a controller
int main()
{
  // First generate data based on the vdp system
  torch::Tensor x0 = torch::tensor({2.0, 0.0});
  torch::Tensor mu = torch::tensor({100.0});
  torch::Tensor ft = torch::tensor({3.0 * mu.item<double>()});
  int Nt =10;
  auto res = u2d::vdp::gen_data(x0, mu, ft, Nt);
  //Extract the tensor
  auto t = std::get<0>(res);
  auto y = std::get<1>(res);
  // Now we have the data in t and y

  //For each segment we solve the PMP using KINSOL so the key is to generate the sensitivity equations
  //using the fS routine in CVODES and then feed it to the KINSOL solver to solve the BVP yielding l1 and l2
  //which effectively gives us a continuous time dynamics over this segment.
  //We then generate a data set for the state space dynamics over this segment
  
  // Then use the data to generate the dynamics using control theory
  // Finally use the dynamics generated by the control theory to learn the vdp system
  // This is a control centered method to do machine learning on the vdp system by generating the
  // dynamics as data
  // It can learn any 2D system
  // The control method is a simple neural network+open loop control
  // It uses an augmented version of the pontryagin's minimum principle to generate the dynamics
  // The dynamics are then used to learn the vdp system directly as opposed to Neural ODEs or PINNS
  // that embed the dynamics in the loss function.

  return 0;
}