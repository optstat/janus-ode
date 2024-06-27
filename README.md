# janus-ode
ODE solvers based on efficient dual tensors implemented to operate in a data parallel manner.
The goal of this project is to create solvers that can produce solutions in a massively parallel manner using data parallelism with the end goal of efficiently training deep neural networks for highly nonlinear problems.

This family of ODEs is slower on a per trajectory basis but outperform traditional ODE solvers when scaled to many thousands of instances.  The implementations presented here utilized janus-tensor-dual and janus-linear which are dual number based data parallel methods.

The ODE solvers presented here are Hamiltonian based-one supplies a hamiltonian method instead a dynamical equation.  The Hamiltonian may a deep neural network based or explicit.  The only requirement is that the Hamiltonian is differentiable with respect to its input.  In the case of optimal control an extra function needs to be provided to calculate the optimal control.  The dynamics and jacobians are automatically calculated efficiently using a combination of back propagation and dual numbers.
