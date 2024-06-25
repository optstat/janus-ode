# janus-ode
ODE solvers based on efficient dual tensors implemented to operate in a data parallel manner.
The goal of this project is to create solvers that can produce solutions in a massively parallel manner using data parallelism with the end goal of efficiently training deep neural networks for highly nonlinear problems.

This family of ODEs is slower on a per trajectory basis but outperform traditional ODE solvers when scaled to many thousands of instances.  The implementations presented here utilized janus-tensor-dual and janus-linear which are dual number based data parallel methods.
