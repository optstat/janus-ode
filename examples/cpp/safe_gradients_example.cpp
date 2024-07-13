#include <torch/torch.h>

// Define a generic function z = f(x, y)
torch::Tensor f(const torch::Tensor& x, const torch::Tensor& y) {
    // Example function; replace this with the actual function
    return x * y.pow(2); // This can be replaced with any function
}

/*
Assume the input tensors are 2D and that the first dimension is a batch dimension.
*/
torch::Tensor safe_jac(const torch::Tensor& y, const torch::Tensor& x) {
    //Assume the first dimension is a batch dimension
    int M = y.size(0);
    assert(x.size(0) == M);
    int Ny = y.size(1);
    int Nx = x.size(1);
    //Create a zero jacobian as the default
    auto jac = torch::zeros({M, Ny, Nx}, y.options());

    // Compute the gradient of y with respect to x
    if (y.requires_grad()) {
        jac = torch::autograd::grad({y}, {x}, {torch::ones_like(y)}, true, true, true)[0];
    }
    //This will safely return a zero tensor if y is not dependent on x
    //In this case y does not have the gradient attribute set to true
    return jac;
}


torch::Tensor unsafe_jac(const torch::Tensor& y, const torch::Tensor& x) {
    return torch::autograd::grad({y}, {x}, {torch::ones_like(y)}, true, true, true)[0];
}


int main() {
    // Define x and y with requires_grad=true to track gradients
    auto x = torch::tensor({{2.0}}, torch::requires_grad());
    auto y = torch::tensor({{3.0}});

    // Compute the function z = f(x, y)
    auto z = f(x, y);

    auto dz_dx = safe_jac(z, x);
    std::cout << "Safe dz_dx: " << dz_dx << std::endl;
    auto d2z_dx2 = safe_jac(dz_dx, x);
    std::cout << "Safe d2z_dx2: " << d2z_dx2 << std::endl;

    dz_dx = unsafe_jac(z, x);
    std::cout << "UnSafe dz_dx: " << dz_dx << std::endl;
    d2z_dx2 = unsafe_jac(dz_dx, x);
    std::cout << "UnSafe d2z_dx2: " << d2z_dx2 << std::endl;

    return 0;
}