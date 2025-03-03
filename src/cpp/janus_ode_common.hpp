#ifndef JANUS_ODE_COMMON
#define JANUS_ODE_COMMON
#include <torch/torch.h>
#include <janus/janus_util.hpp>
#include <functional>
#include <iostream>
#include <cvodes/cvodes.h> /* prototypes for CVODES fcts., consts. */
#include <math.h>
#include <nvector/nvector_serial.h> /* access to serial N_Vector            */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tuple> /* access to tuple */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <kinsol/kinsol.h>            // Main KINSOL header
#include <petscksp.h>  // For KSP, Mat, Vec, etc.

// Utility function to update tensor without in-place operations

//Common macros and constants for CVODES integration
#define Ith(v, i) NV_Ith_S(v, i - 1) /* i-th vector component i=1..NEQ */
#define IJth(A, i, j) \
  SM_ELEMENT_D(A, i - 1, j - 1) /* (i,j)-th matrix component i,j=1..NEQ */

/* Precision specific math function macros */

#if defined(SUNDIALS_DOUBLE_PRECISION)
#define ABS(x) (fabs((x)))
#elif defined(SUNDIALS_SINGLE_PRECISION)
#define ABS(x) (fabsf((x)))
#elif defined(SUNDIALS_EXTENDED_PRECISION)
#define ABS(x) (fabsl((x)))
#endif

/* Problem Constants */


#define ZERO SUN_RCONST(0.0)
#define ONE  SUN_RCONST(1.0)


namespace janus 
{

// Custom autograd function to log operations
struct TrackOpFunction : public torch::autograd::Function<TrackOpFunction> {
    static torch::Tensor forward(torch::autograd::AutogradContext* ctx, const torch::Tensor& input) {
        ctx->save_for_backward({input});
        return input;
    }

    static torch::autograd::variable_list backward(torch::autograd::AutogradContext* ctx, torch::autograd::variable_list grad_output) {
        auto saved = ctx->get_saved_variables();
        return {grad_output[0]};
    }
};

// Helper function to apply the custom autograd function
torch::Tensor track_op(const torch::Tensor& x) {
    return TrackOpFunction::apply(x);
}

std::string removeWhitespace(std::string str)
{
  str.erase(std::remove_if(str.begin(), str.end(), ::isspace), str.end());
  return str;
}



bool are_tensors_connected(const torch::Tensor& x, const torch::Tensor& y) {
    auto grad_fn = y.grad_fn();
    std::unordered_set<void*> seen_nodes;

    while (grad_fn) {
        seen_nodes.insert(grad_fn.get());
        if (grad_fn->next_edges().empty()) {
            break;
        }
        grad_fn = grad_fn->next_edges()[0].function;
    }

    grad_fn = x.grad_fn();
    while (grad_fn) {
        if (seen_nodes.find(grad_fn.get()) != seen_nodes.end()) {
            return true;
        }
        if (grad_fn->next_edges().empty()) {
            break;
        }
        grad_fn = grad_fn->next_edges()[0].function;
    }

    return false;
}


void resetTensor(torch::Tensor& tensor) {
    tensor = torch::Tensor(); // Assign an empty tensor to deallocate the original tensor
}
void resetGradients(torch::Tensor& tensor) {
    tensor.mutable_grad().reset();
}


void safe_detach(torch::Tensor& tensor) {
    if (tensor.requires_grad()) {
        tensor.detach_();
    }
}

/*
Assume the input tensors are 2D and that the first dimension is a batch dimension.
*/
torch::Tensor safe_jac(const torch::Tensor& y, const torch::Tensor& x) {
    //Assume y is a scalar and x is 2D
    int M = y.size(0);
    int Nx = x.size(1);
    //Create a zero jacobian as the default
    auto jac = torch::zeros({M, 1, Nx}, y.options());
    
    // Compute the gradient of y with respect to x
    if (y.requires_grad()) {
        jac = torch::autograd::grad({y}, {x}, {torch::ones_like(y)}, true, true, true)[0];
    }
    else {
        y.set_requires_grad(true);
        jac = torch::autograd::grad({y}, {x}, {torch::ones_like(y)}, true, true, true)[0];
    }
    //This will safely return a zero tensor if y is not dependent on x
    //In this case y does not have the gradient attribute set to true
    return jac;
}





torch::Tensor computeGradients(const torch::Tensor& x, const torch::Tensor& p, double W, std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, double)> H) {
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the gradients using torch::autograd::grad
    auto gradients = torch::autograd::grad({Hvalue}, 
                                           {xt, pt}, 
                                           /*grad_outputs=*/{}, 
                                           /*retain_graph=*/true, 
                                           /*create_graph=*/true);

    auto grad_xt = gradients[0];
    auto grad_pt = gradients[1];

    // Optionally, print the gradients for debugging
    std::cout << "Gradient wrt xt: " << grad_xt << std::endl;
    std::cout << "Gradient wrt pt: " << grad_pt << std::endl;

    // Perform further operations if needed

    return grad_xt;  // or any other gradient tensor needed
}

torch::Tensor ensure_grad(const torch::Tensor& tensor) {
    if (!tensor.requires_grad()) {
        std::cerr << "Tensor does not require gradients. Cloning and setting requires_grad=true." << std::endl;
        return tensor.clone().set_requires_grad(true);
    }
    return tensor;
}



template<typename T>
torch::Tensor pxH(const torch::Tensor &x, 
                  const torch::Tensor &p, 
                  T W, 
                  std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, T)> H)
{
    // Create tensors with gradient tracking for x and no gradient tracking for p
    auto xt = x.clone().detach();
    xt=xt.set_requires_grad(true);
    auto pt = p.clone().detach();
    pt=pt.set_requires_grad(false);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the gradient of Hvalue with respect to xt
    //auto grad_H_wrt_x = torch::autograd::grad({Hvalue}, {xt}, {torch::ones_like(Hvalue)})[0];
    auto grad_H_wrt_x = safe_jac(Hvalue, xt);
    auto xtc = xt.detach();
    auto ptc = pt.detach();
    auto Hvaluec = Hvalue.detach();
    auto grad_H_wrt_xc = grad_H_wrt_x.detach();

    return grad_H_wrt_xc;
}


template<typename T>
torch::Tensor pxHu(const torch::Tensor &x, 
                   const torch::Tensor &p,
                   const torch::Tensor &u, 
                   T W, 
                   std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&,const torch::Tensor&,  T)> Hu)
{
    // Create tensors with gradient tracking for x and no gradient tracking for p
    auto xt = x.clone().detach();
    xt=xt.set_requires_grad(true);
    auto pt = p.clone().detach();
    pt=pt.set_requires_grad(false);

    // Compute the Hamiltonian value
    auto Hvalue = Hu(xt, pt, u, W);

    // Compute the gradient of Hvalue with respect to xt
    //auto grad_H_wrt_x = torch::autograd::grad({Hvalue}, {xt}, {torch::ones_like(Hvalue)})[0];
    auto grad_H_wrt_x = safe_jac(Hvalue, xt);
    auto xtc = xt.detach();
    auto ptc = pt.detach();
    auto Hvaluec = Hvalue.detach();
    auto grad_H_wrt_xc = grad_H_wrt_x.detach();

    return grad_H_wrt_xc;
}

template<typename T>
torch::Tensor ppH(const torch::Tensor &x, 
                  const torch::Tensor &p, 
                  T W,
                  std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, T)> H)
{
    // Create tensors with gradient tracking for p and no gradient tracking for x
    auto xt = x.clone().detach();
    xt=xt.set_requires_grad(false);
    auto pt = p.clone().detach();
    pt=pt.set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the gradient of Hvalue with respect to pt
    auto grad_H_wrt_p = torch::autograd::grad({Hvalue}, {pt}, {torch::ones_like(Hvalue)})[0];
    auto ptc = pt.detach();
    auto Hvaluec = Hvalue.detach();
    auto grad_H_wrt_pc = grad_H_wrt_p.detach();
    return grad_H_wrt_pc;
}

template<typename T>
torch::Tensor ppHu(const torch::Tensor &x, 
                   const torch::Tensor &p,
                   const torch::Tensor &u, 
                   T W,
                   std::function<torch::Tensor(const torch::Tensor&, 
                                               const torch::Tensor&,
                                               const torch::Tensor&, 
                                               T)> Hu)
{
    // Create tensors with gradient tracking for p and no gradient tracking for x
    auto xt = x.clone().detach();
    xt=xt.set_requires_grad(false);
    auto pt = p.clone().detach();
    pt=pt.set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = Hu(xt, pt, u, W);

    // Compute the gradient of Hvalue with respect to pt
    auto grad_H_wrt_p = torch::autograd::grad({Hvalue}, {pt}, {torch::ones_like(Hvalue)})[0];
    auto ptc = pt.detach();
    auto Hvaluec = Hvalue.detach();
    auto grad_H_wrt_pc = grad_H_wrt_p.detach();
    return grad_H_wrt_pc;
}


template<typename T>
torch::Tensor ppppH(const torch::Tensor &x, 
                    const torch::Tensor &p, 
                    T W,
                    std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, T)> H)
{
    // Create tensors with gradient tracking for x and no gradient tracking for p
    auto xt = x.clone().detach();
    xt=xt.set_requires_grad(false);
    auto pt = p.clone().detach();
    pt=pt.set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the first-order gradient with respect to xt
    auto first_order_grad = torch::autograd::grad({Hvalue},
                                                  {pt}, 
                                                  {torch::ones_like(Hvalue)}, 
                                                  true, 
                                                  true)[0];
    // Initialize a tensor to hold the second-order gradients (Hessian)
    auto batch_size = x.size(0);
    auto x_dim = x.size(1);
    torch::Tensor hessian = torch::zeros({batch_size, x_dim, x_dim}, x.options());

    // Compute the second-order gradients for each dimension
    for (int i = 0; i < x_dim; ++i)
    {

        // Zero out the gradients of pt before computing the gradient for the next dimension
        if (pt.grad().defined()) {
            pt.grad().zero_();
        }

        // Compute the gradient of the i-th component of the first-order gradient with respect to xt
        //This is the row of the original vector 
        //which is then used to populate the row of the hessian
        //\partial x_j (\partial_{x_i} H )
        /*auto grad_p_i = torch::autograd::grad({first_order_grad.index({Slice(), i})}, 
                                              {pt}, 
                                              {torch::ones_like(first_order_grad.index({Slice(), i}))} 
                                              )[0];*/
        auto grad_p_i = safe_jac(first_order_grad.index({Slice(), i}), pt);
        if (grad_p_i.requires_grad()) 
        {
          grad_p_i.detach();
          // Assign the gradient to the corresponding row of the Hessian matrix
          hessian.index_put_({Slice(), i, Slice()}, grad_p_i);
        }
    }
    auto ptc      = pt.detach();
    auto Hvaluec  = Hvalue.detach();
    auto hessianc = hessian.detach();

    // Return the Hessian
    return hessianc;
}


template<typename T>
torch::Tensor ppppHu(const torch::Tensor &x, 
                     const torch::Tensor &p,
                     const torch::Tensor &u, 
                     T W,
                     std::function<torch::Tensor(const torch::Tensor&, 
                                                 const torch::Tensor&,
                                                 const torch::Tensor&, 
                                                 T)> Hu)
{
    // Create tensors with gradient tracking for x and no gradient tracking for p
    auto xt = x.clone().detach();
    xt=xt.set_requires_grad(false);
    auto pt = p.clone().detach();
    pt=pt.set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = Hu(xt, pt, u, W);

    // Compute the first-order gradient with respect to xt
    auto first_order_grad = torch::autograd::grad({Hvalue},
                                                  {pt}, 
                                                  {torch::ones_like(Hvalue)}, 
                                                  true, 
                                                  true)[0];
    // Initialize a tensor to hold the second-order gradients (Hessian)
    auto batch_size = x.size(0);
    auto x_dim = x.size(1);
    torch::Tensor hessian = torch::zeros({batch_size, x_dim, x_dim}, x.options());

    // Compute the second-order gradients for each dimension
    for (int i = 0; i < x_dim; ++i)
    {

        // Zero out the gradients of pt before computing the gradient for the next dimension
        if (pt.grad().defined()) {
            pt.grad().zero_();
        }

        // Compute the gradient of the i-th component of the first-order gradient with respect to xt
        //This is the row of the original vector 
        //which is then used to populate the row of the hessian
        //\partial x_j (\partial_{x_i} H )
        /*auto grad_p_i = torch::autograd::grad({first_order_grad.index({Slice(), i})}, 
                                              {pt}, 
                                              {torch::ones_like(first_order_grad.index({Slice(), i}))} 
                                              )[0];*/
        auto grad_p_i = safe_jac(first_order_grad.index({Slice(), i}), pt);
        
       
        // Assign the gradient to the corresponding row of the Hessian matrix
        if (grad_p_i.requires_grad()) 
        {
            grad_p_i.detach();
            hessian.index_put_({Slice(), i, Slice()}, grad_p_i);
        }
    }
    auto ptc      = pt.detach();
    auto Hvaluec  = Hvalue.detach();
    auto hessianc = hessian.detach();

    // Return the Hessian
    return hessianc;
}





template<typename T>
torch::Tensor pxpxH(const torch::Tensor &x, 
                    const torch::Tensor &p, 
                    T W,
                    std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, T)> H)
{
    // Create tensors with gradient tracking for x and no gradient tracking for p
    auto xt = x.clone().detach();
    xt=xt.set_requires_grad(true);
    auto pt = p.clone().detach();
    pt=pt.set_requires_grad(false);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the first-order gradient with respect to xt
    auto first_order_grad = torch::autograd::grad({Hvalue}, 
                                                  {xt}, 
                                                  {torch::ones_like(Hvalue)}, 
                                                  true, 
                                                  true)[0];
    auto batch_size = x.size(0);
    auto x_dim = x.size(1);
    torch::Tensor hessian = torch::zeros({batch_size, x_dim, x_dim}, x.options());
    // Compute the second-order gradients for each dimension
    for (int i = 0; i < x_dim; ++i)
    {
        if ( xt.grad().defined()) {
            xt.grad().zero_();
        }
        

        // Compute the gradient of the i-th component of the first-order gradient with respect to xt
        //This is the row of the original vector 
        //which is then used to populate the row of the hessian
        //\partial x_j (\partial_{x_i} H )
        /*auto grad_x_i = torch::autograd::grad({first_order_grad.index({Slice(), i})}, 
                                              {xt}, 
                                              {torch::ones_like(first_order_grad.index({Slice(), i}))}
                                              )[0];*/
        auto grad_x_i = safe_jac(first_order_grad.index({Slice(), i}), xt);

        // Assign the gradient to the corresponding column of the Hessian matrix
        if ( grad_x_i.requires_grad()) 
        {
            grad_x_i.detach();
            hessian.index_put_({Slice(), i}, grad_x_i);
        }
    }
    auto xtc = xt.detach();
    auto Hvaluec = Hvalue.detach();
    auto hessianc = hessian.detach();
    // Return the Hessian
    return hessianc;
}


template<typename T>
torch::Tensor pxpxHu(const torch::Tensor &x, 
                     const torch::Tensor &p,
                     const torch::Tensor &u, 
                     T W,
                     std::function<torch::Tensor(const torch::Tensor&, 
                                                 const torch::Tensor&,
                                                 const torch::Tensor&, 
                                                 T)> Hu)
{
    // Create tensors with gradient tracking for x and no gradient tracking for p
    auto xt = x.clone().detach();
    xt=xt.set_requires_grad(true);
    auto pt = p.clone().detach();
    pt=pt.set_requires_grad(false);

    // Compute the Hamiltonian value
    auto Hvalue = Hu(xt, pt, u, W);

    // Compute the first-order gradient with respect to xt
    auto first_order_grad = torch::autograd::grad({Hvalue}, 
                                                  {xt}, 
                                                  {torch::ones_like(Hvalue)}, 
                                                  true, 
                                                  true)[0];
    auto batch_size = x.size(0);
    auto x_dim = x.size(1);
    torch::Tensor hessian = torch::zeros({batch_size, x_dim, x_dim}, x.options());
    // Compute the second-order gradients for each dimension
    for (int i = 0; i < x_dim; ++i)
    {
        if ( xt.grad().defined()) {
            xt.grad().zero_();
        }
        

        // Compute the gradient of the i-th component of the first-order gradient with respect to xt
        //This is the row of the original vector 
        //which is then used to populate the row of the hessian
        //\partial x_j (\partial_{x_i} H )
        /*auto grad_x_i = torch::autograd::grad({first_order_grad.index({Slice(), i})}, 
                                              {xt}, 
                                              {torch::ones_like(first_order_grad.index({Slice(), i}))}
                                              )[0];*/
        auto grad_x_i = safe_jac(first_order_grad.index({Slice(), i}), xt);

        // Assign the gradient to the corresponding column of the Hessian matrix
        if ( grad_x_i.requires_grad()) 
        {
            grad_x_i.detach();
            hessian.index_put_({Slice(), i}, grad_x_i);
        }
    }
    auto xtc = xt.detach();
    auto Hvaluec = Hvalue.detach();
    auto hessianc = hessian.detach();
    // Return the Hessian
    return hessianc;
}



template<typename T>
torch::Tensor pxppH(const torch::Tensor &x, 
                    const torch::Tensor &p, 
                    T W,
                    std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, T)> H)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach();
    xt=xt.set_requires_grad(true);
    auto pt = p.clone().detach();
    pt=pt.set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the gradient of H with respect to x
    auto grad_H_p = torch::autograd::grad({Hvalue}, 
                                          {pt}, 
                                          {torch::ones_like(Hvalue)}, 
                                          true, 
                                          true)[0];
    // Initialize a tensor to hold the mixed second-order gradients
    auto batch_size = x.size(0);
    auto x_dim = x.size(1);
    auto p_dim = p.size(1);
    torch::Tensor mixed_hessian = torch::zeros({batch_size, p_dim, x_dim}, x.options());
    // Compute the second-order gradients for each dimension of x and p
    for (int i = 0; i < p_dim; ++i)
    {
        // Zero out the gradients of xt and pt before computing the gradient for the next dimension
        if (xt.grad().defined()) {
            xt.grad().zero_();
        }
        if ( pt.grad().defined()) {
            pt.grad().zero_();
        }


        // Compute the gradient of the i-th component of grad_H_x with respect to xt
        /*auto grad_H_p_i = torch::autograd::grad({grad_H_p.index({Slice(), i})}, 
                                                {xt}, 
                                                {torch::ones_like(grad_H_p.index({Slice(), i}))}, 
                                                true, 
                                                true)[0];*/
        auto grad_H_p_i = safe_jac(grad_H_p.index({Slice(), i}), xt);
        // Assign the gradient to the corresponding slice of the mixed Hessian matrix
        //mixed_hessian.select(1, i).copy_(grad_H_p_i);
        if ( grad_H_p_i.requires_grad()) 
        {
          grad_H_p_i.detach();
          mixed_hessian.index_put_({Slice(), i}, grad_H_p_i);
        }
    }
    auto xtc = xt.detach();
    auto ptc = pt.detach();
    auto Hvaluec = Hvalue.detach();
    auto mixed_hessianc = mixed_hessian.detach();
    // Return the mixed Hessian
    return mixed_hessianc;
}


template<typename T>
torch::Tensor pxppHu(const torch::Tensor &x, 
                     const torch::Tensor &p,
                     const torch::Tensor &u, 
                     T W,
                     std::function<torch::Tensor(const torch::Tensor&, 
                                                 const torch::Tensor&,
                                                 const torch::Tensor&, 
                                                 T)> Hu)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach();
    xt=xt.set_requires_grad(true);
    auto pt = p.clone().detach();
    pt=pt.set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = Hu(xt, pt, u, W);

    // Compute the gradient of H with respect to x
    auto grad_H_p = torch::autograd::grad({Hvalue}, 
                                          {pt}, 
                                          {torch::ones_like(Hvalue)}, 
                                          true, 
                                          true)[0];
    // Initialize a tensor to hold the mixed second-order gradients
    auto batch_size = x.size(0);
    auto x_dim = x.size(1);
    auto p_dim = p.size(1);
    torch::Tensor mixed_hessian = torch::zeros({batch_size, p_dim, x_dim}, x.options());
    // Compute the second-order gradients for each dimension of x and p
    for (int i = 0; i < p_dim; ++i)
    {
        // Zero out the gradients of xt and pt before computing the gradient for the next dimension
        if (xt.grad().defined()) {
            xt.grad().zero_();
        }
        if ( pt.grad().defined()) {
            pt.grad().zero_();
        }


        // Compute the gradient of the i-th component of grad_H_x with respect to xt
        /*auto grad_H_p_i = torch::autograd::grad({grad_H_p.index({Slice(), i})}, 
                                                {xt}, 
                                                {torch::ones_like(grad_H_p.index({Slice(), i}))}, 
                                                true, 
                                                true)[0];*/
        auto grad_H_p_i = safe_jac(grad_H_p.index({Slice(), i}), xt);
        // Assign the gradient to the corresponding slice of the mixed Hessian matrix
        //mixed_hessian.select(1, i).copy_(grad_H_p_i);
        if ( grad_H_p_i.requires_grad()) 
        {
          grad_H_p_i.detach();
          mixed_hessian.index_put_({Slice(), i}, grad_H_p_i);
        }
    }
    auto xtc = xt.detach();
    auto ptc = pt.detach();
    auto Hvaluec = Hvalue.detach();
    auto mixed_hessianc = mixed_hessian.detach();
    // Return the mixed Hessian
    return mixed_hessianc;
}




template<typename T>
torch::Tensor pppxH(const torch::Tensor &x, 
                    const torch::Tensor &p, 
                    T W,
                    std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, T)> H)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach();
    xt=xt.set_requires_grad(true);
    auto pt = p.clone().detach();
    pt=pt.set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the gradient of H with respect to x
    auto grad_H_x = torch::autograd::grad({Hvalue}, 
                                          {xt}, 
                                          {torch::ones_like(Hvalue)}, 
                                          true, 
                                          true)[0];
    // Initialize a tensor to hold the mixed second-order gradients
    auto batch_size = x.size(0);
    auto x_dim = x.size(1);
    auto p_dim = p.size(1);
    torch::Tensor mixed_hessian = torch::zeros({batch_size, x_dim, p_dim}, x.options());
    // Zero out the gradients of pt

    // Compute the second-order gradients for each dimension of x and p
    for (int i = 0; i < x_dim; ++i)
    {
        if ( pt.grad().defined()) {
            pt.grad().zero_();
        }
        if ( xt.grad().defined()) {
            xt.grad().zero_();
        }


        // Compute the gradient of the i-th component of grad_H_x with respect to pt
        /*auto grad_H_x_i = torch::autograd::grad({grad_H_x.index({Slice(), i})}, 
                                                {pt}, 
                                                {torch::ones_like(grad_H_x.index({Slice(), i}))})[0];*/
        auto grad_H_x_i = safe_jac(grad_H_x.index({Slice(), i}), pt);
        // Assign the gradient to the corresponding slice of the mixed Hessian matrix
        //mixed_hessian.select(1, i).copy_(grad_H_x_i);
        if ( grad_H_x_i.requires_grad()) 
        {
          grad_H_x_i.detach();
          mixed_hessian.index_put_({Slice(), i}, grad_H_x_i);
        }
    }
    auto ptc = pt.detach();
    auto xtc = xt.detach();
    auto Hvaluec = Hvalue.detach();
    auto mixed_hessianc = mixed_hessian.detach();
    // Return the mixed Hessian
    return mixed_hessianc;
}


template<typename T>
torch::Tensor pppxHu(const torch::Tensor &x, 
                     const torch::Tensor &p,
                     const torch::Tensor &u, 
                     T W,
                     std::function<torch::Tensor(const torch::Tensor&, 
                                                 const torch::Tensor&,
                                                 const torch::Tensor&, 
                                                 T)> Hu)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach();
    xt=xt.set_requires_grad(true);
    auto pt = p.clone().detach();
    pt=pt.set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = Hu(xt, pt, u, W);

    // Compute the gradient of H with respect to x
    auto grad_H_x = torch::autograd::grad({Hvalue}, 
                                          {xt}, 
                                          {torch::ones_like(Hvalue)}, 
                                          true, 
                                          true)[0];
    // Initialize a tensor to hold the mixed second-order gradients
    auto batch_size = x.size(0);
    auto x_dim = x.size(1);
    auto p_dim = p.size(1);
    torch::Tensor mixed_hessian = torch::zeros({batch_size, x_dim, p_dim}, x.options());
    // Zero out the gradients of pt

    // Compute the second-order gradients for each dimension of x and p
    for (int i = 0; i < x_dim; ++i)
    {
        if ( pt.grad().defined()) {
            pt.grad().zero_();
        }
        if ( xt.grad().defined()) {
            xt.grad().zero_();
        }


        // Compute the gradient of the i-th component of grad_H_x with respect to pt
        /*auto grad_H_x_i = torch::autograd::grad({grad_H_x.index({Slice(), i})}, 
                                                {pt}, 
                                                {torch::ones_like(grad_H_x.index({Slice(), i}))})[0];*/
        auto grad_H_x_i = safe_jac(grad_H_x.index({Slice(), i}), pt);
        // Assign the gradient to the corresponding slice of the mixed Hessian matrix
        //mixed_hessian.select(1, i).copy_(grad_H_x_i);
        if ( grad_H_x_i.requires_grad()) 
        {
          grad_H_x_i.detach();
          mixed_hessian.index_put_({Slice(), i}, grad_H_x_i);
        }
    }
    auto ptc = pt.detach();
    auto xtc = xt.detach();
    auto Hvaluec = Hvalue.detach();
    auto mixed_hessianc = mixed_hessian.detach();
    // Return the mixed Hessian
    return mixed_hessianc;
}



template<typename T>
torch::Tensor ppppppH(const torch::Tensor &x, 
                      const torch::Tensor &p, 
                      T W,
                      std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, T)> H)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach().set_requires_grad(false);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);
    // Compute the first-order gradient of H with respect to pt
    auto grad_H_p = torch::autograd::grad({Hvalue}, 
                                          {pt}, 
                                          {torch::ones_like(Hvalue)}, 
                                          true, 
                                          true,
                                          true)[0];

    // Initialize a tensor to hold the third-order gradients
    auto batch_size = p.size(0);
    auto p_dim = p.size(1);
    torch::Tensor third_order_derivative = torch::zeros({batch_size, p_dim, p_dim, p_dim}, p.options());

    // Compute the second-order and third-order gradients
    for (int i = 0; i < p_dim; ++i)
    {
        // Zero out the gradients of pt before computing the gradient for the next dimension
        if (pt.grad().defined()) {
            pt.grad().zero_();
        }
        
        // Compute the gradient of the i-th component of grad_H_p with respect to pt
        /*auto grad_H_p_i = torch::autograd::grad({grad_H_p.index({torch::indexing::Slice(), i})}, 
                                                {pt}, 
                                                {torch::ones_like(grad_H_p.index({torch::indexing::Slice(), i}))}, 
                                                true, 
                                                true,
                                                true)[0];*/
        auto grad_H_p_i = safe_jac(grad_H_p.index({torch::indexing::Slice(), i}), pt);
        // Ensure grad_H_p_i requires gradients
        for (int j = 0; j < p_dim; ++j)
        {

            // Zero out the gradients of pt before computing the gradient for the next dimension
            if (pt.grad().defined()) {
                pt.grad().zero_();
            }

            //grad_H_p_i = ensure_grad(grad_H_p_i);
            // Compute the gradient of the j-th component of grad_H_p_i with respect to pt
            /*auto grad_H_p_ij = torch::autograd::grad({grad_H_p_i.index({torch::indexing::Slice(), j})}, 
                                                     {pt}, 
                                                     {torch::ones_like(grad_H_p_i.index({torch::indexing::Slice(), j}))},
                                                     true,
                                                     true,
                                                     true)[0];*/
            auto grad_H_p_ij = safe_jac(grad_H_p_i.index({torch::indexing::Slice(), j}), pt);
            if ( grad_H_p_ij.requires_grad()) 
            {
              grad_H_p_ij.detach();
              // Store the third-order gradient
              third_order_derivative.index_put_({torch::indexing::Slice(), i, j}, grad_H_p_ij);
            }
        }
        grad_H_p_i.detach();
    
    }
    auto ptc = pt.detach();
    auto Hvaluec = Hvalue.detach();
    auto third_order_derivativec = third_order_derivative.detach();
    // Return the third-order derivative tensor
    return third_order_derivativec;
}


template<typename T>
torch::Tensor ppppppHu(const torch::Tensor &x, 
                       const torch::Tensor &p,
                       const torch::Tensor &u, 
                       T W,
                       std::function<torch::Tensor(const torch::Tensor&, 
                                                   const torch::Tensor&,
                                                   const torch::Tensor&, 
                                                   T)> Hu)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach().set_requires_grad(false);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = Hu(xt, pt, u, W);
    // Compute the first-order gradient of H with respect to pt
    auto grad_H_p = torch::autograd::grad({Hvalue}, 
                                          {pt}, 
                                          {torch::ones_like(Hvalue)}, 
                                          true, 
                                          true,
                                          true)[0];

    // Initialize a tensor to hold the third-order gradients
    auto batch_size = p.size(0);
    auto p_dim = p.size(1);
    torch::Tensor third_order_derivative = torch::zeros({batch_size, p_dim, p_dim, p_dim}, p.options());

    // Compute the second-order and third-order gradients
    for (int i = 0; i < p_dim; ++i)
    {
        // Zero out the gradients of pt before computing the gradient for the next dimension
        if (pt.grad().defined()) {
            pt.grad().zero_();
        }
        
        // Compute the gradient of the i-th component of grad_H_p with respect to pt
        /*auto grad_H_p_i = torch::autograd::grad({grad_H_p.index({torch::indexing::Slice(), i})}, 
                                                {pt}, 
                                                {torch::ones_like(grad_H_p.index({torch::indexing::Slice(), i}))}, 
                                                true, 
                                                true,
                                                true)[0];*/
        auto grad_H_p_i = safe_jac(grad_H_p.index({torch::indexing::Slice(), i}), pt);
        if ( grad_H_p_i.requires_grad())
        {
        // Ensure grad_H_p_i requires gradients
        for (int j = 0; j < p_dim; ++j)
        {

            // Zero out the gradients of pt before computing the gradient for the next dimension
            if (pt.grad().defined()) {
                pt.grad().zero_();
            }

            //grad_H_p_i = ensure_grad(grad_H_p_i);
            // Compute the gradient of the j-th component of grad_H_p_i with respect to pt
            /*auto grad_H_p_ij = torch::autograd::grad({grad_H_p_i.index({torch::indexing::Slice(), j})}, 
                                                     {pt}, 
                                                     {torch::ones_like(grad_H_p_i.index({torch::indexing::Slice(), j}))},
                                                     true,
                                                     true,
                                                     true)[0];*/
            auto grad_H_p_ij = safe_jac(grad_H_p_i.index({torch::indexing::Slice(), j}), pt);
            if ( grad_H_p_ij.requires_grad()) 
            {
              grad_H_p_ij.detach();
              // Store the third-order gradient
              third_order_derivative.index_put_({torch::indexing::Slice(), i, j}, grad_H_p_ij);
            }
        }
        grad_H_p_i.detach();
        }
    }
    auto ptc = pt.detach();
    auto Hvaluec = Hvalue.detach();
    auto third_order_derivativec = third_order_derivative.detach();
    // Return the third-order derivative tensor
    return third_order_derivativec;
}



template<typename T>
torch::Tensor pxpxpxH(const torch::Tensor &x, 
                      const torch::Tensor &p, 
                      T W,
                      std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, T)> H)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(false);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the first-order gradient of H with respect to p
    auto grad_H_p = torch::autograd::grad({Hvalue}, 
                                          {xt}, 
                                          {torch::ones_like(Hvalue)}, 
                                          true, 
                                          true)[0];
    auto batch_size = p.size(0);
    auto x_dim = x.size(1);
    torch::Tensor third_order_derivative = torch::zeros({batch_size, x_dim, x_dim, x_dim}, p.options());

    // Compute the second-order and third-order gradients
    for (int i = 0; i < x_dim; ++i)
    {
        // Zero out the gradients of pt before computing the gradient for the next dimension
        if (xt.grad().defined()) {
            xt.grad().zero_();
        }
        // Compute the gradient of the i-th component of grad_H_p with respect to p
        /*auto grad_H_p_i = torch::autograd::grad({grad_H_p.index({Slice(), i})}, 
                                                {xt}, 
                                                {torch::ones_like(grad_H_p.index({Slice(), i}))}, 
                                                true, 
                                                true)[0];*/
        auto grad_H_p_i = safe_jac(grad_H_p.index({Slice(), i}), xt);
        for (int j = 0; j < x_dim; ++j)
        {
            if (xt.grad().defined()) {
              xt.grad().zero_();
            }



            // Compute the gradient of the j-th component of grad_H_p_i with respect to p
            /*auto grad_H_p_ij = torch::autograd::grad({grad_H_p_i.index({Slice(), j})}, 
                                                     {xt}, 
                                                     {torch::ones_like(grad_H_p_i.index({Slice(), j}))}, 
                                                     true, 
                                                     true)[0];*/
            auto grad_H_p_ij = safe_jac(grad_H_p_i.index({Slice(), j}), xt);
            if (grad_H_p_ij.requires_grad()) 
            {
              grad_H_p_ij.detach();
              //third_order_derivative.select(1, i).select(2, j).copy_(grad_H_p_ij);
              third_order_derivative.index_put_({Slice(), i, j}, grad_H_p_ij);
            }
        }
        grad_H_p_i.detach();
    }
    auto xtc = xt.detach();
    auto Hvaluec = Hvalue.detach();
    auto third_order_derivativec = third_order_derivative.detach();
    // Return the third-order derivative tensor
    return third_order_derivativec;
}


template<typename T>
torch::Tensor pxpxpxHu(const torch::Tensor &x, 
                       const torch::Tensor &p, 
                       const torch::Tensor &u,
                       T W,
                       std::function<torch::Tensor(const torch::Tensor&, 
                                                   const torch::Tensor&,
                                                   const torch::Tensor&, 
                                                   T)> Hu)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(false);

    // Compute the Hamiltonian value
    auto Hvalue = Hu(xt, pt, u, W);

    // Compute the first-order gradient of H with respect to p
    auto grad_H_p = torch::autograd::grad({Hvalue}, 
                                          {xt}, 
                                          {torch::ones_like(Hvalue)}, 
                                          true, 
                                          true)[0];
    auto batch_size = p.size(0);
    auto x_dim = x.size(1);
    torch::Tensor third_order_derivative = torch::zeros({batch_size, x_dim, x_dim, x_dim}, p.options());

    // Compute the second-order and third-order gradients
    for (int i = 0; i < x_dim; ++i)
    {
        // Zero out the gradients of pt before computing the gradient for the next dimension
        if (xt.grad().defined()) {
            xt.grad().zero_();
        }
        // Compute the gradient of the i-th component of grad_H_p with respect to p
        /*auto grad_H_p_i = torch::autograd::grad({grad_H_p.index({Slice(), i})}, 
                                                {xt}, 
                                                {torch::ones_like(grad_H_p.index({Slice(), i}))}, 
                                                true, 
                                                true)[0];*/
        auto grad_H_p_i = safe_jac(grad_H_p.index({Slice(), i}), xt);
        for (int j = 0; j < x_dim; ++j)
        {
            if (xt.grad().defined()) {
              xt.grad().zero_();
            }



            // Compute the gradient of the j-th component of grad_H_p_i with respect to p
            /*auto grad_H_p_ij = torch::autograd::grad({grad_H_p_i.index({Slice(), j})}, 
                                                     {xt}, 
                                                     {torch::ones_like(grad_H_p_i.index({Slice(), j}))}, 
                                                     true, 
                                                     true)[0];*/
            auto grad_H_p_ij = safe_jac(grad_H_p_i.index({Slice(), j}), xt);
            if (grad_H_p_ij.requires_grad()) 
            {
              grad_H_p_ij.detach();
              //third_order_derivative.select(1, i).select(2, j).copy_(grad_H_p_ij);
              third_order_derivative.index_put_({Slice(), i, j}, grad_H_p_ij);
            }
        }
        grad_H_p_i.detach();
    }
    auto xtc = xt.detach();
    auto Hvaluec = Hvalue.detach();
    auto third_order_derivativec = third_order_derivative.detach();
    // Return the third-order derivative tensor
    return third_order_derivativec;
}


template<typename T>
torch::Tensor pppppxH(const torch::Tensor &x, 
                      const torch::Tensor &p, 
                      T W,
                      std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, T)> H)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(true);
    if (xt.grad().defined()) {
      xt.grad().zero_();
    }
    if ( pt.grad().defined()) {
      pt.grad().zero_();
    }
    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the first-order gradient of H with respect to p
    auto grad_H_x = torch::autograd::grad({Hvalue}, 
                                          {xt}, 
                                          {torch::ones_like(Hvalue)}, 
                                          true, 
                                          true, 
                                          true)[0];
    // Initialize a tensor to hold the third-order gradients
    auto batch_size = p.size(0);
    auto x_dim = x.size(1);
    auto p_dim = p.size(1);
    torch::Tensor third_order_derivative = torch::zeros({batch_size, x_dim, p_dim, p_dim}, p.options());

    // Compute the second-order and third-order gradients
    for (int i = 0; i < p_dim; ++i)
    {
        // Zero out the gradients of pt before computing the gradient for the next dimension
        if ( xt.grad().defined()) {
            xt.grad().zero_();
        }
        if ( pt.grad().defined()) {
            pt.grad().zero_();
        }

        // Compute the gradient of the i-th component of grad_H_p with respect to p
        /*auto grad_H_x_p_i = torch::autograd::grad({grad_H_x.index({Slice(), i})}, 
                                                {pt}, 
                                                {torch::ones_like(grad_H_x.index({Slice(), i}))}, 
                                                true, 
                                                true)[0];*/
        auto grad_H_x_p_i = safe_jac(grad_H_x.index({Slice(), i}), pt);
        for (int j = 0; j < p_dim; ++j)
        {
            // Zero out the gradients of pt before computing the gradient for the next dimension
            if ( xt.grad().defined()) {
                xt.grad().zero_();
            }
            if ( pt.grad().defined()) {
                pt.grad().zero_();
            }

            // Compute the gradient of the j-th component of grad_H_p_i with respect to p
            /*auto grad_H_x_p_ij = torch::autograd::grad({grad_H_x_p_i.index({Slice(), j})}, 
                                                       {pt}, 
                                                       {torch::ones_like(grad_H_x_p_i.index({Slice(), j}))}, 
                                                       true, 
                                                       true)[0];*/
            auto grad_H_x_p_ij = safe_jac(grad_H_x_p_i.index({Slice(), j}), pt);
            if ( grad_H_x_p_ij.requires_grad()) {
              grad_H_x_p_ij.detach();
              //third_order_derivative.select(1, i).select(2, j).copy_(grad_H_x_p_ij);
              third_order_derivative.index_put_({Slice(), i, j}, grad_H_x_p_ij);
            }
        }
        grad_H_x_p_i.detach();
    }
    auto xtc = xt.detach();
    auto ptc = pt.detach();
    auto Hvaluec = Hvalue.detach();

    auto third_order_derivativec = third_order_derivative.detach();
    // Return the third-order derivative tensor
    return third_order_derivativec;
}

template<typename T>
torch::Tensor pppppxHu(const torch::Tensor &x, 
                       const torch::Tensor &p,
                       const torch::Tensor &u, 
                       T W,
                       std::function<torch::Tensor(const torch::Tensor&, 
                                                   const torch::Tensor&,
                                                   const torch::Tensor&, 
                                                   T)> Hu)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(true);
    if (xt.grad().defined()) {
      xt.grad().zero_();
    }
    if ( pt.grad().defined()) {
      pt.grad().zero_();
    }
    // Compute the Hamiltonian value
    auto Hvalue = Hu(xt, pt, u, W);

    // Compute the first-order gradient of H with respect to p
    auto grad_H_x = torch::autograd::grad({Hvalue}, 
                                          {xt}, 
                                          {torch::ones_like(Hvalue)}, 
                                          true, 
                                          true, 
                                          true)[0];
    // Initialize a tensor to hold the third-order gradients
    auto batch_size = p.size(0);
    auto x_dim = x.size(1);
    auto p_dim = p.size(1);
    torch::Tensor third_order_derivative = torch::zeros({batch_size, x_dim, p_dim, p_dim}, p.options());

    // Compute the second-order and third-order gradients
    for (int i = 0; i < p_dim; ++i)
    {
        // Zero out the gradients of pt before computing the gradient for the next dimension
        if ( xt.grad().defined()) {
            xt.grad().zero_();
        }
        if ( pt.grad().defined()) {
            pt.grad().zero_();
        }

        // Compute the gradient of the i-th component of grad_H_p with respect to p
        /*auto grad_H_x_p_i = torch::autograd::grad({grad_H_x.index({Slice(), i})}, 
                                                {pt}, 
                                                {torch::ones_like(grad_H_x.index({Slice(), i}))}, 
                                                true, 
                                                true)[0];*/
        auto grad_H_x_p_i = safe_jac(grad_H_x.index({Slice(), i}), pt);
        for (int j = 0; j < p_dim; ++j)
        {
            // Zero out the gradients of pt before computing the gradient for the next dimension
            if ( xt.grad().defined()) {
                xt.grad().zero_();
            }
            if ( pt.grad().defined()) {
                pt.grad().zero_();
            }

            // Compute the gradient of the j-th component of grad_H_p_i with respect to p
            /*auto grad_H_x_p_ij = torch::autograd::grad({grad_H_x_p_i.index({Slice(), j})}, 
                                                       {pt}, 
                                                       {torch::ones_like(grad_H_x_p_i.index({Slice(), j}))}, 
                                                       true, 
                                                       true)[0];*/
            auto grad_H_x_p_ij = safe_jac(grad_H_x_p_i.index({Slice(), j}), pt);
            if ( grad_H_x_p_ij.requires_grad()) {
              grad_H_x_p_ij.detach();
              //third_order_derivative.select(1, i).select(2, j).copy_(grad_H_x_p_ij);
              third_order_derivative.index_put_({Slice(), i, j}, grad_H_x_p_ij);
            }
        }
        grad_H_x_p_i.detach();
    }
    auto xtc = xt.detach();
    auto ptc = pt.detach();
    auto Hvaluec = Hvalue.detach();

    auto third_order_derivativec = third_order_derivative.detach();
    // Return the third-order derivative tensor
    return third_order_derivativec;
}




template<typename T>
torch::Tensor pppxpxH(const torch::Tensor &x, 
                      const torch::Tensor &p, 
                      T W,
                      std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, T)> H)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the first-order gradient of H with respect to p
    auto grad_H_x = torch::autograd::grad({Hvalue}, 
                                          {xt}, 
                                          {torch::ones_like(Hvalue)}, 
                                          true, 
                                          true)[0];
    // Initialize a tensor to hold the third-order gradients
    auto batch_size = p.size(0);
    auto x_dim = x.size(1);
    auto p_dim = p.size(1);
    torch::Tensor third_order_derivative = torch::zeros({batch_size, x_dim, x_dim, p_dim}, p.options());

    // Compute the second-order and third-order gradients
    for (int i = 0; i < x_dim; ++i)
    {
        // Zero out the gradients of pt before computing the gradient for the next dimension
        if ( xt.grad().defined()) {
            xt.grad().zero_();
        }
        if ( pt.grad().defined()) {
            pt.grad().zero_();
        }

        // Compute the gradient of the i-th component of grad_H_x with respect to x
        /*auto grad_H_x_i = torch::autograd::grad({grad_H_x.index({Slice(), i})}, 
                                                {xt}, 
                                                {torch::ones_like(grad_H_x.index({Slice(), i}))}, 
                                                true, 
                                                true)[0];*/
        auto grad_H_x_i = safe_jac(grad_H_x.index({Slice(), i}), xt);
        for (int j = 0; j < x_dim; ++j)
        {
            // Zero out the gradients of pt before computing the gradient for the next dimension
            if ( xt.grad().defined()) {
                xt.grad().zero_();
            }
            if ( pt.grad().defined()) {
                pt.grad().zero_();
            }


            // Compute the gradient of the j-th component of grad_H_x_i with respect to p
            /*auto grad_H_x_ij = torch::autograd::grad({grad_H_x_i.index({Slice(), j})}, 
                                                     {pt}, 
                                                     {torch::ones_like(grad_H_x_i.index({Slice(), j}))}, 
                                                     true, 
                                                     true)[0];*/
            auto grad_H_x_ij = safe_jac(grad_H_x_i.index({Slice(), j}), pt);
            if ( grad_H_x_ij.requires_grad()) {
              grad_H_x_ij.detach();
              third_order_derivative.index_put_({Slice(), i, j}, grad_H_x_ij);
            }
        }
        grad_H_x_i.detach();
    }
    auto xtc = xt.detach();
    auto ptc = pt.detach();
    auto Hvaluec = Hvalue.detach();
    auto third_order_derivativec = third_order_derivative.detach();
    // Return the third-order derivative tensor
    return third_order_derivativec;
}


template<typename T>
torch::Tensor pppxpxHu(const torch::Tensor &x, 
                       const torch::Tensor &p,
                       const torch::Tensor &u, 
                       T W,
                       std::function<torch::Tensor(const torch::Tensor&, 
                                                   const torch::Tensor&, 
                                                   const torch::Tensor&,
                                                   T)> Hu)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = Hu(xt, pt, u, W);

    // Compute the first-order gradient of H with respect to p
    auto grad_H_x = torch::autograd::grad({Hvalue}, 
                                          {xt}, 
                                          {torch::ones_like(Hvalue)}, 
                                          true, 
                                          true)[0];
    // Initialize a tensor to hold the third-order gradients
    auto batch_size = p.size(0);
    auto x_dim = x.size(1);
    auto p_dim = p.size(1);
    torch::Tensor third_order_derivative = torch::zeros({batch_size, x_dim, x_dim, p_dim}, p.options());

    // Compute the second-order and third-order gradients
    for (int i = 0; i < x_dim; ++i)
    {
        // Zero out the gradients of pt before computing the gradient for the next dimension
        if ( xt.grad().defined()) {
            xt.grad().zero_();
        }
        if ( pt.grad().defined()) {
            pt.grad().zero_();
        }

        // Compute the gradient of the i-th component of grad_H_x with respect to x
        /*auto grad_H_x_i = torch::autograd::grad({grad_H_x.index({Slice(), i})}, 
                                                {xt}, 
                                                {torch::ones_like(grad_H_x.index({Slice(), i}))}, 
                                                true, 
                                                true)[0];*/
        auto grad_H_x_i = safe_jac(grad_H_x.index({Slice(), i}), xt);
        for (int j = 0; j < x_dim; ++j)
        {
            // Zero out the gradients of pt before computing the gradient for the next dimension
            if ( xt.grad().defined()) {
                xt.grad().zero_();
            }
            if ( pt.grad().defined()) {
                pt.grad().zero_();
            }


            // Compute the gradient of the j-th component of grad_H_x_i with respect to p
            /*auto grad_H_x_ij = torch::autograd::grad({grad_H_x_i.index({Slice(), j})}, 
                                                     {pt}, 
                                                     {torch::ones_like(grad_H_x_i.index({Slice(), j}))}, 
                                                     true, 
                                                     true)[0];*/
            auto grad_H_x_ij = safe_jac(grad_H_x_i.index({Slice(), j}), pt);
            if ( grad_H_x_ij.requires_grad()) {
              grad_H_x_ij.detach();
              third_order_derivative.index_put_({Slice(), i, j}, grad_H_x_ij);
            }
        }
        grad_H_x_i.detach();
    }
    auto xtc = xt.detach();
    auto ptc = pt.detach();
    auto Hvaluec = Hvalue.detach();
    auto third_order_derivativec = third_order_derivative.detach();
    // Return the third-order derivative tensor
    return third_order_derivativec;
}





template<typename T>
torch::Tensor pxpppxH(const torch::Tensor &x, 
                      const torch::Tensor &p, 
                      T W,
                      std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, T)> H)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the first-order gradient of H with respect to p
    auto grad_H_x = torch::autograd::grad({Hvalue}, 
                                          {xt}, 
                                          {torch::ones_like(Hvalue)}, 
                                          true, 
                                          true)[0];
    // Initialize a tensor to hold the third-order gradients
    auto batch_size = p.size(0);
    auto x_dim = x.size(1);
    auto p_dim = p.size(1);
    torch::Tensor third_order_derivative = torch::zeros({batch_size, x_dim, p_dim, x_dim}, p.options());

    // Compute the second-order and third-order gradients
    for (int i = 0; i < x_dim; ++i)
    {
        // Zero out the gradients of pt before computing the gradient for the next dimension
        if ( xt.grad().defined()) {
            xt.grad().zero_();
        }
        if ( pt.grad().defined()) {
            pt.grad().zero_();
        }

        // Compute the gradient of the i-th component of grad_H_p with respect to p
        /*auto grad_H_p_i = torch::autograd::grad({grad_H_x.index({Slice(), i})}, 
                                                {pt}, 
                                                {torch::ones_like(grad_H_x.index({Slice(), i}))}, 
                                                true, 
                                                true)[0];*/
        auto grad_H_p_i = safe_jac(grad_H_x.index({Slice(), i}), pt);
        for (int j = 0; j < p_dim; ++j)
        {
            // Zero out the gradients of pt before computing the gradient for the next dimension
            if ( xt.grad().defined()) {
                xt.grad().zero_();
            }
            if ( pt.grad().defined()) {
                pt.grad().zero_();
            }


            // Compute the gradient of the j-th component of grad_H_p_i with respect to p
            /*auto grad_H_p_ij = torch::autograd::grad({grad_H_p_i.index({Slice(), j})}, 
                                                     {xt}, 
                                                     {torch::ones_like(grad_H_p_i.index({Slice(), j}))}, 
                                                     true, 
                                                     true)[0];*/
            auto grad_H_p_ij = safe_jac(grad_H_p_i.index({Slice(), j}), xt);
            if ( grad_H_p_ij.requires_grad()) {
              grad_H_p_ij.detach();
              //third_order_derivative.select(1, i).select(2, j).copy_(grad_H_p_ij);
              third_order_derivative.index_put_({Slice(), i, j}, grad_H_p_ij);
            }
        }
        grad_H_p_i.detach();
    }
    auto xtc     = xt.detach();
    auto ptc     = pt.detach();
    auto Hvaluec = Hvalue.detach();
    auto third_order_derivativec = third_order_derivative.detach();
    // Return the third-order derivative tensor
    return third_order_derivativec;
}


template<typename T>
torch::Tensor pxpppxHu(const torch::Tensor &x, 
                       const torch::Tensor &p,
                       const torch::Tensor &u, 
                       T W,
                       std::function<torch::Tensor(const torch::Tensor&, 
                                                   const torch::Tensor&, 
                                                   const torch::Tensor&,
                                                   T)> Hu)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = Hu(xt, pt, u, W);

    // Compute the first-order gradient of H with respect to p
    auto grad_H_x = torch::autograd::grad({Hvalue}, 
                                          {xt}, 
                                          {torch::ones_like(Hvalue)}, 
                                          true, 
                                          true)[0];
    // Initialize a tensor to hold the third-order gradients
    auto batch_size = p.size(0);
    auto x_dim = x.size(1);
    auto p_dim = p.size(1);
    torch::Tensor third_order_derivative = torch::zeros({batch_size, x_dim, p_dim, x_dim}, p.options());

    // Compute the second-order and third-order gradients
    for (int i = 0; i < x_dim; ++i)
    {
        // Zero out the gradients of pt before computing the gradient for the next dimension
        if ( xt.grad().defined()) {
            xt.grad().zero_();
        }
        if ( pt.grad().defined()) {
            pt.grad().zero_();
        }

        // Compute the gradient of the i-th component of grad_H_p with respect to p
        /*auto grad_H_p_i = torch::autograd::grad({grad_H_x.index({Slice(), i})}, 
                                                {pt}, 
                                                {torch::ones_like(grad_H_x.index({Slice(), i}))}, 
                                                true, 
                                                true)[0];*/
        auto grad_H_p_i = safe_jac(grad_H_x.index({Slice(), i}), pt);
        for (int j = 0; j < p_dim; ++j)
        {
            // Zero out the gradients of pt before computing the gradient for the next dimension
            if ( xt.grad().defined()) {
                xt.grad().zero_();
            }
            if ( pt.grad().defined()) {
                pt.grad().zero_();
            }


            // Compute the gradient of the j-th component of grad_H_p_i with respect to p
            /*auto grad_H_p_ij = torch::autograd::grad({grad_H_p_i.index({Slice(), j})}, 
                                                     {xt}, 
                                                     {torch::ones_like(grad_H_p_i.index({Slice(), j}))}, 
                                                     true, 
                                                     true)[0];*/
            auto grad_H_p_ij = safe_jac(grad_H_p_i.index({Slice(), j}), xt);
            if ( grad_H_p_ij.requires_grad()) {
              grad_H_p_ij.detach();
              //third_order_derivative.select(1, i).select(2, j).copy_(grad_H_p_ij);
              third_order_derivative.index_put_({Slice(), i, j}, grad_H_p_ij);
            }
        }
        grad_H_p_i.detach();
    }
    auto xtc     = xt.detach();
    auto ptc     = pt.detach();
    auto Hvaluec = Hvalue.detach();
    auto third_order_derivativec = third_order_derivative.detach();
    // Return the third-order derivative tensor
    return third_order_derivativec;
}



template<typename T>
torch::Tensor pppxppH(const torch::Tensor &x, 
                      const torch::Tensor &p, 
                      T W,
                      std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, T)> H)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the first-order gradient of H with respect to p
    auto grad_H_p = torch::autograd::grad({Hvalue}, 
                                          {pt}, 
                                          {torch::ones_like(Hvalue)}, 
                                          true, 
                                          true)[0];
    // Initialize a tensor to hold the third-order gradients
    auto batch_size = p.size(0);
    auto x_dim = x.size(1);
    auto p_dim = p.size(1);
    torch::Tensor third_order_derivative = torch::zeros({batch_size, p_dim, x_dim, x_dim}, p.options());

    // Compute the second-order and third-order gradients
    for (int i = 0; i < x_dim; ++i)
    {
        // Zero out the gradients of pt before computing the gradient for the next dimension
        if (xt.grad().defined()) {
            xt.grad().zero_();
        }
        if ( pt.grad().defined()) {
            pt.grad().zero_();
        }

        grad_H_p = ensure_grad(grad_H_p);

        // Compute the gradient of the i-th component of grad_H_p with respect to p
        /*auto grad_H_p_i = torch::autograd::grad({grad_H_p.index({Slice(), i})}, 
                                                {xt}, 
                                                {torch::ones_like(grad_H_p.index({Slice(), i}))}, 
                                                true, 
                                                true)[0];*/
        auto grad_H_p_i = safe_jac(grad_H_p.index({Slice(), i}), xt);
        for (int j = 0; j < x_dim; ++j)
        {
            // Zero out the gradients of pt before computing the gradient for the next dimension
            if (xt.grad().defined()) {
                xt.grad().zero_();
            }
            if ( pt.grad().defined()) {
                pt.grad().zero_();
            }

            // Compute the gradient of the j-th component of grad_H_p_i with respect to p
            /*auto grad_H_p_ij = torch::autograd::grad({grad_H_p_i.index({Slice(), j})}, 
                                                     {pt}, 
                                                     {torch::ones_like(grad_H_p_i.index({Slice(), j}))}, 
                                                     true, 
                                                     true)[0];*/
            auto grad_H_p_ij = safe_jac(grad_H_p_i.index({Slice(), j}), pt);
            if (grad_H_p_ij.requires_grad()) 
            {
              grad_H_p_ij.detach();
              //third_order_derivative.select(1, i).select(2, j).copy_(grad_H_p_ij);
              third_order_derivative.index_put_({Slice(), i, j}, grad_H_p_ij);
            }
        }
        grad_H_p_i.detach();
    }
    auto xtc     = xt.detach();
    auto ptc     = pt.detach();
    auto Hvaluec = Hvalue.detach();
    auto third_order_derivativec = third_order_derivative.detach();
    // Return the third-order derivative tensor
    return third_order_derivativec;
}


template<typename T>
torch::Tensor pppxppHu(const torch::Tensor &x, 
                       const torch::Tensor &p,
                       const torch::Tensor &u, 
                       T W,
                       std::function<torch::Tensor(const torch::Tensor&, 
                                                   const torch::Tensor&, 
                                                   const torch::Tensor&,
                                                   T)> Hu)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = Hu(xt, pt, u, W);

    // Compute the first-order gradient of H with respect to p
    auto grad_H_p = torch::autograd::grad({Hvalue}, 
                                          {pt}, 
                                          {torch::ones_like(Hvalue)}, 
                                          true, 
                                          true)[0];
    // Initialize a tensor to hold the third-order gradients
    auto batch_size = p.size(0);
    auto x_dim = x.size(1);
    auto p_dim = p.size(1);
    torch::Tensor third_order_derivative = torch::zeros({batch_size, p_dim, x_dim, x_dim}, p.options());

    // Compute the second-order and third-order gradients
    for (int i = 0; i < x_dim; ++i)
    {
        // Zero out the gradients of pt before computing the gradient for the next dimension
        if (xt.grad().defined()) {
            xt.grad().zero_();
        }
        if ( pt.grad().defined()) {
            pt.grad().zero_();
        }

        grad_H_p = ensure_grad(grad_H_p);

        // Compute the gradient of the i-th component of grad_H_p with respect to p
        /*auto grad_H_p_i = torch::autograd::grad({grad_H_p.index({Slice(), i})}, 
                                                {xt}, 
                                                {torch::ones_like(grad_H_p.index({Slice(), i}))}, 
                                                true, 
                                                true)[0];*/
        auto grad_H_p_i = safe_jac(grad_H_p.index({Slice(), i}), xt);
        for (int j = 0; j < x_dim; ++j)
        {
            // Zero out the gradients of pt before computing the gradient for the next dimension
            if (xt.grad().defined()) {
                xt.grad().zero_();
            }
            if ( pt.grad().defined()) {
                pt.grad().zero_();
            }

            // Compute the gradient of the j-th component of grad_H_p_i with respect to p
            /*auto grad_H_p_ij = torch::autograd::grad({grad_H_p_i.index({Slice(), j})}, 
                                                     {pt}, 
                                                     {torch::ones_like(grad_H_p_i.index({Slice(), j}))}, 
                                                     true, 
                                                     true)[0];*/
            auto grad_H_p_ij = safe_jac(grad_H_p_i.index({Slice(), j}), pt);
            if ( grad_H_p_ij.requires_grad()) {
              grad_H_p_ij.detach();
              //third_order_derivative.select(1, i).select(2, j).copy_(grad_H_p_ij);
              third_order_derivative.index_put_({Slice(), i, j}, grad_H_p_ij);
            }
        }
        grad_H_p_i.detach();
    }
    auto xtc     = xt.detach();
    auto ptc     = pt.detach();
    auto Hvaluec = Hvalue.detach();
    auto third_order_derivativec = third_order_derivative.detach();
    // Return the third-order derivative tensor
    return third_order_derivativec;
}



template<typename T>
torch::Tensor pxpxppH(const torch::Tensor &x, 
                      const torch::Tensor &p, 
                      T W,
                      std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, T)> H)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the first-order gradient of H with respect to p
    auto grad_H_p = torch::autograd::grad({Hvalue}, 
                                          {pt}, 
                                          {torch::ones_like(Hvalue)}, 
                                          true, 
                                          true)[0];
    // Initialize a tensor to hold the third-order gradients
    auto batch_size = p.size(0);
    auto x_dim = x.size(1);
    auto p_dim = p.size(1);
    torch::Tensor third_order_derivative = torch::zeros({batch_size, p_dim, x_dim, x_dim}, p.options());

    // Compute the second-order and third-order gradients
    for (int i = 0; i < x_dim; ++i)
    {
        // Zero out the gradients of pt before computing the gradient for the next dimension
        if (xt.grad().defined()) {
            xt.grad().zero_();
        }
        if ( pt.grad().defined()) {
            pt.grad().zero_();
        }

        // Compute the gradient of the i-th component of grad_H_p with respect to p
        /*auto grad_H_p_i = torch::autograd::grad({grad_H_p.index({Slice(), i})}, 
                                                {xt}, 
                                                {torch::ones_like(grad_H_p.index({Slice(), i}))}, 
                                                true, 
                                                true)[0];*/
        auto grad_H_p_i = safe_jac(grad_H_p.index({Slice(), i}), xt);
        for (int j = 0; j < x_dim; ++j)
        {
            // Zero out the gradients of pt before computing the gradient for the next dimension
            if (xt.grad().defined()) {
                xt.grad().zero_();
            }
            if ( pt.grad().defined()) {
                pt.grad().zero_();
            }


            // Compute the gradient of the j-th component of grad_H_p_i with respect to p
            /*auto grad_H_p_ij = torch::autograd::grad({grad_H_p_i.index({Slice(), j})}, 
                                                     {xt}, 
                                                     {torch::ones_like(grad_H_p_i.index({Slice(), j}))}, 
                                                     true, 
                                                     true)[0];*/
            auto grad_H_p_ij = safe_jac(grad_H_p_i.index({Slice(), j}), xt);
            if ( grad_H_p_ij.requires_grad()) {
              grad_H_p_ij.detach();
              //third_order_derivative.select(1, i).select(2, j).copy_(grad_H_p_ij);
              third_order_derivative.index_put_({Slice(), i, j}, grad_H_p_ij);
            }
        }
        grad_H_p_i.detach();
    }
    auto xtc     = xt.detach();
    auto ptc     = pt.detach();
    auto Hvaluec = Hvalue.detach();
    auto third_order_derivativec = third_order_derivative.detach();
    // Return the third-order derivative tensor
    return third_order_derivativec;
}

template<typename T>
torch::Tensor pxpxppHu(const torch::Tensor &x, 
                       const torch::Tensor &p,
                       const torch::Tensor &u, 
                       T W,
                       std::function<torch::Tensor(const torch::Tensor&, 
                                                   const torch::Tensor&, 
                                                   const torch::Tensor&,
                                                   T)> Hu)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = Hu(xt, pt, u, W);

    // Compute the first-order gradient of H with respect to p
    auto grad_H_p = torch::autograd::grad({Hvalue}, 
                                          {pt}, 
                                          {torch::ones_like(Hvalue)}, 
                                          true, 
                                          true)[0];
    // Initialize a tensor to hold the third-order gradients
    auto batch_size = p.size(0);
    auto x_dim = x.size(1);
    auto p_dim = p.size(1);
    torch::Tensor third_order_derivative = torch::zeros({batch_size, p_dim, x_dim, x_dim}, p.options());

    // Compute the second-order and third-order gradients
    for (int i = 0; i < x_dim; ++i)
    {
        // Zero out the gradients of pt before computing the gradient for the next dimension
        if (xt.grad().defined()) {
            xt.grad().zero_();
        }
        if ( pt.grad().defined()) {
            pt.grad().zero_();
        }

        // Compute the gradient of the i-th component of grad_H_p with respect to p
        /*auto grad_H_p_i = torch::autograd::grad({grad_H_p.index({Slice(), i})}, 
                                                {xt}, 
                                                {torch::ones_like(grad_H_p.index({Slice(), i}))}, 
                                                true, 
                                                true)[0];*/
        auto grad_H_p_i = safe_jac(grad_H_p.index({Slice(), i}), xt);
        for (int j = 0; j < x_dim; ++j)
        {
            // Zero out the gradients of pt before computing the gradient for the next dimension
            if (xt.grad().defined()) {
                xt.grad().zero_();
            }
            if ( pt.grad().defined()) {
                pt.grad().zero_();
            }


            // Compute the gradient of the j-th component of grad_H_p_i with respect to p
            /*auto grad_H_p_ij = torch::autograd::grad({grad_H_p_i.index({Slice(), j})}, 
                                                     {xt}, 
                                                     {torch::ones_like(grad_H_p_i.index({Slice(), j}))}, 
                                                     true, 
                                                     true)[0];*/
            auto grad_H_p_ij = safe_jac(grad_H_p_i.index({Slice(), j}), xt);
            if ( grad_H_p_ij.requires_grad()) {
              grad_H_p_ij.detach();
              //third_order_derivative.select(1, i).select(2, j).copy_(grad_H_p_ij);
              third_order_derivative.index_put_({Slice(), i, j}, grad_H_p_ij);
            }
        }
        grad_H_p_i.detach();
    }
    auto xtc     = xt.detach();
    auto ptc     = pt.detach();
    auto Hvaluec = Hvalue.detach();
    auto third_order_derivativec = third_order_derivative.detach();
    // Return the third-order derivative tensor
    return third_order_derivativec;
}



template<typename T>
torch::Tensor pxppppH(const torch::Tensor &x, 
                      const torch::Tensor &p, 
                      T W,
                      std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, T)> H)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = H(xt, pt, W);

    // Compute the first-order gradient of H with respect to p
    auto grad_H_p = torch::autograd::grad({Hvalue}, 
                                          {pt}, 
                                          {torch::ones_like(Hvalue)}, 
                                          true, 
                                          true)[0];
    // Initialize a tensor to hold the third-order gradients
    auto batch_size = p.size(0);
    auto x_dim = x.size(1);
    auto p_dim = p.size(1);
    torch::Tensor third_order_derivative = torch::zeros({batch_size, p_dim, p_dim, x_dim}, p.options());

    // Compute the second-order and third-order gradients
    for (int i = 0; i < x_dim; ++i)
    {
        // Zero out the gradients of pt before computing the gradient for the next dimension
        if (xt.grad().defined()) {
            xt.grad().zero_();
        }
        if ( pt.grad().defined()) {
            pt.grad().zero_();
        }


        // Compute the gradient of the i-th component of grad_H_p with respect to p
        /*auto grad_H_p_i = torch::autograd::grad({grad_H_p.index({Slice(), i})}, 
                                                {pt}, 
                                                {torch::ones_like(grad_H_p.index({Slice(), i}))}, 
                                                true, 
                                                true)[0];*/
        auto grad_H_p_i = safe_jac(grad_H_p.index({Slice(), i}), pt);
        for (int j = 0; j < x_dim; ++j)
        {
            // Zero out the gradients of pt before computing the gradient for the next dimension
            if (xt.grad().defined()) {
                xt.grad().zero_();
            }
            if ( pt.grad().defined()) {
                pt.grad().zero_();
            }



            // Compute the gradient of the j-th component of grad_H_p_i with respect to p
            /*auto grad_H_p_ij = torch::autograd::grad({grad_H_p_i.index({Slice(), j})}, 
                                                     {xt}, 
                                                     {torch::ones_like(grad_H_p_i.index({Slice(), j}))}, 
                                                     true, 
                                                     true)[0];*/
            auto grad_H_p_ij = safe_jac(grad_H_p_i.index({Slice(), j}), xt);
            if ( grad_H_p_ij.requires_grad()) {
              grad_H_p_ij.detach();
              //third_order_derivative.select(1, i).select(2, j).copy_(grad_H_p_ij);
              third_order_derivative.index_put_({Slice(), i, j}, grad_H_p_ij);
            }
        }
        grad_H_p_i.detach();
    }
    auto xtc = xt.detach();
    auto ptc = pt.detach();
    auto Hvaluec = Hvalue.detach();
    auto third_order_derivativec = third_order_derivative.detach();
    // Return the third-order derivative tensor
    return third_order_derivativec;
}


template<typename T>
torch::Tensor pxppppHu(const torch::Tensor &x, 
                       const torch::Tensor &p,
                       const torch::Tensor &u, 
                       T W,
                       std::function<torch::Tensor(const torch::Tensor&, 
                                                   const torch::Tensor&,
                                                   const torch::Tensor&, 
                                                   T)> Hu)
{
    // Create tensors with gradient tracking for both x and p
    auto xt = x.clone().detach().set_requires_grad(true);
    auto pt = p.clone().detach().set_requires_grad(true);

    // Compute the Hamiltonian value
    auto Hvalue = Hu(xt, pt, u, W);

    // Compute the first-order gradient of H with respect to p
    auto grad_H_p = torch::autograd::grad({Hvalue}, 
                                          {pt}, 
                                          {torch::ones_like(Hvalue)}, 
                                          true, 
                                          true)[0];
    // Initialize a tensor to hold the third-order gradients
    auto batch_size = p.size(0);
    auto x_dim = x.size(1);
    auto p_dim = p.size(1);
    torch::Tensor third_order_derivative = torch::zeros({batch_size, p_dim, p_dim, x_dim}, p.options());

    // Compute the second-order and third-order gradients
    for (int i = 0; i < x_dim; ++i)
    {
        // Zero out the gradients of pt before computing the gradient for the next dimension
        if (xt.grad().defined()) {
            xt.grad().zero_();
        }
        if ( pt.grad().defined()) {
            pt.grad().zero_();
        }


        // Compute the gradient of the i-th component of grad_H_p with respect to p
        /*auto grad_H_p_i = torch::autograd::grad({grad_H_p.index({Slice(), i})}, 
                                                {pt}, 
                                                {torch::ones_like(grad_H_p.index({Slice(), i}))}, 
                                                true, 
                                                true)[0];*/
        auto grad_H_p_i = safe_jac(grad_H_p.index({Slice(), i}), pt);
        if (grad_H_p_i.requires_grad()) {
        for (int j = 0; j < x_dim; ++j)
        {
            // Zero out the gradients of pt before computing the gradient for the next dimension
            if (xt.grad().defined()) {
                xt.grad().zero_();
            }
            if ( pt.grad().defined()) {
                pt.grad().zero_();
            }



            // Compute the gradient of the j-th component of grad_H_p_i with respect to p
            /*auto grad_H_p_ij = torch::autograd::grad({grad_H_p_i.index({Slice(), j})}, 
                                                     {xt}, 
                                                     {torch::ones_like(grad_H_p_i.index({Slice(), j}))}, 
                                                     true, 
                                                     true)[0];*/
            auto grad_H_p_ij = safe_jac(grad_H_p_i.index({Slice(), j}), xt);
            if ( grad_H_p_ij.requires_grad()) {
              grad_H_p_ij.detach();
              //third_order_derivative.select(1, i).select(2, j).copy_(grad_H_p_ij);
              third_order_derivative.index_put_({Slice(), i, j}, grad_H_p_ij);
            }
        }
        grad_H_p_i.detach();
        }
    }
    auto xtc = xt.detach();
    auto ptc = pt.detach();
    auto Hvaluec = Hvalue.detach();
    auto third_order_derivativec = third_order_derivative.detach();
    // Return the third-order derivative tensor
    return third_order_derivativec;
}


template<typename T>
TensorDual evalDynsDual(const TensorDual &y, T W, std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, T)> H)
{
    int M = y.r.size(0);
    int N = y.r.size(1)/2; //The dual tensor y contains [p,x] in strict order
    auto p = y.index({Slice(), Slice(0, N)});
    auto x = y.index({Slice(), Slice(N, 2*N)});
    // Compute the Hamiltonian value
    auto Hvalue = H(x.r, p.r, W);

    // Compute the gradient of H with respect to x
    auto grad_H_x = pxH(x.r, p.r, W, H);

    // Compute the gradient of H with respect to p
    auto grad_H_p = ppH(x.r, p.r, W, H);
    //We need to calculate the dual parts

    
    /*auto dp0pxH = torch::einsum("mij, mjk->mik", {pxpxHval, dxdp0})+
                  //****Note here that the index wrt is i and the index wrt p is j****
                  torch::einsum("mij, mjk->mik", {pppxHval, dpdp0});  */
    auto pxpxHval = pxpxH(x.r, p.r, W, H);
    auto pppxHval = pppxH(x.r, p.r, W, H);
    auto ppppHval = ppppH(x.r, p.r, W, H);
    auto pxppHval = pxppH(x.r, p.r, W, H);

    auto grad_H_x_dual = torch::einsum("mij,mjk->mik", {pxpxHval, x.d})+
                         torch::einsum("mij, mjk->mik", {pppxHval, p.d});
    auto grad_H_p_dual = torch::einsum("mij,mjk->mik", {ppppHval, p.d})+
                         torch::einsum("mij, mjk->mik", {pxppHval, x.d});
    // Return the dynamics
    auto dotp = TensorDual(grad_H_x, grad_H_x_dual);
    auto dotx = TensorDual(grad_H_p, grad_H_p_dual);

    auto dyns = TensorDual::cat({dotp, dotx});
    return dyns;
}

/**
 * Dynamics for Hamiltonians with control inputs
 */
template<typename T>
TensorDual evalDynsUDual(const TensorDual &y,
                         const torch::Tensor &u, 
                         T W, 
                         std::function<torch::Tensor(const torch::Tensor&, 
                                                     const torch::Tensor&,
                                                     const torch::Tensor&, 
                                                     T)> Hu)
{
    int M = y.r.size(0);
    int N = y.r.size(1)/2; //The dual tensor y contains [p,x] in strict order
    auto p = y.index({Slice(), Slice(0, N)});
    auto x = y.index({Slice(), Slice(N, 2*N)});
    // Compute the Hamiltonian value
    auto Hvalue = Hu(x.r, p.r, u, W);

    // Compute the gradient of H with respect to x
    auto grad_H_x = pxHu(x.r, p.r, u, W, Hu);

    // Compute the gradient of H with respect to p
    auto grad_H_p = ppHu(x.r, p.r, u, W, Hu);
    //We need to calculate the dual parts

    
    /*auto dp0pxH = torch::einsum("mij, mjk->mik", {pxpxHval, dxdp0})+
                  //****Note here that the index wrt is i and the index wrt p is j****
                  torch::einsum("mij, mjk->mik", {pppxHval, dpdp0});  */
    auto pxpxHval = pxpxHu(x.r, p.r, u, W, Hu);
    auto pppxHval = pppxHu(x.r, p.r, u, W, Hu);
    auto ppppHval = ppppHu(x.r, p.r, u, W, Hu);
    auto pxppHval = pxppHu(x.r, p.r, u, W, Hu);

    auto grad_H_x_dual = torch::einsum("mij,mjk->mik", {pxpxHval, x.d})+
                         torch::einsum("mij, mjk->mik", {pppxHval, p.d});
    auto grad_H_p_dual = torch::einsum("mij,mjk->mik", {ppppHval, p.d})+
                         torch::einsum("mij, mjk->mik", {pxppHval, x.d});
    // Return the dynamics
    auto dotp = TensorDual(grad_H_x, grad_H_x_dual);
    auto dotx = TensorDual(grad_H_p, grad_H_p_dual);

    auto dyns = TensorDual::cat({dotp, dotx});
    return dyns;
}


template<typename T>
torch::Tensor evalDyns(const torch::Tensor &y, 
                    T W, 
                    std::function<torch::Tensor(const torch::Tensor&, const torch::Tensor&, T)> H)
{
    int M = y.size(0);
    int N = y.size(1)/2; //The tensor y contains [p,x] in strict order
    auto p = y.index({Slice(), Slice(0, N)});
    auto x = y.index({Slice(), Slice(N, 2*N)});
    // Compute the Hamiltonian value
    auto Hvalue = H(x, p, W);

    // Compute the gradient of H with respect to x
    auto grad_H_x = pxH(x, p, W, H);

    // Compute the gradient of H with respect to p
    auto grad_H_p = ppH(x, p, W, H);
    // Return the dynamics
    auto dotp = grad_H_x;
    auto dotx = grad_H_p;
    auto dyns = torch::zeros({M, 2*N}, y.options());
    //auto dyns = torch::cat({dotp, dotx},1);
    dyns.index_put_({Slice(), Slice(0, N)}, dotp);
    dyns.index_put_({Slice(), Slice(N, 2*N)}, dotx);
    return dyns.clone();
}

template<typename T>
torch::Tensor evalDynsU(const torch::Tensor &y,
                        const torch::Tensor &u,  
                        T W, 
                        std::function<torch::Tensor(const torch::Tensor&, 
                                                    const torch::Tensor&,
                                                    const torch::Tensor&, 
                                                    T)> Hu)
{
    int M = y.size(0);
    int N = y.size(1)/2; //The tensor y contains [p,x] in strict order
    auto p = y.index({Slice(), Slice(0, N)});
    auto x = y.index({Slice(), Slice(N, 2*N)});
    // Compute the Hamiltonian value
    auto Hvalue = Hu(x, p, u, W);

    // Compute the gradient of H with respect to x
    auto grad_H_x = pxHu(x, p, u, W, Hu);

    // Compute the gradient of H with respect to p
    auto grad_H_p = ppHu(x, p, u, W, Hu);
    // Return the dynamics
    auto dotp = grad_H_x;
    auto dotx = grad_H_p;
    auto dyns = torch::zeros({M, 2*N}, y.options());
    //auto dyns = torch::cat({dotp, dotx},1);
    dyns.index_put_({Slice(), Slice(0, N)}, dotp);
    dyns.index_put_({Slice(), Slice(N, 2*N)}, dotx);
    return dyns.clone();
}


template<typename T>
torch::Tensor evalJac(const torch::Tensor & y, T W, std::function<torch::Tensor(const torch::Tensor&, 
                                                                             const torch::Tensor&, 
                                                                             T)> H)
{
    int M = y.size(0);
    int N = y.size(1)/2; //The dual tensor y contains [p,x] in strict order
    auto p = y.index({Slice(), Slice(0, N)});
    auto x = y.index({Slice(), Slice(N, 2*N)});
    // Compute the Hamiltonian value
    auto Hvalue = H(x, p, W);

    
    /*auto dp0pxH = torch::einsum("mij, mjk->mik", {pxpxHval, dxdp0})+
                  //****Note here that the index wrt is i and the index wrt p is j****
                  torch::einsum("mij, mjk->mik", {pppxHval, dpdp0});  */
    auto pppxHval         = pppxH(x, p, W, H);
    torch::Tensor pp_dotp = pppxHval;

    auto pxpxHval = pxpxH(x, p, W, H);
    torch::Tensor px_dotp = pxpxHval;
    
    
    auto ppppHval = ppppH(x, p, W, H);
    torch::Tensor pp_dotx = ppppHval;

    auto pxppHval = pxppH(x, p, W, H);
    torch::Tensor px_dotx = pxppHval;



    auto jac1 = torch::cat({pp_dotp, px_dotp}, 2);
    auto jac2 = torch::cat({pp_dotx, px_dotx}, 2);
    auto jac = torch::cat({jac1, jac2}, 1);

    return jac;
}


template<typename T>
torch::Tensor evalJacU(const torch::Tensor & y,
                       const torch::Tensor &u,
                       T W, 
                       std::function<torch::Tensor(const torch::Tensor&, 
                                                   const torch::Tensor&, 
                                                   const torch::Tensor&,
                                                   T)> Hu)
{
    int M = y.size(0);
    int N = y.size(1)/2; //The dual tensor y contains [p,x] in strict order
    auto p = y.index({Slice(), Slice(0, N)});
    auto x = y.index({Slice(), Slice(N, 2*N)});
    // Compute the Hamiltonian value
    auto Hvalue = Hu(x, p, u, W);

    
    /*auto dp0pxH = torch::einsum("mij, mjk->mik", {pxpxHval, dxdp0})+
                  //****Note here that the index wrt is i and the index wrt p is j****
                  torch::einsum("mij, mjk->mik", {pppxHval, dpdp0});  */
    auto pppxHval         = pppxHu(x, p, u, W, Hu);
    torch::Tensor pp_dotp = pppxHval;

    auto pxpxHval = pxpxHu(x, p, u, W, Hu);
    torch::Tensor px_dotp = pxpxHval;
    
    
    auto ppppHval = ppppHu(x, p, u, W, Hu);
    torch::Tensor pp_dotx = ppppHval;

    auto pxppHval = pxppHu(x, p, u, W, Hu);
    torch::Tensor px_dotx = pxppHval;



    auto jac1 = torch::cat({pp_dotp, px_dotp}, 2);
    auto jac2 = torch::cat({pp_dotx, px_dotx}, 2);
    auto jac = torch::cat({jac1, jac2}, 1);

    return jac;
}


template<typename T>
TensorMatDual evalJacDual(const TensorDual & y, T W, std::function<torch::Tensor(const torch::Tensor&, 
                                                                             const torch::Tensor&, 
                                                                             T)> H)
{
    int M = y.r.size(0);
    int N = y.r.size(1)/2; //The dual tensor y contains [p,x] in strict order
    auto p = y.index({Slice(), Slice(0, N)});
    auto x = y.index({Slice(), Slice(N, 2*N)});
    // Compute the Hamiltonian value
    auto Hvalue = H(x.r, p.r, W);

    
    /*auto dp0pxH = torch::einsum("mij, mjk->mik", {pxpxHval, dxdp0})+
                  //****Note here that the index wrt is i and the index wrt p is j****
                  torch::einsum("mij, mjk->mik", {pppxHval, dpdp0});  */
    auto pppxHval   = pppxH(x.r, p.r, W, H);
    auto pppppxHval = pppppxH(x.r, p.r, W, H);
    auto pxpppxHval = pxpppxH(x.r, p.r, W, H);
    auto pppxHval_dual = torch::einsum("mijk,mkl->mijl", {pxpppxHval, x.d})+
                         torch::einsum("mijk,mkl->mijl", {pppppxHval, p.d});
    TensorMatDual pp_dotp = TensorMatDual(pppxHval, pppxHval_dual);

    auto pxpxHval = pxpxH(x.r, p.r, W, H);
    auto pppxpxHval = pppxpxH(x.r, p.r, W, H);
    auto pxpxpxHval = pxpxpxH(x.r, p.r, W, H);
    auto pxpxHval_dual = torch::einsum("mijk,mkl->mijl", {pxpxpxHval, x.d})+
                         torch::einsum("mijk,mkl->mijl", {pppxpxHval, p.d});

    TensorMatDual px_dotp = TensorMatDual(pxpxHval, pxpxHval_dual);
    
    
    auto ppppHval   = ppppH(x.r, p.r, W, H);
    auto ppppppHval = ppppppH(x.r, p.r, W, H);
    auto pxppppHval = pxppppH(x.r, p.r, W, H);
    auto ppppHval_dual = torch::einsum("mijk,mkl->mijl", {pxppppHval, x.d})+
                         torch::einsum("mijk,mkl->mijl", {ppppppHval, p.d});
    TensorMatDual pp_dotx = TensorMatDual(ppppHval, ppppHval_dual);




    auto pxppHval = pxppH(x.r, p.r, W, H);
    auto pppxppHval = pppxppH(x.r, p.r, W, H);
    auto pxpxppHval = pxpxppH(x.r, p.r, W, H);
    auto pxppHval_dual = torch::einsum("mijk,mkl->mijl", {pxpxppHval, x.d})+
                         torch::einsum("mijk, mkl->mijl", {pppxppHval, p.d});
    TensorMatDual px_dotx = TensorMatDual(pxppHval, pxppHval_dual);

    auto jac1 = TensorMatDual::cat(pp_dotp, px_dotp, 2);
    auto jac2 = TensorMatDual::cat(pp_dotx, px_dotx, 2);
    auto jac = TensorMatDual::cat(jac1, jac2, 1);

    return jac;
}

template<typename T>
TensorMatDual evalJacDualU(const TensorDual & y, 
                           const torch::Tensor &u,
                           T W, 
                           std::function<torch::Tensor(const torch::Tensor&, 
                                                       const torch::Tensor&,
                                                       const torch::Tensor&, 
                                                       T)> Hu)
{
    int M = y.r.size(0);
    int N = y.r.size(1)/2; //The dual tensor y contains [p,x] in strict order
    auto p = y.index({Slice(), Slice(0, N)});
    auto x = y.index({Slice(), Slice(N, 2*N)});
    // Compute the Hamiltonian value
    auto Hvalue = Hu(x.r, p.r, u, W);

    
    /*auto dp0pxH = torch::einsum("mij, mjk->mik", {pxpxHval, dxdp0})+
                  //****Note here that the index wrt is i and the index wrt p is j****
                  torch::einsum("mij, mjk->mik", {pppxHval, dpdp0});  */
    auto pppxHval   = pppxHu(x.r, p.r, u, W, Hu);
    auto pppppxHval = pppppxHu(x.r, p.r, u, W, Hu);
    auto pxpppxHval = pxpppxHu(x.r, p.r, u, W, Hu);
    auto pppxHval_dual = torch::einsum("mijk,mkl->mijl", {pxpppxHval, x.d})+
                         torch::einsum("mijk,mkl->mijl", {pppppxHval, p.d});
    TensorMatDual pp_dotp = TensorMatDual(pppxHval, pppxHval_dual);

    auto pxpxHval = pxpxHu(x.r, p.r, u, W, Hu);
    auto pppxpxHval = pppxpxHu(x.r, p.r, u, W, Hu);
    auto pxpxpxHval = pxpxpxHu(x.r, p.r, u,  W, Hu);
    auto pxpxHval_dual = torch::einsum("mijk,mkl->mijl", {pxpxpxHval, x.d})+
                         torch::einsum("mijk,mkl->mijl", {pppxpxHval, p.d});

    TensorMatDual px_dotp = TensorMatDual(pxpxHval, pxpxHval_dual);
    
    
    auto ppppHval   = ppppHu(x.r, p.r, u, W, Hu);
    auto ppppppHval = ppppppHu(x.r, p.r, u, W, Hu);
    auto pxppppHval = pxppppHu(x.r, p.r, u, W, Hu);
    auto ppppHval_dual = torch::einsum("mijk,mkl->mijl", {pxppppHval, x.d})+
                         torch::einsum("mijk,mkl->mijl", {ppppppHval, p.d});
    TensorMatDual pp_dotx = TensorMatDual(ppppHval, ppppHval_dual);




    auto pxppHval = pxppHu(x.r, p.r, u, W, Hu);
    auto pppxppHval = pppxppHu(x.r, p.r, u,  W, Hu);
    auto pxpxppHval = pxpxppHu(x.r, p.r, u, W, Hu);
    auto pxppHval_dual = torch::einsum("mijk,mkl->mijl", {pxpxppHval, x.d})+
                         torch::einsum("mijk, mkl->mijl", {pppxppHval, p.d});
    TensorMatDual px_dotx = TensorMatDual(pxppHval, pxppHval_dual);

    auto jac1 = TensorMatDual::cat(pp_dotp, px_dotp, 2);
    auto jac2 = TensorMatDual::cat(pp_dotx, px_dotx, 2);
    auto jac = TensorMatDual::cat(jac1, jac2, 1);

    return jac;
}

void print_N_Vector(N_Vector v) {
    // Get pointer to data
    sunindextype length = N_VGetLength(v);
    sunrealtype *data = N_VGetArrayPointer(v);

    if (data == NULL) {
        printf("Error: Unable to access N_Vector data.\n");
        return;
    }

    printf("N_Vector contents:\n");
    for (sunindextype i = 0; i < length; i++) {
        printf("%g ", data[i]);
    }
    printf("\n");
}


}  //Namespace janus

#endif