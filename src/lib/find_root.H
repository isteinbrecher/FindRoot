#pragma once

#include "utils.H"
#include "math_utils.H"

namespace FindRoot
{

  struct IterationParameters
  {
    unsigned int max_iterations = 20;
    double tol_f = 1e-12;
    double tol_x = 1e-12;
    bool min_one_iteration = false;
  };

  struct IterationData
  {
    bool error = false;
    bool converged = false;
    unsigned int iterations = 0;
  };

  template <typename fun, typename T_vec, typename... Args>
  inline IterationData NewtonRaphson(const IterationParameters& params, T_vec& x, Args... args)
  {
    // Shortcut for system size
    constexpr unsigned short n_dim = fun::n_dim;

    // Shortcuts for types.
    using T = typename fun::T;
    using T_mat = typename fun::T_mat;

    // Initialize variables for the Newton iteration
    IterationData return_data;
    T_vec r;
    T_vec delta_x;
    T_mat jac;

    // delta_x has to be initialized, otherwise there are compiler warnings
    for (unsigned short i = 0; i < n_dim; i++) GetV(delta_x, i) = GetFloatingPointType<T>::max;

    // Local Newton iteration
    unsigned int counter = 0;
    while (counter < params.max_iterations)
    {
      // Evaluate the function and the Jacobian
      fun::eval_f_jac(x, r, jac, args...);

      // Check for convergence
      if (params.min_one_iteration && counter == 0)
      {
        // In this case we want to perform at least a single Newton step, no need to check for
        // convergence
      }
      if (Math::Norm2<n_dim, T>(r) < params.tol_f && Math::Norm2<n_dim, T>(delta_x) < params.tol_x)
      {
        return_data.converged = true;
        break;
      }

      // Solve the linear system
      if (Math::LinearSolve<n_dim, T>::SolveLinearSystem(jac, r, delta_x))
      {
        // Update the state vector and advance Newton iteration counter.
        for (unsigned short i = 0; i < n_dim; i++) GetV(x, i) -= GetV(delta_x, i);
        counter++;
      }
      else
      {
        return_data.error = true;
        break;
      }
    }

    return_data.iterations = counter;
    return return_data;
  };

}  // namespace FindRoot
