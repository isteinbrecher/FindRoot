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
  };

  struct IterationData
  {
    bool error = false;
    bool converged = false;
    unsigned int iterations = 0;
  };

  template <typename fun, typename T_vec, typename... Args>
  inline IterationData NewtonRaphson(IterationParameters& params, T_vec& x, Args... args)
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
    UTILS::FillVector<n_dim, T_vec>::Fill(delta_x, 1e100);
    T_mat jac;

    // Local Newton iteration
    unsigned int counter = 0;
    while (counter < params.max_iterations)
    {
      // Evaluate the function and the Jacobian
      fun::eval_f_jac(x, r, jac, args...);

      // Check for convergence
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
