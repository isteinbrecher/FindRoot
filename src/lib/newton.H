
#pragma once

#include <array>
#include "math_utils.H"
#include <iostream>

namespace newton
{
  struct newton_data
  {
    unsigned int local_newton_iter_max = 20;
  };



  template <typename T>
  struct function
  {
    static inline bool check_convergence(newton_data& params) { return false; }
  };


  template <typename T>
  inline bool check_convergence(newton_data& params)
  {
    return true;
  };



  template <typename fun, typename T, typename... Args>
  inline void newton(newton_data& params, T* x, Args... args)
  {
    constexpr unsigned short n_dim = fun::n_dim;

    // Initialize variables for the Newton iteration
    T r[n_dim];
    T delta_x[n_dim];
    T jac[n_dim * n_dim];

    // Local Newton iteration.
    {
      check_convergence<fun>(params);
      std::cout << "\nconvergence: " << function<fun>::check_convergence();


      unsigned int counter = 0;
      while (counter < params.local_newton_iter_max)
      {
        fun::eval_f_jac(x, r, jac);

        std::cout << "\n res: " << newton::math::norm2<T, n_dim>(r);
        if (newton::math::norm2<T, n_dim>(r) < 1e-12)
        {
          std::cout << "\n counter: " << counter;
          break;
        }

        // Solve the linearized system.
        if (newton::math::SolveLinearSystem<T, n_dim>(jac, r, delta_x))
        {
          // Set the new parameter coordinates.
          for (unsigned short i = 0; i < n_dim; i++) x[i] -= delta_x[i];

          // Advance Newton iteration counter.
          counter++;
        }
        else
          break;
      }
    }
  };


}  // namespace newton
