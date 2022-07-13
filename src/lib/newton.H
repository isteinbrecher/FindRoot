
#pragma once

#include <array>
#include "math_utils.H"
#include <iostream>

namespace newton
{
  template <typename T>
  struct function
  {
    static inline bool check_convergence() { return false; }
  };



  template <typename fun, unsigned short n_dim, typename T, typename... Args>
  inline void newton(T* x, Args... args)
  {
    constexpr unsigned short local_newton_iter_max = 20;

    T r[n_dim];
    T delta_x[n_dim];
    T jac[n_dim * n_dim];


    std::cout << "\nconvergence: " << function<fun>::check_convergence();

    // Local Newton iteration.
    {
      unsigned int counter = 0;
      while (counter < local_newton_iter_max)
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
