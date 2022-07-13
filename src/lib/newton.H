
#pragma once

#include <array>
#include "math_utils.H"
#include <iostream>

namespace newton
{

  template <typename fun, typename T>
  inline void newton(T x[fun::n_dim])
  {
    constexpr unsigned short n_dim = fun::n_dim;
    constexpr unsigned short local_newton_iter_max = 20;

    T r[n_dim];
    T delta_x[n_dim];
    T jac[n_dim * n_dim];

    // Local Newton iteration.
    {
      unsigned int counter = 0;
      while (counter < local_newton_iter_max)
      {
        fun::eval_f_jac(x, r, jac);

        if (newton::math::norm2<T, n_dim>(r) < 1e-12)
        {
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
