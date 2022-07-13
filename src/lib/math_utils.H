
#pragma once


#include <math.h>

namespace FindRoot
{
  namespace Math
  {
    template <typename T, unsigned short n_dim>
    bool SolveLinearSystem(const T* A, const T* b, T* x)
    {
      return false;
    }

    template <>
    bool SolveLinearSystem<double, 1>(const double* A, const double* b, double* x)
    {
      x[0] = b[0] / A[0];
      return true;
    }


    template <typename T, unsigned short n_dim>
    T Norm2(const T* r)
    {
      T norm = 0.0;
      for (unsigned short i = 0; i < n_dim; i++) norm += std::pow(r[i], 2);
      return std::sqrt(norm);
    }
  }  // namespace Math
}  // namespace FindRoot
