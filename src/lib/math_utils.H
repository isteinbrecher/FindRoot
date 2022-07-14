
#pragma once


#include <math.h>

namespace FindRoot
{
  namespace Math
  {
    template <unsigned short n_dim, typename T>
    inline T mat(const T* A, const unsigned short i_row, const unsigned short i_col)
    {
      // Default is Fortran type access
      return A[i_row + n_dim * i_col];
    }

    template <typename T, unsigned short n_dim>
    inline T Norm2(const T* r)
    {
      T norm = 0.0;
      for (unsigned short i = 0; i < n_dim; i++) norm += std::pow(r[i], 2);
      return std::sqrt(norm);
    }

    template <unsigned short n_dim>
    struct LinearSolve
    {
    };

    template <>
    struct LinearSolve<1>
    {
      template <typename T>
      static inline bool SolveLinearSystem(const T* A, const T* b, T* x)
      {
        if (std::abs(A[0]) < 1e-12)
          return false;
        else
        {
          x[0] = b[0] / A[0];
          return true;
        }
      }
    };

    template <>
    struct LinearSolve<2>
    {
      template <typename T>
      static inline T Determinant(const T* A)
      {
        return A[0] * A[3] - A[1] * A[2];
      }

      template <typename T>
      static inline bool SolveLinearSystem(const T* A, const T* b, T* x)
      {
        T det = Determinant(A);
        if (std::abs(det) < 1e-12)
          return false;
        else
        {
          const T det_inv = 1.0 / det;
          x[0] = (A[4] * b[0] - A[2] * b[1]) * det_inv;
          x[1] = (-A[1] * b[0] + A[0] * b[1]) * det_inv;
          return true;
        }
      }
    };

    template <>
    struct LinearSolve<3>
    {
      static constexpr unsigned int n_dim = 3;

      template <typename T>
      static inline T Determinant(const T* A)
      {
        return mat<n_dim>(A, 0, 2) * (-mat<n_dim>(A, 1, 1) * mat<n_dim>(A, 2, 0) +
                                         mat<n_dim>(A, 1, 0) * mat<n_dim>(A, 2, 1)) +
               mat<n_dim>(A, 0, 1) * (mat<n_dim>(A, 1, 2) * mat<n_dim>(A, 2, 0) -
                                         mat<n_dim>(A, 1, 0) * mat<n_dim>(A, 2, 2)) +
               mat<n_dim>(A, 0, 0) * (-mat<n_dim>(A, 1, 2) * mat<n_dim>(A, 2, 1) +
                                         mat<n_dim>(A, 1, 1) * mat<n_dim>(A, 2, 2));
      }

      template <typename T>
      static inline bool SolveLinearSystem(const T* A, const T* b, T* x)
      {
        T det = Determinant(A);
        if (std::abs(det) < 1e-12)
          return false;
        else
        {
          const T det_inv = 1.0 / det;
          x[0] = (mat<n_dim>(A, 0, 2) * mat<n_dim>(A, 2, 1) -
                     mat<n_dim>(A, 0, 1) * mat<n_dim>(A, 2, 2)) *
                     b[1] +
                 mat<n_dim>(A, 1, 2) * (-mat<n_dim>(A, 2, 1) * b[0] + mat<n_dim>(A, 0, 1) * b[2]) +
                 mat<n_dim>(A, 1, 1) * (mat<n_dim>(A, 2, 2) * b[0] - mat<n_dim>(A, 0, 2) * b[2]);
          x[1] = (-mat<n_dim>(A, 0, 2) * mat<n_dim>(A, 2, 0) +
                     mat<n_dim>(A, 0, 0) * mat<n_dim>(A, 2, 2)) *
                     b[1] +
                 mat<n_dim>(A, 1, 2) * (mat<n_dim>(A, 2, 0) * b[0] - mat<n_dim>(A, 0, 0) * b[2]) +
                 mat<n_dim>(A, 1, 0) * (-mat<n_dim>(A, 2, 2) * b[0] + mat<n_dim>(A, 0, 2) * b[2]);
          x[2] = (mat<n_dim>(A, 0, 1) * mat<n_dim>(A, 2, 0) -
                     mat<n_dim>(A, 0, 0) * mat<n_dim>(A, 2, 1)) *
                     b[1] +
                 mat<n_dim>(A, 1, 1) * (-mat<n_dim>(A, 2, 0) * b[0] + mat<n_dim>(A, 0, 0) * b[2]) +
                 mat<n_dim>(A, 1, 0) * (mat<n_dim>(A, 2, 1) * b[0] - mat<n_dim>(A, 0, 1) * b[2]);
          for (unsigned short i = 0; i < n_dim; i++) x[i] = x[i] * det_inv;
          return true;
        }
      }
    };
  }  // namespace Math
}  // namespace FindRoot
