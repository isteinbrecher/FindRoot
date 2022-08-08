
#pragma once


#include <math.h>

#include "utils.H"


namespace FindRoot
{
  using namespace UTILS;

  namespace Math
  {
    template <unsigned short n_dim, typename T>
    inline T GetLinearSystemScaleFactor(const T* A)
    {
      // Scale each row so that the largest entry is one
      T scale_factor = 1.0;
      for (unsigned short i_row = 0; i_row < n_dim; i_row++)
      {
        T max_value = 0.0;
        for (unsigned short i_col = 0; i_col < n_dim; i_col++)
        {
          if (std::abs(mat<n_dim>(A, i_row, i_col)) > max_value)
          {
            max_value = mat<n_dim>(A, i_row, i_col);
          }
        }
        scale_factor /= max_value;
      }
      return scale_factor;
    }

    template <unsigned short n_dim, typename T>
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
      static constexpr unsigned int n_dim = 1;

      template <typename T>
      static inline bool SolveLinearSystem(T* A, T* b, T* x)
      {
        if (std::abs(mat<n_dim>(A, 0, 0)) < 1e-12)
          return false;
        else
        {
          vec<n_dim, T>(x, 0) = vec<n_dim, T>(b, 0) / mat<n_dim>(A, 0, 0);
          return true;
        }
      }
    };

    template <>
    struct LinearSolve<2>
    {
      static constexpr unsigned int n_dim = 2;

      template <typename T>
      static inline T Determinant(const T* A)
      {
        return mat<n_dim>(A, 0, 0) * mat<n_dim>(A, 1, 1) -
               mat<n_dim>(A, 0, 1) * mat<n_dim>(A, 1, 0);
      }

      template <typename T>
      static inline bool SolveLinearSystem(T* A, T* b, T* x)
      {
        const T det = Determinant(A);
        const T det_scaled = det * GetLinearSystemScaleFactor<n_dim>(A);
        if (std::abs(det_scaled) < 1e-12)
          return false;
        else
        {
          const T det_inv = 1.0 / det;
          vec<n_dim, T>(x, 0) = (mat<n_dim>(A, 1, 1) * vec<n_dim, T>(b, 0) -
                                    mat<n_dim>(A, 0, 1) * vec<n_dim, T>(b, 1)) *
                                det_inv;
          vec<n_dim, T>(x, 1) = (-mat<n_dim>(A, 1, 0) * vec<n_dim, T>(b, 0) +
                                    mat<n_dim>(A, 0, 0) * vec<n_dim, T>(b, 1)) *
                                det_inv;
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
      static inline bool SolveLinearSystem(T* A, T* b, T* x)
      {
        const T det = Determinant(A);
        const T det_scaled = det * GetLinearSystemScaleFactor<n_dim>(A);
        if (std::abs(det_scaled) < 1e-12)
          return false;
        else
        {
          const T det_inv = 1.0 / det;
          vec<n_dim, T>(x, 0) =
              (mat<n_dim>(A, 0, 2) * mat<n_dim>(A, 2, 1) -
                  mat<n_dim>(A, 0, 1) * mat<n_dim>(A, 2, 2)) *
                  vec<n_dim, T>(b, 1) +
              mat<n_dim>(A, 1, 2) * (-mat<n_dim>(A, 2, 1) * vec<n_dim, T>(b, 0) +
                                        mat<n_dim>(A, 0, 1) * vec<n_dim, T>(b, 2)) +
              mat<n_dim>(A, 1, 1) * (mat<n_dim>(A, 2, 2) * vec<n_dim, T>(b, 0) -
                                        mat<n_dim>(A, 0, 2) * vec<n_dim, T>(b, 2));
          vec<n_dim, T>(x, 1) =
              (-mat<n_dim>(A, 0, 2) * mat<n_dim>(A, 2, 0) +
                  mat<n_dim>(A, 0, 0) * mat<n_dim>(A, 2, 2)) *
                  vec<n_dim, T>(b, 1) +
              mat<n_dim>(A, 1, 2) * (mat<n_dim>(A, 2, 0) * vec<n_dim, T>(b, 0) -
                                        mat<n_dim>(A, 0, 0) * vec<n_dim, T>(b, 2)) +
              mat<n_dim>(A, 1, 0) * (-mat<n_dim>(A, 2, 2) * vec<n_dim, T>(b, 0) +
                                        mat<n_dim>(A, 0, 2) * vec<n_dim, T>(b, 2));
          vec<n_dim, T>(x, 2) =
              (mat<n_dim>(A, 0, 1) * mat<n_dim>(A, 2, 0) -
                  mat<n_dim>(A, 0, 0) * mat<n_dim>(A, 2, 1)) *
                  vec<n_dim, T>(b, 1) +
              mat<n_dim>(A, 1, 1) * (-mat<n_dim>(A, 2, 0) * vec<n_dim, T>(b, 0) +
                                        mat<n_dim>(A, 0, 0) * vec<n_dim, T>(b, 2)) +
              mat<n_dim>(A, 1, 0) * (mat<n_dim>(A, 2, 1) * vec<n_dim, T>(b, 0) -
                                        mat<n_dim>(A, 0, 1) * vec<n_dim, T>(b, 2));
          for (unsigned short i = 0; i < n_dim; i++) x[i] = x[i] * det_inv;
          return true;
        }
      }
    };
  }  // namespace Math
}  // namespace FindRoot
