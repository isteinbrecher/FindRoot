
#pragma once


#include <math.h>

#include "utils.H"


namespace FindRoot
{
  using namespace UTILS;

  namespace Math
  {
    template <unsigned short n_dim, typename T, typename T_mat>
    inline T GetLinearSystemScaleFactor(T_mat& A)
    {
      // Scale each row so that the largest entry is one
      T inv_scale_factor = 1.0;
      for (unsigned short i_row = 0; i_row < n_dim; i_row++)
      {
        T max_value = 0.0;
        for (unsigned short i_col = 0; i_col < n_dim; i_col++)
        {
          if (std::abs(GetM(A, i_row, i_col)) > max_value)
          {
            max_value = GetM(A, i_row, i_col);
          }
        }
        inv_scale_factor *= max_value;
      }
      return 1.0 / inv_scale_factor;
    }

    template <unsigned short n_dim, typename T, typename T_vec>
    inline T Norm2(T_vec& r)
    {
      T norm = 0.0;
      for (unsigned short i = 0; i < n_dim; i++) norm += std::pow(GetV(r, i), 2);
      return std::sqrt(norm);
    }

    template <typename T, unsigned short n_dim>
    struct LinearSolve
    {
    };

    template <typename T>
    struct LinearSolve<T, 1>
    {
      static constexpr unsigned int n_dim = 1;

      template <typename T_mat, typename T_vec>
      static inline bool SolveLinearSystem(T_mat& A, T_vec& b, T_vec& x)
      {
        if (std::abs(GetM(A, 0, 0)) < 1e-12)
          return false;
        else
        {
          GetV(x, 0) = GetV(b, 0) / GetM(A, 0, 0);
          return true;
        }
      }
    };

    template <typename T>
    struct LinearSolve<T, 2>
    {
      static constexpr unsigned int n_dim = 2;

      template <typename T_mat>
      static inline T Determinant(T_mat& A)
      {
        return GetM(A, 0, 0) * GetM(A, 1, 1) - GetM(A, 0, 1) * GetM(A, 1, 0);
      }

      template <typename T_mat, typename T_vec>
      static inline bool SolveLinearSystem(T_mat& A, T_vec& b, T_vec& x)
      {
        const T det = Determinant(A);
        const T det_scaled = det * GetLinearSystemScaleFactor<n_dim, T>(A);
        if (std::abs(det_scaled) < 1e-12)
          return false;
        else
        {
          const T det_inv = 1.0 / det;
          GetV(x, 0) = (GetM(A, 1, 1) * GetV(b, 0) - GetM(A, 0, 1) * GetV(b, 1)) * det_inv;
          GetV(x, 1) = (-GetM(A, 1, 0) * GetV(b, 0) + GetM(A, 0, 0) * GetV(b, 1)) * det_inv;
          return true;
        }
      }
    };

    template <typename T>
    struct LinearSolve<T, 3>
    {
      static constexpr unsigned int n_dim = 3;

      template <typename T_mat>
      static inline T Determinant(T_mat& A)
      {
        return GetM(A, 0, 2) * (-GetM(A, 1, 1) * GetM(A, 2, 0) + GetM(A, 1, 0) * GetM(A, 2, 1)) +
               GetM(A, 0, 1) * (GetM(A, 1, 2) * GetM(A, 2, 0) - GetM(A, 1, 0) * GetM(A, 2, 2)) +
               GetM(A, 0, 0) * (-GetM(A, 1, 2) * GetM(A, 2, 1) + GetM(A, 1, 1) * GetM(A, 2, 2));
      }

      template <typename T_mat, typename T_vec>
      static inline bool SolveLinearSystem(T_mat& A, T_vec& b, T_vec& x)
      {
        const T det = Determinant(A);
        const T det_scaled = det * GetLinearSystemScaleFactor<n_dim, T>(A);
        if (std::abs(det_scaled) < 1e-12)
          return false;
        else
        {
          const T det_inv = 1.0 / det;
          GetV(x, 0) = (GetM(A, 0, 2) * GetM(A, 2, 1) - GetM(A, 0, 1) * GetM(A, 2, 2)) * GetV(b, 1) +
                       GetM(A, 1, 2) * (-GetM(A, 2, 1) * GetV(b, 0) + GetM(A, 0, 1) * GetV(b, 2)) +
                       GetM(A, 1, 1) * (GetM(A, 2, 2) * GetV(b, 0) - GetM(A, 0, 2) * GetV(b, 2));
          GetV(x, 1) = (-GetM(A, 0, 2) * GetM(A, 2, 0) + GetM(A, 0, 0) * GetM(A, 2, 2)) * GetV(b, 1) +
                       GetM(A, 1, 2) * (GetM(A, 2, 0) * GetV(b, 0) - GetM(A, 0, 0) * GetV(b, 2)) +
                       GetM(A, 1, 0) * (-GetM(A, 2, 2) * GetV(b, 0) + GetM(A, 0, 2) * GetV(b, 2));
          GetV(x, 2) = (GetM(A, 0, 1) * GetM(A, 2, 0) - GetM(A, 0, 0) * GetM(A, 2, 1)) * GetV(b, 1) +
                       GetM(A, 1, 1) * (-GetM(A, 2, 0) * GetV(b, 0) + GetM(A, 0, 0) * GetV(b, 2)) +
                       GetM(A, 1, 0) * (GetM(A, 2, 1) * GetV(b, 0) - GetM(A, 0, 1) * GetV(b, 2));
          for (unsigned short i = 0; i < n_dim; i++) GetV(x, i) = GetV(x, i) * det_inv;
          return true;
        }
      }
    };
  }  // namespace Math
}  // namespace FindRoot
