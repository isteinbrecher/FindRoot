
#pragma once


#include <math.h>

#include "utils.H"


namespace FindRoot
{
  using namespace UTILS;

  namespace Math
  {
    template <unsigned short n_dim, typename T, typename T_mat>
    inline T GetLinearSystemScaleFactor(const T_mat& A)
    {
      // Scale each row so that the largest entry is one
      T scale_factor = 1.0;
      for (unsigned short i_row = 0; i_row < n_dim; i_row++)
      {
        T row_max_value = 0.0;
        for (unsigned short i_col = 0; i_col < n_dim; i_col++)
        {
          const T abs_entry = std::abs(GetM(A, i_row, i_col));
          if (abs_entry > row_max_value)
          {
            row_max_value = abs_entry;
          }
        }
        scale_factor *= row_max_value;
      }
      return scale_factor;
    }

    template <unsigned short n_dim, typename T, typename T_vec>
    inline T Norm2(const T_vec& r)
    {
      T norm = 0.0;
      for (unsigned short i = 0; i < n_dim; i++) norm += std::pow(GetV(r, i), 2);
      return std::sqrt(norm);
    }

    template <typename T>
    inline bool GetInvertedDeterminant(const T& det, const T& scaling_factor, T& det_inv)
    {
      if (scaling_factor == 0.0) return false;
      const T det_scaled = det / scaling_factor;
      if (std::abs(det_scaled) < 1e-12) return false;
      det_inv = 1.0 / det;
      return true;
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
      static inline bool SolveLinearSystem(const T_mat& A, const T_vec& b, T_vec& x)
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
      static inline T Determinant(const T_mat& A)
      {
        return GetM(A, 0, 0) * GetM(A, 1, 1) - GetM(A, 0, 1) * GetM(A, 1, 0);
      }

      template <typename T_mat, typename T_vec>
      static inline bool SolveLinearSystem(const T_mat& A, const T_vec& b, T_vec& x)
      {
        const T det = Determinant(A);
        const T scaling_factor = GetLinearSystemScaleFactor<n_dim, T>(A);
        T det_inv = 0.0;
        if (!GetInvertedDeterminant(det, scaling_factor, det_inv)) return false;
        GetV(x, 0) = (GetM(A, 1, 1) * GetV(b, 0) - GetM(A, 0, 1) * GetV(b, 1)) * det_inv;
        GetV(x, 1) = (-GetM(A, 1, 0) * GetV(b, 0) + GetM(A, 0, 0) * GetV(b, 1)) * det_inv;
        return true;
      }
    };

    template <typename T>
    struct LinearSolve<T, 3>
    {
      static constexpr unsigned int n_dim = 3;

      template <typename T_mat>
      static inline T Determinant(const T_mat& A)
      {
        return GetM(A, 0, 2) * (-GetM(A, 1, 1) * GetM(A, 2, 0) + GetM(A, 1, 0) * GetM(A, 2, 1)) +
               GetM(A, 0, 1) * (GetM(A, 1, 2) * GetM(A, 2, 0) - GetM(A, 1, 0) * GetM(A, 2, 2)) +
               GetM(A, 0, 0) * (-GetM(A, 1, 2) * GetM(A, 2, 1) + GetM(A, 1, 1) * GetM(A, 2, 2));
      }

      template <typename T_mat, typename T_vec>
      static inline bool SolveLinearSystem(const T_mat& A, const T_vec& b, T_vec& x)
      {
        const T det = Determinant(A);
        const T scaling_factor = GetLinearSystemScaleFactor<n_dim, T>(A);
        T det_inv = 0.0;
        if (!GetInvertedDeterminant(det, scaling_factor, det_inv)) return false;
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
    };

    template <typename T>
    struct LinearSolve<T, 4>
    {
      static constexpr unsigned int n_dim = 4;

      template <typename T_mat>
      static inline T Determinant(const T_mat& A)
      {
        return GetM(A, 0, 1) * GetM(A, 1, 3) * GetM(A, 2, 2) * GetM(A, 3, 0) -
               GetM(A, 0, 1) * GetM(A, 1, 2) * GetM(A, 2, 3) * GetM(A, 3, 0) -
               GetM(A, 0, 0) * GetM(A, 1, 3) * GetM(A, 2, 2) * GetM(A, 3, 1) +
               GetM(A, 0, 0) * GetM(A, 1, 2) * GetM(A, 2, 3) * GetM(A, 3, 1) -
               GetM(A, 0, 1) * GetM(A, 1, 3) * GetM(A, 2, 0) * GetM(A, 3, 2) +
               GetM(A, 0, 0) * GetM(A, 1, 3) * GetM(A, 2, 1) * GetM(A, 3, 2) +
               GetM(A, 0, 1) * GetM(A, 1, 0) * GetM(A, 2, 3) * GetM(A, 3, 2) -
               GetM(A, 0, 0) * GetM(A, 1, 1) * GetM(A, 2, 3) * GetM(A, 3, 2) +
               GetM(A, 0, 3) * (GetM(A, 1, 2) * (GetM(A, 2, 1) * GetM(A, 3, 0) - GetM(A, 2, 0) * GetM(A, 3, 1)) +
                                   GetM(A, 1, 1) * (-(GetM(A, 2, 2) * GetM(A, 3, 0)) + GetM(A, 2, 0) * GetM(A, 3, 2)) +
                                   GetM(A, 1, 0) * (GetM(A, 2, 2) * GetM(A, 3, 1) - GetM(A, 2, 1) * GetM(A, 3, 2))) +
               GetM(A, 0, 1) * GetM(A, 1, 2) * GetM(A, 2, 0) * GetM(A, 3, 3) -
               GetM(A, 0, 0) * GetM(A, 1, 2) * GetM(A, 2, 1) * GetM(A, 3, 3) -
               GetM(A, 0, 1) * GetM(A, 1, 0) * GetM(A, 2, 2) * GetM(A, 3, 3) +
               GetM(A, 0, 0) * GetM(A, 1, 1) * GetM(A, 2, 2) * GetM(A, 3, 3) +
               GetM(A, 0, 2) * (GetM(A, 1, 3) * (-(GetM(A, 2, 1) * GetM(A, 3, 0)) + GetM(A, 2, 0) * GetM(A, 3, 1)) +
                                   GetM(A, 1, 1) * (GetM(A, 2, 3) * GetM(A, 3, 0) - GetM(A, 2, 0) * GetM(A, 3, 3)) +
                                   GetM(A, 1, 0) * (-(GetM(A, 2, 3) * GetM(A, 3, 1)) + GetM(A, 2, 1) * GetM(A, 3, 3)));
      }

      template <typename T_mat, typename T_vec>
      static inline bool SolveLinearSystem(const T_mat& A, const T_vec& b, T_vec& x)
      {
        const T det = Determinant(A);
        const T scaling_factor = GetLinearSystemScaleFactor<n_dim, T>(A);
        T det_inv = 0.0;
        if (!GetInvertedDeterminant(det, scaling_factor, det_inv)) return false;
        GetV(x, 0) = -(GetM(A, 1, 1) * GetM(A, 2, 3) * GetM(A, 3, 2) * GetV(b, 0)) +
                     GetM(A, 1, 1) * GetM(A, 2, 2) * GetM(A, 3, 3) * GetV(b, 0) +
                     GetM(A, 0, 3) * GetM(A, 2, 2) * GetM(A, 3, 1) * GetV(b, 1) -
                     GetM(A, 0, 2) * GetM(A, 2, 3) * GetM(A, 3, 1) * GetV(b, 1) -
                     GetM(A, 0, 3) * GetM(A, 2, 1) * GetM(A, 3, 2) * GetV(b, 1) +
                     GetM(A, 0, 1) * GetM(A, 2, 3) * GetM(A, 3, 2) * GetV(b, 1) +
                     GetM(A, 0, 2) * GetM(A, 2, 1) * GetM(A, 3, 3) * GetV(b, 1) -
                     GetM(A, 0, 1) * GetM(A, 2, 2) * GetM(A, 3, 3) * GetV(b, 1) +
                     GetM(A, 0, 3) * GetM(A, 1, 1) * GetM(A, 3, 2) * GetV(b, 2) -
                     GetM(A, 0, 2) * GetM(A, 1, 1) * GetM(A, 3, 3) * GetV(b, 2) -
                     GetM(A, 0, 3) * GetM(A, 1, 1) * GetM(A, 2, 2) * GetV(b, 3) +
                     GetM(A, 0, 2) * GetM(A, 1, 1) * GetM(A, 2, 3) * GetV(b, 3) +
                     GetM(A, 1, 3) * ((GetM(A, 0, 2) * GetM(A, 3, 1) - GetM(A, 0, 1) * GetM(A, 3, 2)) * GetV(b, 2) +
                                         GetM(A, 2, 2) * (-(GetM(A, 3, 1) * GetV(b, 0)) + GetM(A, 0, 1) * GetV(b, 3)) +
                                         GetM(A, 2, 1) * (GetM(A, 3, 2) * GetV(b, 0) - GetM(A, 0, 2) * GetV(b, 3))) +
                     GetM(A, 1, 2) * ((-(GetM(A, 0, 3) * GetM(A, 3, 1)) + GetM(A, 0, 1) * GetM(A, 3, 3)) * GetV(b, 2) +
                                         GetM(A, 2, 3) * (GetM(A, 3, 1) * GetV(b, 0) - GetM(A, 0, 1) * GetV(b, 3)) +
                                         GetM(A, 2, 1) * (-(GetM(A, 3, 3) * GetV(b, 0)) + GetM(A, 0, 3) * GetV(b, 3)));
        GetV(x, 1) = GetM(A, 1, 0) * GetM(A, 2, 3) * GetM(A, 3, 2) * GetV(b, 0) -
                     GetM(A, 1, 0) * GetM(A, 2, 2) * GetM(A, 3, 3) * GetV(b, 0) -
                     GetM(A, 0, 3) * GetM(A, 2, 2) * GetM(A, 3, 0) * GetV(b, 1) +
                     GetM(A, 0, 2) * GetM(A, 2, 3) * GetM(A, 3, 0) * GetV(b, 1) +
                     GetM(A, 0, 3) * GetM(A, 2, 0) * GetM(A, 3, 2) * GetV(b, 1) -
                     GetM(A, 0, 0) * GetM(A, 2, 3) * GetM(A, 3, 2) * GetV(b, 1) -
                     GetM(A, 0, 2) * GetM(A, 2, 0) * GetM(A, 3, 3) * GetV(b, 1) +
                     GetM(A, 0, 0) * GetM(A, 2, 2) * GetM(A, 3, 3) * GetV(b, 1) -
                     GetM(A, 0, 3) * GetM(A, 1, 0) * GetM(A, 3, 2) * GetV(b, 2) +
                     GetM(A, 0, 2) * GetM(A, 1, 0) * GetM(A, 3, 3) * GetV(b, 2) +
                     GetM(A, 0, 3) * GetM(A, 1, 0) * GetM(A, 2, 2) * GetV(b, 3) -
                     GetM(A, 0, 2) * GetM(A, 1, 0) * GetM(A, 2, 3) * GetV(b, 3) +
                     GetM(A, 1, 3) * ((-(GetM(A, 0, 2) * GetM(A, 3, 0)) + GetM(A, 0, 0) * GetM(A, 3, 2)) * GetV(b, 2) +
                                         GetM(A, 2, 2) * (GetM(A, 3, 0) * GetV(b, 0) - GetM(A, 0, 0) * GetV(b, 3)) +
                                         GetM(A, 2, 0) * (-(GetM(A, 3, 2) * GetV(b, 0)) + GetM(A, 0, 2) * GetV(b, 3))) +
                     GetM(A, 1, 2) * ((GetM(A, 0, 3) * GetM(A, 3, 0) - GetM(A, 0, 0) * GetM(A, 3, 3)) * GetV(b, 2) +
                                         GetM(A, 2, 3) * (-(GetM(A, 3, 0) * GetV(b, 0)) + GetM(A, 0, 0) * GetV(b, 3)) +
                                         GetM(A, 2, 0) * (GetM(A, 3, 3) * GetV(b, 0) - GetM(A, 0, 3) * GetV(b, 3)));
        GetV(x, 2) = -(GetM(A, 1, 0) * GetM(A, 2, 3) * GetM(A, 3, 1) * GetV(b, 0)) +
                     GetM(A, 1, 0) * GetM(A, 2, 1) * GetM(A, 3, 3) * GetV(b, 0) +
                     GetM(A, 0, 3) * GetM(A, 2, 1) * GetM(A, 3, 0) * GetV(b, 1) -
                     GetM(A, 0, 1) * GetM(A, 2, 3) * GetM(A, 3, 0) * GetV(b, 1) -
                     GetM(A, 0, 3) * GetM(A, 2, 0) * GetM(A, 3, 1) * GetV(b, 1) +
                     GetM(A, 0, 0) * GetM(A, 2, 3) * GetM(A, 3, 1) * GetV(b, 1) +
                     GetM(A, 0, 1) * GetM(A, 2, 0) * GetM(A, 3, 3) * GetV(b, 1) -
                     GetM(A, 0, 0) * GetM(A, 2, 1) * GetM(A, 3, 3) * GetV(b, 1) +
                     GetM(A, 0, 3) * GetM(A, 1, 0) * GetM(A, 3, 1) * GetV(b, 2) -
                     GetM(A, 0, 1) * GetM(A, 1, 0) * GetM(A, 3, 3) * GetV(b, 2) -
                     GetM(A, 0, 3) * GetM(A, 1, 0) * GetM(A, 2, 1) * GetV(b, 3) +
                     GetM(A, 0, 1) * GetM(A, 1, 0) * GetM(A, 2, 3) * GetV(b, 3) +
                     GetM(A, 1, 3) * ((GetM(A, 0, 1) * GetM(A, 3, 0) - GetM(A, 0, 0) * GetM(A, 3, 1)) * GetV(b, 2) +
                                         GetM(A, 2, 1) * (-(GetM(A, 3, 0) * GetV(b, 0)) + GetM(A, 0, 0) * GetV(b, 3)) +
                                         GetM(A, 2, 0) * (GetM(A, 3, 1) * GetV(b, 0) - GetM(A, 0, 1) * GetV(b, 3))) +
                     GetM(A, 1, 1) * ((-(GetM(A, 0, 3) * GetM(A, 3, 0)) + GetM(A, 0, 0) * GetM(A, 3, 3)) * GetV(b, 2) +
                                         GetM(A, 2, 3) * (GetM(A, 3, 0) * GetV(b, 0) - GetM(A, 0, 0) * GetV(b, 3)) +
                                         GetM(A, 2, 0) * (-(GetM(A, 3, 3) * GetV(b, 0)) + GetM(A, 0, 3) * GetV(b, 3)));
        GetV(x, 3) = GetM(A, 1, 0) * GetM(A, 2, 2) * GetM(A, 3, 1) * GetV(b, 0) -
                     GetM(A, 1, 0) * GetM(A, 2, 1) * GetM(A, 3, 2) * GetV(b, 0) -
                     GetM(A, 0, 2) * GetM(A, 2, 1) * GetM(A, 3, 0) * GetV(b, 1) +
                     GetM(A, 0, 1) * GetM(A, 2, 2) * GetM(A, 3, 0) * GetV(b, 1) +
                     GetM(A, 0, 2) * GetM(A, 2, 0) * GetM(A, 3, 1) * GetV(b, 1) -
                     GetM(A, 0, 0) * GetM(A, 2, 2) * GetM(A, 3, 1) * GetV(b, 1) -
                     GetM(A, 0, 1) * GetM(A, 2, 0) * GetM(A, 3, 2) * GetV(b, 1) +
                     GetM(A, 0, 0) * GetM(A, 2, 1) * GetM(A, 3, 2) * GetV(b, 1) -
                     GetM(A, 0, 2) * GetM(A, 1, 0) * GetM(A, 3, 1) * GetV(b, 2) +
                     GetM(A, 0, 1) * GetM(A, 1, 0) * GetM(A, 3, 2) * GetV(b, 2) +
                     GetM(A, 0, 2) * GetM(A, 1, 0) * GetM(A, 2, 1) * GetV(b, 3) -
                     GetM(A, 0, 1) * GetM(A, 1, 0) * GetM(A, 2, 2) * GetV(b, 3) +
                     GetM(A, 1, 2) * ((-(GetM(A, 0, 1) * GetM(A, 3, 0)) + GetM(A, 0, 0) * GetM(A, 3, 1)) * GetV(b, 2) +
                                         GetM(A, 2, 1) * (GetM(A, 3, 0) * GetV(b, 0) - GetM(A, 0, 0) * GetV(b, 3)) +
                                         GetM(A, 2, 0) * (-(GetM(A, 3, 1) * GetV(b, 0)) + GetM(A, 0, 1) * GetV(b, 3))) +
                     GetM(A, 1, 1) * ((GetM(A, 0, 2) * GetM(A, 3, 0) - GetM(A, 0, 0) * GetM(A, 3, 2)) * GetV(b, 2) +
                                         GetM(A, 2, 2) * (-(GetM(A, 3, 0) * GetV(b, 0)) + GetM(A, 0, 0) * GetV(b, 3)) +
                                         GetM(A, 2, 0) * (GetM(A, 3, 2) * GetV(b, 0) - GetM(A, 0, 2) * GetV(b, 3)));
        for (unsigned short i = 0; i < n_dim; i++) GetV(x, i) = GetV(x, i) * det_inv;
        return true;
      }
    };
  }  // namespace Math
}  // namespace FindRoot
