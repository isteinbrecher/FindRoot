
#pragma once


#include <math.h>

#include "utils.H"


namespace FindRoot
{
  using namespace UTILS;

  namespace Math
  {
    template <unsigned short n_dim, typename T, typename T_vec>
    inline auto Norm2(const T_vec& r)
    {
      using T_out = typename GetFloatingPointType<T>::type;
      T_out norm = 0.0;
      for (unsigned short i = 0; i < n_dim; i++) norm += std::pow(FindRoot::GetFloatingPointValue(GetV(r, i)), 2);
      return std::sqrt(norm);
    }

    // Calculate the inverse factor for the determinant if each row is scaled such that the largest
    // entry is 1.
    template <unsigned short n_dim, typename T, typename T_mat>
    inline auto GetLinearSystemScaleFactor(const T_mat& A)
    {
      using T_out = typename GetFloatingPointType<T>::type;
      T_out scale_factor = 1.0;
      for (unsigned short i_row = 0; i_row < n_dim; i_row++)
      {
        T_out row_max_value = 0.0;
        for (unsigned short i_col = 0; i_col < n_dim; i_col++)
        {
          row_max_value = std::max(row_max_value, std::abs(FindRoot::GetFloatingPointValue(GetM(A, i_row, i_col))));
        }
        scale_factor *= row_max_value;
      }
      return scale_factor;
    }

    template <unsigned short n_dim, typename T>
    struct LinearSystem
    {
    };


    template <typename T>
    struct LinearSystem<1, T>
    {
      static constexpr unsigned int n_dim = 1;

      template <typename T_mat>
      static inline T Determinant(const T_mat& A)
      {
        return GetM(A, 0, 0);
      }

      template <typename T_mat, typename T_vec>
      static inline void SolveNotScaling(const T_mat&, const T_vec& b, T_vec& x)
      {
        GetV(x, 0) = GetV(b, 0);
      }
    };

    template <typename T>
    struct LinearSystem<2, T>
    {
      static constexpr unsigned int n_dim = 2;

      template <typename T_mat>
      static inline T Determinant(const T_mat& A)
      {
        return GetM(A, 0, 0) * GetM(A, 1, 1) - GetM(A, 0, 1) * GetM(A, 1, 0);
      }

      template <typename T_mat, typename T_vec>
      static inline void SolveNotScaling(const T_mat& A, const T_vec& b, T_vec& x)
      {
        GetV(x, 0) = (GetM(A, 1, 1) * GetV(b, 0) - GetM(A, 0, 1) * GetV(b, 1));
        GetV(x, 1) = (-GetM(A, 1, 0) * GetV(b, 0) + GetM(A, 0, 0) * GetV(b, 1));
      }
    };

    template <typename T>
    struct LinearSystem<3, T>
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
      static inline void SolveNotScaling(const T_mat& A, const T_vec& b, T_vec& x)
      {
        GetV(x, 0) = (GetM(A, 0, 2) * GetM(A, 2, 1) - GetM(A, 0, 1) * GetM(A, 2, 2)) * GetV(b, 1) +
                     GetM(A, 1, 2) * (-GetM(A, 2, 1) * GetV(b, 0) + GetM(A, 0, 1) * GetV(b, 2)) +
                     GetM(A, 1, 1) * (GetM(A, 2, 2) * GetV(b, 0) - GetM(A, 0, 2) * GetV(b, 2));
        GetV(x, 1) = (-GetM(A, 0, 2) * GetM(A, 2, 0) + GetM(A, 0, 0) * GetM(A, 2, 2)) * GetV(b, 1) +
                     GetM(A, 1, 2) * (GetM(A, 2, 0) * GetV(b, 0) - GetM(A, 0, 0) * GetV(b, 2)) +
                     GetM(A, 1, 0) * (-GetM(A, 2, 2) * GetV(b, 0) + GetM(A, 0, 2) * GetV(b, 2));
        GetV(x, 2) = (GetM(A, 0, 1) * GetM(A, 2, 0) - GetM(A, 0, 0) * GetM(A, 2, 1)) * GetV(b, 1) +
                     GetM(A, 1, 1) * (-GetM(A, 2, 0) * GetV(b, 0) + GetM(A, 0, 0) * GetV(b, 2)) +
                     GetM(A, 1, 0) * (GetM(A, 2, 1) * GetV(b, 0) - GetM(A, 0, 1) * GetV(b, 2));
      }
    };

    template <typename T>
    struct LinearSystem<4, T>
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
      static inline void SolveNotScaling(const T_mat& A, const T_vec& b, T_vec& x)
      {
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
      }
    };

    template <unsigned short n_dim, typename T>
    struct LinearSolve
    {
      template <typename T_vec, typename T_mat>
      static inline bool SolveLinearSystem(const T_mat& A, const T_vec& b, T_vec& x, double eps = 1e-12)
      {
        const auto scaling_factor = GetLinearSystemScaleFactor<n_dim, T>(A);
        const T det = LinearSystem<n_dim, T>::Determinant(A);

        // Check if the two values are "normal" floating point values
        if (std::fpclassify(FindRoot::GetFloatingPointValue(scaling_factor)) != FP_NORMAL or
            std::fpclassify(FindRoot::GetFloatingPointValue(det)) != FP_NORMAL)
        {
          return false;
        }

        // Check if the scaled determinant is within a sensible range
        const T det_scaled = det / scaling_factor;
        if (std::abs(det_scaled) < eps)
        {
          return false;
        }

        // At this point the determinant is invertible
        const T det_inv = 1.0 / det;

        // Solve the linear system (without multiplying the inverse determinant)
        LinearSystem<n_dim, T>::SolveNotScaling(A, b, x);

        // Multiply with the inverse determinant
        for (unsigned short i_row = 0; i_row < n_dim; i_row++)
        {
          GetV(x, i_row) *= det_inv;
        }
        return true;
      }
    };
  }  // namespace Math
}  // namespace FindRoot
