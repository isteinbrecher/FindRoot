
#pragma once


#include <math.h>
#include "find_root_access_traits.H"
#include "find_root_math_det_inv.H"


namespace FindRoot
{
  namespace Math
  {
    template <unsigned short n_dim, typename T, typename T_vec>
    inline auto Norm2(const T_vec& r)
    {
      using T_out = typename GetFloatingPointType<T>::type;
      T_out norm = 0.0;
      for (unsigned short i = 0; i < n_dim; i++) norm += std::pow(GetFloatingPointType<T>::Value(GetV(r, i)), 2);
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
          row_max_value = std::max(row_max_value, std::abs(GetFloatingPointType<T>::Value(GetM(A, i_row, i_col))));
        }
        scale_factor *= row_max_value;
      }
      return scale_factor;
    }

    template <unsigned short n_dim, typename T>
    struct LinearSolve
    {
      template <typename T_vec, typename T_mat>
      static inline bool SolveLinearSystem(const T_mat& A, const T_vec& b, T_vec& x, double eps = 1e-12)
      {
        const auto scaling_factor = GetLinearSystemScaleFactor<n_dim, T>(A);
        const T det = LinearSystem<n_dim, T>::Determinant(A);

        // Check if the two values are "normal" floating point values
        if (std::fpclassify(GetFloatingPointType<T>::Value(scaling_factor)) != FP_NORMAL or
            std::fpclassify(GetFloatingPointType<T>::Value(det)) != FP_NORMAL)
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
