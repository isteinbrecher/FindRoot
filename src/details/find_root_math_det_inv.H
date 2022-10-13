
#pragma once


#include <math.h>

#include "find_root_access_traits.H"


namespace FindRoot
{
  namespace Math
  {
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
  }  // namespace Math
}  // namespace FindRoot
