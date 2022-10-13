
#pragma once


#include <type_traits>
#include <limits>

// Shotcuts to access vectors and matrices
#define GetV(v, i) (FindRoot::AV<n_dim, T_vec>::Get(v, i))
#define GetM(A, i, j) (FindRoot::AM<n_dim, T_mat>::Get(A, i, j))


namespace FindRoot
{
  namespace UTILS
  {
    template <unsigned short n_dim, typename T_vec>
    struct AccessVector
    {
      inline static auto& Get(T_vec& v, const unsigned short i_row) { return v[i_row]; }
    };

    template <unsigned short n_dim, typename T_mat>
    struct AccessMatrix
    {
      inline static auto& Get(T_mat& A, const unsigned short i_row, const unsigned short i_col)
      {
        return A[i_row][i_col];
      }
    };

    template <unsigned short n_dim, typename T_vec>
    struct FillVector
    {
      template <typename T>
      inline static void Fill(T_vec& v, const T value)
      {
        for (unsigned short i_row = 0; i_row < n_dim; i_row++)
        {
          AccessVector<n_dim, T_vec>::Get(v, i_row) = value;
        }
      }
    };

    template <unsigned short n_dim, typename T_mat>
    struct InitializeMatrix
    {
      inline static auto& Init() { return T_mat(); }
    };


    template <bool is_floating_point, typename T>
    struct FloatingPoint
    {
    };

    // For c++ float types
    // This has to be overloaded for other scalar types, e.g., automatic differentiation types
    template <typename T>
    struct FloatingPoint<true, T>
    {
      using type = T;
      inline static const T& Value(const T& value) { return value; }
    };

  }  // namespace UTILS

  template <typename T>
  struct GetFloatingPointType
  {
    using type = typename UTILS::FloatingPoint<std::is_floating_point<T>::value, T>::type;

    const static auto max = std::numeric_limits<type>::max();

    static inline auto Value(const T& value)
    {
      return UTILS::FloatingPoint<std::is_floating_point<T>::value, T>::Value(value);
    }
  };

  template <unsigned short n_dim, typename T_vec>
  struct AV
  {
    inline static const auto& Get(const T_vec& v, const unsigned short i_row)
    {
      return FindRoot::UTILS::AccessVector<n_dim, T_vec>::Get(const_cast<T_vec&>(v), i_row);
    }

    inline static auto& Get(T_vec& v, const unsigned short i_row)
    {
      return FindRoot::UTILS::AccessVector<n_dim, T_vec>::Get(v, i_row);
    }
  };

  template <unsigned short n_dim, typename T_mat>
  struct AM
  {
    inline static const auto& Get(const T_mat& A, const unsigned short i_row, const unsigned short i_col)
    {
      return FindRoot::UTILS::AccessMatrix<n_dim, T_mat>::Get(const_cast<T_mat&>(A), i_row, i_col);
    }

    inline static auto& Get(T_mat& A, const unsigned short i_row, const unsigned short i_col)
    {
      return FindRoot::UTILS::AccessMatrix<n_dim, T_mat>::Get(A, i_row, i_col);
    }
  };

}  // namespace FindRoot
