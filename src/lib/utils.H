
#pragma once


namespace FindRoot
{
  namespace UTILS
  {
    template <unsigned short n_dim, typename T>
    inline T& mat(T* A, const unsigned short i_row, const unsigned short i_col)
    {
      // Default is Fortran type access
      return A[i_row + n_dim * i_col];
    }

    template <unsigned short n_dim, typename T, typename V>
    inline T& vec(V v, const unsigned short i_row)
    {
      return v[i_row];
    }
  }  // namespace UTILS
}  // namespace FindRoot
