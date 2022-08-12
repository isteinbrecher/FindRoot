
#pragma once


// Shotcuts to access vectors and matrices
#define GetV(v, i) (AccessVector<n_dim, T_vec>::Get(v, i))
#define GetM(A, i, j) (AccessMatrix<n_dim, T_mat>::Get(A, i, j))


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
    struct InitializeVector
    {
      inline static auto& Init() { return T_vec(); }
    };

    template <unsigned short n_dim, typename T_mat>
    struct InitializeMatrix
    {
      inline static auto& Init() { return T_mat(); }
    };
  }  // namespace UTILS
}  // namespace FindRoot