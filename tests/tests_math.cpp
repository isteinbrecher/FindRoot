/**
 * \brief Test some math functions.
 */

#include <array>
#include "../src/lib/math_utils.H"
#include <gtest/gtest.h>

#include "utils.H"


using scalar_t = double;
template <unsigned short n_dim>
using vector_t = typename std::array<scalar_t, n_dim>;
template <unsigned short n_dim>
using matrix_t = typename std::array<std::array<scalar_t, n_dim>, n_dim>;


template <unsigned short n_dim>
void get_linear_solve_data(matrix_t<n_dim>& A, vector_t<n_dim>& b, vector_t<n_dim>& x, scalar_t& det)
{
}

template <>
void get_linear_solve_data<2>(matrix_t<2>& A, vector_t<2>& b, vector_t<2>& x, scalar_t& det)
{
  constexpr unsigned int n_dim = 2;
  using T_vec = vector_t<n_dim>;
  using T_mat = matrix_t<n_dim>;

  GetM(A, 0, 0) = -0.10615727466752567;
  GetM(A, 0, 1) = -0.14160992102647585;
  GetM(A, 1, 0) = -0.9443651110542777;
  GetM(A, 1, 1) = 0.6997375460212294;
  GetV(b, 0) = -0.03113166628149644;
  GetV(b, 1) = -0.7623759042430871;

  GetV(x, 0) = 0.6237280889135993;
  GetV(x, 1) = -0.2477341101305778;

  det = -0.2080136996647114;
}


template <>
void get_linear_solve_data<3>(matrix_t<3>& A, vector_t<3>& b, vector_t<3>& x, scalar_t& det)
{
  constexpr unsigned int n_dim = 3;
  using T_vec = vector_t<n_dim>;
  using T_mat = matrix_t<n_dim>;

  GetM(A, 0, 0) = -0.36911970151990614;
  GetM(A, 0, 1) = -0.8916737898122307;
  GetM(A, 0, 2) = -0.6382048551478618;
  GetM(A, 1, 0) = 0.8100067165245184;
  GetM(A, 1, 1) = -0.27112522751058155;
  GetM(A, 1, 2) = -0.08995039455166554;
  GetM(A, 2, 0) = -0.7654076843449245;
  GetM(A, 2, 1) = 0.8919211016464259;
  GetM(A, 2, 2) = 0.2679893153861612;

  GetV(b, 0) = 0.6451121136737776;
  GetV(b, 1) = -0.4783899250893686;
  GetV(b, 2) = 0.8888017769104009;

  GetV(x, 0) = -0.4033651349402938;
  GetV(x, 1) = 1.5235482083401302;
  GetV(x, 2) = -2.906167330744681;

  det = -0.19926408748766503;
}


template <>
void get_linear_solve_data<4>(matrix_t<4>& A, vector_t<4>& b, vector_t<4>& x, scalar_t& det)
{
  constexpr unsigned int n_dim = 4;
  using T_vec = vector_t<n_dim>;
  using T_mat = matrix_t<n_dim>;

  GetM(A, 0, 0) = 0.9251550125915169;
  GetM(A, 0, 1) = 0.2661668312122867;
  GetM(A, 0, 2) = -0.6580087706866702;
  GetM(A, 0, 3) = -0.06248620475302191;
  GetM(A, 1, 0) = -0.8907841114889101;
  GetM(A, 1, 1) = -0.8058921155342684;
  GetM(A, 1, 2) = 0.823626211382376;
  GetM(A, 1, 3) = 0.33905181036807264;
  GetM(A, 2, 0) = 0.4725871655508871;
  GetM(A, 2, 1) = -0.247143252851874;
  GetM(A, 2, 2) = -0.951797166281465;
  GetM(A, 2, 3) = 0.8230594482991904;
  GetM(A, 3, 0) = 0.5044816731186019;
  GetM(A, 3, 1) = 0.6379008472443242;
  GetM(A, 3, 2) = -0.7531489653912398;
  GetM(A, 3, 3) = -0.2295457986525351;
  GetV(b, 0) = -0.3523821168532093;
  GetV(b, 1) = 0.5851054670001141;
  GetV(b, 2) = -0.45672024673513656;
  GetV(b, 3) = -0.3683193206203468;

  GetV(x, 0) = -0.4609895307213335;
  GetV(x, 1) = -1.2526086442790891;
  GetV(x, 2) = -0.5010081582495448;
  GetV(x, 3) = -1.2457112707775218;

  det = -0.05446029862726158;
}


template <unsigned short n_dim>
void check_solution()
{
  using T_vec = vector_t<n_dim>;
  using T_mat = matrix_t<n_dim>;

  T_vec b, x, x_ref;
  T_mat A;
  scalar_t det, det_ref;
  get_linear_solve_data<n_dim>(A, b, x_ref, det_ref);

  det = FindRoot::Math::LinearSolve<scalar_t, n_dim>::Determinant(A);
  const bool result = FindRoot::Math::LinearSolve<scalar_t, n_dim>::SolveLinearSystem(A, b, x);

  for (unsigned short i_dim = 0; i_dim < n_dim; i_dim++) EXPECT_NEAR(GetV(x, i_dim), GetV(x_ref, i_dim), tol);
  EXPECT_NEAR(det, det_ref, tol);
  EXPECT_TRUE(result);
}


/**
 * \brief
 */
TEST(nr_tests, test_linear_solve_2x2) { check_solution<2>(); }
TEST(nr_tests, test_linear_solve_3x3) { check_solution<3>(); }
TEST(nr_tests, test_linear_solve_4x4) { check_solution<4>(); }
