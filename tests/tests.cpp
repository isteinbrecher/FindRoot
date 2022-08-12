/**
 * \brief Test the local Netwon-Raphson algorithm..
 */

#include "../src/lib/find_root.H"
#include <gtest/gtest.h>


constexpr double tol = 1e-12;

namespace FindRoot
{
  namespace UTILS
  {
    template <unsigned short n_dim>
    struct AccessMatrix<n_dim, std::array<double, n_dim * n_dim>>
    {
      inline static auto& Get(
          std::array<double, n_dim * n_dim>& A, const unsigned short i_row, const unsigned short i_col)
      {
        return A[i_row + n_dim * i_col];
      }
    };
  }  // namespace UTILS
}  // namespace FindRoot


struct Function2x2
{
  static constexpr unsigned short n_dim = 2;
  using T = double;
  using T_vec = std::array<double, n_dim>;
  using T_mat = std::array<double, n_dim * n_dim>;

  static void eval_f_jac(const T_vec& x, T_vec& f, T_mat& jac)
  {
    f[0] = x[0] * x[0] + x[1] + 2.0;
    f[1] = 6.0 * x[0] - x[1] * (x[0] + 1.0);
    jac[0] = 2.0 * x[0];
    jac[1] = 6.0 - x[1];
    jac[2] = 1.0;
    jac[3] = -1.0 - x[0];
  }
};


/**
 * \brief
 */
TEST(nr_tests, test_2x2)
{
  FindRoot::IterationParameters params;

  std::array<double, 2> x;
  x[0] = 5.0;
  x[1] = 1.0;
  FindRoot::IterationData data = FindRoot::NewtonRaphson<Function2x2>(params, x);

  EXPECT_NEAR(x[0], -0.25609874104976987, tol);
  EXPECT_NEAR(x[1], -2.0655865651672771, tol);
  EXPECT_EQ(data.iterations, 8);
}


struct Function3x3
{
  static constexpr unsigned short n_dim = 3;

  using T = double;
  using T_vec = std::array<double, n_dim>;
  using T_mat = std::array<double, n_dim * n_dim>;

  static void eval_f_jac(T_vec& x, T_vec& f, T_mat& jac)
  {
    f[0] = x[0] * x[1] - x[1] * x[2] + 1.0;
    f[1] = x[0] * x[1] * x[2] - 5.0;
    f[2] = x[0] + x[2] - 2;

    jac[0] = x[1];
    jac[1] = x[1] * x[2];
    jac[2] = 1.0;

    jac[3] = x[0] - x[2];
    jac[4] = x[0] * x[2];
    jac[5] = 0.0;

    jac[6] = -x[1];
    jac[7] = x[0] * x[1];
    jac[8] = 1.0;
  }
};


/**
 * \brief
 */
TEST(nr_tests, test_3x3)
{
  FindRoot::IterationParameters params;

  std::array<double, 3> x;
  x[0] = 1.0;
  x[1] = 1.0;
  x[2] = 1.0;
  FindRoot::IterationData data = FindRoot::NewtonRaphson<Function3x3>(params, x);

  EXPECT_NEAR(x[0], 0.90098048640721518, tol);
  EXPECT_NEAR(x[1], 5.0495097567963922, tol);
  EXPECT_NEAR(x[2], 1.0990195135927849, tol);
  EXPECT_EQ(data.iterations, 7);
}
