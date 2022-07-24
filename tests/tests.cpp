/**
 * \brief Test the local Netwon-Raphson algorithm..
 */

#include "../src/lib/newton.H"
#include <gtest/gtest.h>


constexpr double tol = 1e-12;

/**
 * \brief
 */
TEST(nr_tests, test_2x2)
{
  // const std::vector<char> input = {0, 16, -125, 16, 81, -121, 32, -110, -117, 48, -45, -113, 65,
  // 20,
  //     -109, 81, 85, -105, 97, -106, -101, 113, -41, -97, -126, 24, -93, -110, 89, -89, -94, -102,
  //     -85, -78, -37, -81, -61, 28, -77, -45, 93, -73, -29, -98, -69, -13, -33, -65, 0, 16, -125,
  //     16, 81, -121, 32, -110, -117, 48, -45, -113, 65, 20, -109, 81, 85, -105, 97, -106, -101,
  //     113, -41, -97, -126, 24, -93, -110, 89, -89, -94, -102, -85, -78, -37, -81, -61, 28, -77,
  //     -45, 93, -73, -29, -98, -69, -13, -33, -65};
  // const auto encoded = base64::encode(input.data(), input.size());
  // EXPECT_EQ(
  //     "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
  //     "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/",
  //     encoded);
  // const auto decoded = base64::decode(encoded);
  // EXPECT_EQ(input, decoded);
}

struct Function3x3
{
  static constexpr unsigned short n_dim = 3;

  static void eval_f_jac(const double x[n_dim], double f[n_dim], double jac[n_dim * n_dim])
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
  FindRoot::IterationData data = FindRoot::NewtonRaphson<Function3x3>(params, x.data());

  EXPECT_NEAR(x[0], 0.90098048640721518, tol);
  EXPECT_NEAR(x[1], 5.0495097567963922, tol);
  EXPECT_NEAR(x[2], 1.0990195135927849, tol);
}
