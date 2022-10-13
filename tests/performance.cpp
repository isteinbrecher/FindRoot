/**
 * \brief Test the local Netwon-Raphson algorithm..
 */

#include <vector>
#include "../src/find_root.H"
#include <gtest/gtest.h>
#include "utils.H"


#include <chrono>
#include <random>


template <typename fun, typename T_init, typename T_check>
void PerformanceTest(
    T_init fun_init, T_check fun_check, const unsigned int factor, const double expected_time, const double var = 0.2)
{
  constexpr unsigned short n_dim = fun::n_dim;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-var, var);
  std::array<double, n_dim> x;
  FindRoot::IterationParameters params;

  auto time_begin = std::chrono::system_clock::now();

  for (unsigned int i = 0; i < factor; i++)
  {
    fun_init(x);
    for (unsigned char j = 0; j < n_dim; j++) x[j] += dis(gen);
    FindRoot::NewtonRaphson<fun>(params, x);
    fun_check(x);
  }

  const std::chrono::duration<double> duration = std::chrono::system_clock::now() - time_begin;
  EXPECT_LT(duration.count(), expected_time);
}


TEST(nr_tests, test_1x1_performance)
{
  PerformanceTest<Function1x1>(
      init_1x1<std::array<double, 1>>, check_solution_1x1_no_iterations<std::array<double, 1>>, 60000, 0.1);
}
TEST(nr_tests, test_2x2_performance)
{
  PerformanceTest<Function2x2>(
      init_2x2<std::array<double, 2>>, check_solution_2x2_no_iterations<std::array<double, 2>>, 130000, 0.1);
}
TEST(nr_tests, test_3x3_performance)
{
  PerformanceTest<Function3x3>(
      init_3x3<std::array<double, 3>>, check_solution_3x3_no_iterations<std::array<double, 3>>, 100000, 0.1);
}