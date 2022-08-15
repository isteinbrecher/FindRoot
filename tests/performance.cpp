/**
 * \brief Test the local Netwon-Raphson algorithm..
 */

#include <vector>
#include "../src/lib/find_root.H"
#include <gtest/gtest.h>
#include "utils.H"


#include <chrono>
#include <random>

void PerformanceTest3x3(const unsigned int factor, const double expected_time)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-0.2, 0.2);
  std::array<double, 3> x;
  FindRoot::IterationParameters params;

  auto time_begin = std::chrono::system_clock::now();

  for (unsigned int i = 0; i < factor; i++)
  {
    init_3x3(x);
    x[0] += dis(gen);
    x[1] += dis(gen);
    x[2] += dis(gen);
    FindRoot::NewtonRaphson<Function3x3>(params, x);
    check_solution_3x3(x);
  }

  const std::chrono::duration<double> duration = std::chrono::system_clock::now() - time_begin;
  EXPECT_LT(duration.count(), expected_time);
}

TEST(nr_tests, test_3x3Performance) { PerformanceTest3x3(100000, 0.1); }