/**
 * \brief Test the local Netwon-Raphson algorithm..
 */

#include <vector>
#include "../src/find_root.H"
#include <gtest/gtest.h>


#include "utils.H"



template <unsigned short n_dim>
struct FindRoot::UTILS::AccessVector<n_dim, Vector>
{
  inline static auto& Get(Vector& A, const unsigned short i_row) { return A.data_[i_row]; }
};

template <unsigned short n_dim>
struct FindRoot::UTILS::AccessMatrix<n_dim, Vector>
{
  inline static auto& Get(Vector& A, const unsigned short i_row, const unsigned short i_col)
  {
    return A.data_[i_row + n_dim * i_col];
  }
};


template <unsigned short n_dim>
struct FindRoot::UTILS::AccessMatrix<n_dim, std::array<double, n_dim * n_dim>>
{
  inline static auto& Get(std::array<double, n_dim * n_dim>& A, const unsigned short i_row, const unsigned short i_col)
  {
    return A[i_row + n_dim * i_col];
  }
};



template <typename fun>
void Test1x1()
{
  FindRoot::IterationParameters params;

  std::array<double, 1> x;
  init_1x1(x);
  FindRoot::IterationData data = FindRoot::NewtonRaphson<fun>(params, x);
  check_solution_1x1(data, x);
}

TEST(nr_tests, test_1x1) { Test1x1<Function1x1>(); }


struct Function2x2FortranType
{
  static constexpr unsigned short n_dim = 2;
  using T = double;
  using T_vec = std::array<double, n_dim>;
  using T_mat = std::array<double, n_dim * n_dim>;

  static void eval_f_jac(const T_vec& x, T_vec& f, T_mat& jac) { f_jac_2x2(x, f, jac); }
};



template <typename fun>
void Test2x2()
{
  FindRoot::IterationParameters params;

  std::array<double, 2> x;
  init_2x2(x);
  FindRoot::IterationData data = FindRoot::NewtonRaphson<fun>(params, x);
  check_solution_2x2(data, x);
}



struct Function2x2UserType
{
  static constexpr unsigned short n_dim = 2;
  using T = double;
  using T_vec = Vector;
  using T_mat = Vector;

  static void eval_f_jac(const T_vec& x, T_vec& f, T_mat& jac) { f_jac_2x2(x, f, jac); }
};

TEST(nr_tests, test_2x2UserType)
{
  FindRoot::IterationParameters params;

  Vector x;
  init_2x2(x);
  FindRoot::IterationData data = FindRoot::NewtonRaphson<Function2x2UserType>(params, x);
  check_solution_2x2(data, x);
}

struct Function2x2UserParameter
{
  static constexpr unsigned short n_dim = 2;
  using T = double;
  using T_vec = std::array<double, n_dim>;
  using T_mat = std::array<double, n_dim * n_dim>;

  static void eval_f_jac(const T_vec& x, T_vec& f, T_mat& jac, const double parameter_1, const double parameter_2)
  {
    f_jac_2x2(x, f, jac, parameter_1, parameter_2);
  }
};


TEST(nr_tests, test_2x2UserParameter)
{
  FindRoot::IterationParameters params;

  std::array<double, 2> x;
  init_2x2(x);
  FindRoot::IterationData data = FindRoot::NewtonRaphson<Function2x2UserParameter>(params, x, 2.0, 1.0);
  check_solution_2x2(data, x);
}



/**
 * \brief
 */
TEST(nr_tests, test_2x2) { Test2x2<Function2x2>(); }
TEST(nr_tests, test_2x2FortranType) { Test2x2<Function2x2FortranType>(); }


struct Function3x3FortranType
{
  static constexpr unsigned short n_dim = 3;

  using T = double;
  using T_vec = std::array<double, n_dim>;
  using T_mat = std::array<double, n_dim * n_dim>;

  static void eval_f_jac(T_vec& x, T_vec& f, T_mat& jac) { f_jac_3x3(x, f, jac); }
};


template <typename fun>
void Test3x3()
{
  FindRoot::IterationParameters params;

  std::array<double, 3> x;
  init_3x3(x);
  FindRoot::IterationData data = FindRoot::NewtonRaphson<fun>(params, x);
  check_solution_3x3(data, x);
}

/**
 * \brief
 */
TEST(nr_tests, test_3x3) { Test3x3<Function3x3>(); }
TEST(nr_tests, test_3x3FortranType) { Test3x3<Function3x3FortranType>(); }
