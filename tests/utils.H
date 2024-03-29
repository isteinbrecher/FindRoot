

#include <math.h>
#include "../src/lib/find_root.H"

constexpr double tol = 1e-12;


template <typename scalar_type, unsigned int n_row>
class VectorT
{
 public:
  VectorT() = default;

  inline const scalar_type& operator()(const unsigned int i_row) const { return data_[i_row]; }
  inline scalar_type& operator()(const unsigned int i_row) { return data_[i_row]; }

 private:
  scalar_type data_[n_row];
};

template <typename scalar_type, unsigned int n_row, unsigned int n_col>
class MatrixT
{
 public:
  MatrixT() = default;

  inline const scalar_type& operator()(const unsigned int i_row, const unsigned int i_col) const
  {
    return data_[i_row];
  }
  inline scalar_type& operator()(const unsigned int i_row, const unsigned int i_col)
  {
    return data_[i_row * n_col + i_col];
  }

 private:
  scalar_type data_[n_row * n_col];
};


class Vector
{
 public:
  Vector() { data_.resize(100); };
  std::vector<double> data_;
};



template <typename T_vec, typename T_mat>
void f_jac_1x1(const T_vec& x, T_vec& f, T_mat& jac)
{
  constexpr unsigned short n_dim = 1;

  GetV(f, 0) = std::pow(2.718281828459045, GetV(x, 0.)) + std::sin(GetV(x, 0.));
  GetM(jac, 0, 0) = std::pow(2.718281828459045, GetV(x, 0.)) + std::cos(GetV(x, 0.));
}

struct Function1x1
{
  static constexpr unsigned short n_dim = 1;
  using T = double;
  using T_vec = std::array<double, n_dim>;
  using T_mat = std::array<T_vec, n_dim>;

  static void eval_f_jac(T_vec& x, T_vec& f, T_mat& jac) { f_jac_1x1(x, f, jac); }
};

template <typename T_vec>
void init_1x1(T_vec& x)
{
  constexpr unsigned short n_dim = 1;
  GetV(x, 0) = 0.0;
}

template <typename T_vec>
void check_solution_1x1(const T_vec& x)
{
  constexpr unsigned short n_dim = 1;

  EXPECT_NEAR(GetV(x, 0), -0.5885327439818611, tol);
}

template <typename T_vec>
void check_solution_1x1_no_iterations(const T_vec& x)
{
  check_solution_1x1(x);
}

template <typename T_vec>
void check_solution_1x1(const FindRoot::IterationData& data, const T_vec& x)
{
  check_solution_1x1(x);
  EXPECT_EQ(data.iterations, 6);
}



template <typename T_vec, typename T_mat>
void f_jac_2x2(const T_vec& x, T_vec& f, T_mat& jac, const double parameter_1 = 2.0, const double parameter_2 = 1.0)
{
  constexpr unsigned short n_dim = 2;

  GetV(f, 0) = std::exp(-parameter_1 + GetV(x, 0)) - GetV(x, 1);
  GetV(f, 1) = GetV(x, 1) * GetV(x, 1) - GetV(x, 0) * parameter_2;

  GetM(jac, 0, 0) = std::exp(-parameter_1 + GetV(x, 0));
  GetM(jac, 0, 1) = -1.0;
  GetM(jac, 1, 0) = -parameter_2;
  GetM(jac, 1, 1) = 2.0 * GetV(x, 1);
}

struct Function2x2
{
  static constexpr unsigned short n_dim = 2;
  using T = double;
  using T_vec = std::array<double, n_dim>;
  using T_mat = std::array<T_vec, n_dim>;

  static void eval_f_jac(T_vec& x, T_vec& f, T_mat& jac) { f_jac_2x2(x, f, jac); }
};

template <typename T_vec>
void init_2x2(T_vec& x)
{
  constexpr unsigned short n_dim = 2;
  GetV(x, 0) = 0.0;
  GetV(x, 1) = 0.0;
}

template <typename T_vec>
void check_solution_2x2(const T_vec& x)
{
  constexpr unsigned short n_dim = 2;

  EXPECT_NEAR(GetV(x, 0), 0.019026016103714051, tol);
  EXPECT_NEAR(GetV(x, 1), 0.13793482556524314, tol);
}

template <typename T_vec>
void check_solution_2x2_no_iterations(const T_vec& x)
{
  check_solution_2x2(x);
}

template <typename T_vec>
void check_solution_2x2(const FindRoot::IterationData& data, const T_vec& x)
{
  check_solution_2x2(x);
  EXPECT_EQ(data.iterations, 5);
}


template <typename T_vec, typename T_mat>
void f_jac_3x3(T_vec& x, T_vec& f, T_mat& jac)
{
  constexpr unsigned short n_dim = 3;

  GetV(f, 0) = GetV(x, 0) * GetV(x, 1) - GetV(x, 1) * GetV(x, 2) + 1.0;
  GetV(f, 1) = GetV(x, 0) * GetV(x, 1) * GetV(x, 2) - 5.0;
  GetV(f, 2) = GetV(x, 0) + GetV(x, 2) - 2;

  GetM(jac, 0, 0) = GetV(x, 1);
  GetM(jac, 1, 0) = GetV(x, 1) * GetV(x, 2);
  GetM(jac, 2, 0) = 1.0;

  GetM(jac, 0, 1) = GetV(x, 0) - GetV(x, 2);
  GetM(jac, 1, 1) = GetV(x, 0) * GetV(x, 2);
  GetM(jac, 2, 1) = 0.0;

  GetM(jac, 0, 2) = -GetV(x, 1);
  GetM(jac, 1, 2) = GetV(x, 0) * GetV(x, 1);
  GetM(jac, 2, 2) = 1.0;
}


struct Function3x3
{
  static constexpr unsigned short n_dim = 3;

  using T = double;
  using T_vec = std::array<double, n_dim>;
  using T_mat = std::array<T_vec, n_dim>;

  static void eval_f_jac(T_vec& x, T_vec& f, T_mat& jac) { f_jac_3x3(x, f, jac); }
};

template <typename T_vec>
void init_3x3(T_vec& x)
{
  constexpr unsigned short n_dim = 3;
  GetV(x, 0) = 1.0;
  GetV(x, 1) = 1.0;
  GetV(x, 2) = 1.0;
}


template <typename T_vec>
void check_solution_3x3(const T_vec& x)
{
  constexpr unsigned short n_dim = 3;

  EXPECT_NEAR(GetV(x, 0), 0.90098048640721518, tol);
  EXPECT_NEAR(GetV(x, 1), 5.0495097567963922, tol);
  EXPECT_NEAR(GetV(x, 2), 1.0990195135927849, tol);
}

template <typename T_vec>
void check_solution_3x3_no_iterations(const T_vec& x)
{
  check_solution_3x3(x);
}

template <typename T_vec>
void check_solution_3x3(const FindRoot::IterationData& data, const T_vec& x)
{
  check_solution_3x3(x);
  EXPECT_EQ(data.iterations, 7);
}
