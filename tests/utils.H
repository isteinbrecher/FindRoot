


#include "../src/lib/find_root.H"

constexpr double tol = 1e-12;


class Vector
{
 public:
  Vector() { data_.resize(100); };
  std::vector<double> data_;
};

template <typename T_vec, typename T_mat>
void f_jac_2x2(const T_vec& x, T_vec& f, T_mat& jac, const double parameter_1 = 2.0, const double parameter_2 = 1.0)
{
  constexpr unsigned short n_dim = 2;

  GetV(f, 0) = GetV(x, 0) * GetV(x, 0) + GetV(x, 1) + parameter_1;
  GetV(f, 1) = 6.0 * GetV(x, 0) - GetV(x, 1) * (GetV(x, 0) + parameter_2);

  GetM(jac, 0, 0) = 2.0 * GetV(x, 0);
  GetM(jac, 1, 0) = 6.0 - GetV(x, 1);
  GetM(jac, 0, 1) = 1.0;
  GetM(jac, 1, 1) = -1.0 - GetV(x, 0);
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
  GetV(x, 0) = 5.0;
  GetV(x, 1) = 1.0;
}

template <typename T_vec>
void check_solution_2x2(const FindRoot::IterationData& data, const T_vec& x)
{
  constexpr unsigned short n_dim = 2;

  EXPECT_NEAR(GetV(x, 0), -0.25609874104976987, tol);
  EXPECT_NEAR(GetV(x, 1), -2.0655865651672771, tol);
  EXPECT_EQ(data.iterations, 8);
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
void check_solution_3x3(const FindRoot::IterationData& data, const T_vec& x)
{
  check_solution_3x3(x);
  EXPECT_EQ(data.iterations, 7);
}
