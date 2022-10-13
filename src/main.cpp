
#include "find_root.H"
#include <iostream>
#include <array>
#include <math.h>

struct meins
{
  static constexpr unsigned short n_dim = 3;
  using T = double;
  using T_vec = std::array<double, n_dim>;
  using T_mat = std::array<T_vec, n_dim>;

  static void eval_f_jac(const T_vec& x, T_vec& f, T_mat& jac)
  {
    f[0] = x[0] * x[1] - x[1] * x[2] + 1.0;
    f[1] = x[0] * x[1] * x[2] - 5.0;
    f[2] = x[0] + x[2] - 2;

    jac[0][0] = x[1];
    jac[1][0] = x[1] * x[2];
    jac[2][0] = 1.0;

    jac[0][1] = x[0] - x[2];
    jac[1][1] = x[0] * x[2];
    jac[2][1] = 0.0;

    jac[0][2] = -x[1];
    jac[1][2] = x[0] * x[1];
    jac[2][2] = 1.0;
  }
};


int main(int, char*[])
{
  FindRoot::IterationParameters params;
  params.max_iterations = 100;


  std::array<double, 3> x;
  x[0] = 1.0;
  x[1] = 1.0;
  x[2] = 1.0;
  FindRoot::IterationData data = FindRoot::NewtonRaphson<meins>(params, x);


  std::cout << "\nx: " << x[0] << "\n";
  std::cout << "\nx: " << x[1] << "\n";
  std::cout << "\nx: " << x[2] << "\n";
  std::cout << "\nError: " << data.error << "\n";
  std::cout << "\nNumber of iterations: " << data.iterations << "\n";
  std::cout << "\nConverged: " << data.converged << "\n";

  return 0;
}
