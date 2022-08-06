
#include "lib/find_root.H"
#include <iostream>
#include <array>
#include <math.h>

struct meins
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


int main(int, char*[])
{
  FindRoot::IterationParameters params;
  params.max_iterations = 100;


  std::array<double, 2> x;
  x[0] = 1.0;
  x[1] = 1.0;
  x[2] = 1.0;
  FindRoot::IterationData data = FindRoot::NewtonRaphson<meins>(params, x.data());


  std::cout << "\nx: " << x[0] << "\n";
  std::cout << "\nx: " << x[1] << "\n";
  std::cout << "\nx: " << x[2] << "\n";
  std::cout << "\nError: " << data.error << "\n";
  std::cout << "\nNumber of iterations: " << data.iterations << "\n";
  std::cout << "\nConverged: " << data.converged << "\n";

  return 0;
}
