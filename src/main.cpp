
#include "lib/newton.H"
#include <iostream>
#include <array>
#include <math.h>

struct meins
{
  static constexpr unsigned short n_dim = 2;

  static void eval_f_jac(const double x[n_dim], double f[n_dim], double jac[n_dim * n_dim])
  {
    f[0] = x[0] * x[0] + x[1] + 2.0;
    f[1] = 6.0 * x[0] - x[1] * (x[0] + 1.0);
    jac[0] = 2.0 * x[0];
    jac[1] = 6.0 - x[1];
    jac[2] = 1.0;
    jac[3] = -1.0 - x[0];
  }
};


int main(int, char*[])
{
  FindRoot::IterationParameters params;
  params.max_iterations = 100;


  std::array<double, 2> x;
  x[0] = 5;
  x[1] = 1;
  FindRoot::IterationData data = FindRoot::NewtonRaphson<meins>(params, x.data());


  std::cout << "\nx: " << x[0] << "\n";
  std::cout << "\nx: " << x[1] << "\n";
  std::cout << "\nError: " << data.error << "\n";
  std::cout << "\nNumber of iterations: " << data.iterations << "\n";
  std::cout << "\nConverged: " << data.converged << "\n";

  return 0;
}
