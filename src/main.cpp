
#include "lib/newton.H"

#include <array>
#include <math.h>

struct meins
{
  static constexpr unsigned short n_dim = 1;

  static void eval_f_jac(const double x[n_dim], double f[n_dim], double jac[n_dim * n_dim])
  {
    f[0] = std::cos(x[0]);
    jac[0] = -std::sin(x[0]);
  }
};


int main(int, char*[])
{
  FindRoot::IterationParameters params;
  params.max_iterations = 10;


  std::array<double, 1> x;
  x[0] = 2;
  FindRoot::IterationData data = FindRoot::NewtonRaphson<meins>(params, x.data());


  std::cout << "\nx: " << x[0] << "\n";
  std::cout << "\nError: " << data.error << "\n";
  std::cout << "\nNumber of iterations: " << data.iterations << "\n";
  std::cout << "\nConverged: " << data.converged << "\n";

  return 0;
}
