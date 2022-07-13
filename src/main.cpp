
#include "lib/newton.H"

#include <array>
#include <math.h>

struct meins
{
  static constexpr unsigned short n_dim = 1;

  static void eval_f_jac(const double x[n_dim], double f[n_dim], double jac[n_dim * n_dim])
  {
    f[0] = std::sin(x[0]);
    jac[0] = std::cos(x[0]);
  }
};


// template <>
// struct newton::function<Mesh, ArborX::PrimitivesTag>



int main(int argc, char* argv[])
{
  std::array<double, 1> x;
  x[0] = 2;
  newton::newton<meins>(x.data());


  std::cout << "\nx: " << x[0] << "\n";

  return 0;
}
