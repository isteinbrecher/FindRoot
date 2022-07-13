
#include "lib/newton.H"

#include <array>

struct meins
{
  static constexpr unsigned short n_dim = 3;
};


int main(int argc, char* argv[])
{
  std::array<double, 3> x;
  newton::newton<meins>(x.data());

  return 0;
}
