
#pragma once

#include <array>

#include <iostream>

namespace newton
{
  template <typename T>
  struct function
  {
  };

  template <typename T>
  inline void newton(double x[T::n_dim])
  {
    constexpr unsigned short n_dim = T::n_dim;

    double r[n_dim];
    double delta_x[n_dim];


    std::cout << "\nasdfasfsdaf\n";
  };


#if 0
  // Approximated size of surface and beam diameter for valid projection check.
  const scalar_type surface_size = GetSurfaceSize(q_surface);
  const double beam_radius = GetLineRadius();

  // Vectors in 3D.
  LINALG::Matrix<3, 1, scalar_type> r_surface;
  LINALG::Matrix<3, 1, scalar_type> delta_xi(1.0);
  LINALG::Matrix<3, 1, scalar_type> residuum;

  // Jacobian / inverse.
  LINALG::Matrix<3, 3, scalar_type> J_J_inv;

  // Reset the projection result flag.
  projection_result = ProjectionResult::projection_not_found;

  // Local Newton iteration.
  {
    unsigned int counter = 0;
    while (counter < CONSTANTS::local_newton_iter_max)
    {
      // Evaluate the position and its derivative on the surface.
      EvaluateSurfacePositionAndDerivative(q_surface, xi, r_surface, J_J_inv, nodal_normals);

      // Evaluate the residuum $r_{solid} - r_{point} = R_{pos}$.
      residuum = r_surface;
      residuum -= point;

      if (counter == 0 and min_one_iteration)
      {
        // if the min_one_iteration flag is set we run at least one iteration, so the dependency on
        // FAD variables is calculated correctly.
      }
      else if (FADUTILS::VectorNorm(residuum) < CONSTANTS::local_newton_res_tol &&
               FADUTILS::VectorNorm(delta_xi) < CONSTANTS::projection_xi_eta_tol)
      {
        if (ValidParameterSurface(xi, surface_size, beam_radius))
          projection_result = ProjectionResult::projection_found_valid;
        else
          projection_result = ProjectionResult::projection_found_not_valid;
        break;
      }

      // Check if residuum is in a sensible range where we still expect to find a solution.
      if (FADUTILS::VectorNorm(residuum) > CONSTANTS::local_newton_res_max) break;

      // Solve the linearized system.
      if (LINALG::SolveLinearSystemDoNotThrowErrorOnZeroDeterminantScaled(
              J_J_inv, residuum, delta_xi, CONSTANTS::local_newton_det_tol))
      {
        // Set the new parameter coordinates.
        xi -= delta_xi;

        // Advance Newton iteration counter.
        counter++;
      }
      else
        break;
    }
  }
#endif
}  // namespace newton
