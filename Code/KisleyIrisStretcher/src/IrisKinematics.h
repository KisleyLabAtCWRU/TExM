#pragma once
#include "IrisGeometry.h"

namespace kisley {
namespace iris {

// Pure-math forward and inverse maps between crank angle θ and the
// expansion ratio Eₓ. Stateless; safe to call from anywhere.
//
// All work is done in double precision. Earlier code mixed float/double
// while using 1e-10 tolerances that fell below float epsilon (~1.2e-7);
// promoting to double makes those tolerances meaningful and is free on
// the ESP32-S3 FPU.
//
// The kinematic model only naturally produces Eₓ > 1 from positive θ.
// For "CCW expansion" — driving the motor by the same θ magnitude in
// the opposite direction — callers (IrisStretcher) negate the result
// of findTheta. The math itself stays single-sided.
class IrisKinematics {
public:
  // Forward map: θ (radians) → Eₓ (dimensionless expansion ratio).
  // Returns NaN if the geometry has no real solution at this θ.
  static double computeEx(const IrisGeometry& geo, double theta);

  // Inverse map: target Eₓ (must be > 1.0) → positive θ. Returns NaN
  // if Eₓ ≤ 1 or no bracket exists. Default bracket [0.0001, 0.6] rad.
  static double findTheta(const IrisGeometry& geo,
                          double targetEx,
                          double thetaLow  = 0.0001,
                          double thetaHigh = 0.6);

  // Low-level: solve within an explicit θ bracket. Bisection (50 iter)
  // then Newton-Raphson (50 iter). Returns NaN if target is not
  // bracketed.
  static double findThetaInBracket(const IrisGeometry& geo,
                                   double targetEx,
                                   double thetaLow,
                                   double thetaHigh);
};

} // namespace iris
} // namespace kisley
