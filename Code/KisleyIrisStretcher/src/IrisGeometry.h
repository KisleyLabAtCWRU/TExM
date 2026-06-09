#pragma once
#include <Arduino.h>

namespace kisley {
namespace iris {

// Pure-data description of an iris stretcher's mechanical geometry.
// Defaults match the Kisley Lab v1.0 rig.
struct IrisGeometry {
  float r0     = 7.2583f;   // cm — crank radius
  float rp     = 5.3241f;   // cm — pin/link length contributor
  float Y0     = -6.1913f;  // cm — must remain negative
  float rPin   = 0.7665f;   // cm — pin offset for the expansion ratio
  float g      = 12.7f;     // gear ratio
  float maxEx  = 4.65f;    // safety clamp for Xgoto magnitude (applies to both CW and CCW)
  uint16_t pulsesPerRev = 1600; // microsteps per output revolution

  // Blade-arm radius in cm. Used to convert blade linear speed (cm/s) into
  // shaft angular speed (rad/s):  omega = bladeSpeed / bladeRadiusCm.
  float bladeRadiusCm = 13.6f;

  // Default blade speed (cm/s) applied at IrisStretcher::begin() and used as
  // the Xspeed edit-screen starting value. Keep within [0.1, 5.0] to match
  // the UI's editable range.
  float defaultBladeSpeedCmPerSec = 0.1f;

  // E0 = r0 - rp (resting effective travel reference).
  float E0() const { return r0 - rp; }
};

} // namespace iris
} // namespace kisley
