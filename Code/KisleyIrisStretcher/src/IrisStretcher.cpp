#include "IrisStretcher.h"
#include "internal/ReportFormat.h"
#include <math.h>

namespace kisley {
namespace iris {

IrisStretcher::IrisStretcher(uint8_t stepPin,
                             uint8_t dirPin,
                             const IrisGeometry& geo,
                             Stream& io)
  : _geo(geo),
    _driver(stepPin, dirPin),
    _io(io) {}

void IrisStretcher::begin() {
  _driver.begin();
  // Establish the default blade speed (0.1 cm/s unless the geometry
  // overrides it) so the rig boots at a known speed rather than relying
  // on the driver's raw step half-period default.
  setBladeSpeed(_geo.defaultBladeSpeedCmPerSec);
}

void IrisStretcher::rotateThetaRadians(double theta) {
  _rotate(theta, /*quiet=*/false);
}

long IrisStretcher::_rotate(double theta, bool quiet) {
  const long delta = _driver.rotateThetaRadians(_geo, theta);
  if (!quiet) {
    _io.print("Moved ");
    _io.print(delta);
    _io.print(" steps, new position = ");
    _io.println(_driver.currentPosition());
  }
  return delta;
}

void IrisStretcher::_printMoveReport(const char* cmd, unsigned long uptimeMs,
                                     long startSteps, long finalSteps,
                                     long stepsRequired, double targetEx,
                                     double resultingEx, const char* direction) {
  report::title(_io, cmd);
  report::field(_io, F("uptime_ms"));           _io.println(uptimeMs);
  report::field(_io, F("current_position"));    _io.println(startSteps);
  report::field(_io, F("target_expansion"));    _io.println(targetEx, 3);
  report::field(_io, F("resulting_expansion")); _io.println(resultingEx, 3);
  if (direction) { report::field(_io, F("direction")); _io.println(direction); }
  report::field(_io, F("steps_required"));      _io.println(stepsRequired);
  report::field(_io, F("current_step_count"));  _io.println(startSteps);
  report::field(_io, F("final_step_count"));    _io.println(finalSteps);
  report::field(_io, F("blade_speed_cm_s"));    _io.println(_bladeSpeedCmPerSec, 4);
  report::blank(_io);
}

bool IrisStretcher::gotoExpansion(double signedEx, MoveResult* out) {
  // Convention: |signedEx| is the magnitude in [1.0, maxEx]. Sign of
  // signedEx selects rotation direction — positive = CW, negative = CCW.
  // Both directions cover the same range; CCW rotates by the same θ
  // magnitude as the matching CW value but in the opposite direction.
  const unsigned long t0 = millis();
  const long startSteps = _driver.currentPosition();
  if (out) { *out = MoveResult{}; out->startSteps = startSteps; }
  const double magnitude = fabs(signedEx);

  // Center: |Ex| ≤ 1.0 means return to the zero step. Treat both 0 and
  // ±1.0 as "go to center" so the LCD's CCW path can also commit center.
  if (magnitude < 1.0 + 1e-6) {
    const long   delta       = _rotate(0.0, /*quiet=*/true);
    const long   finalSteps  = _driver.currentPosition();
    const double resultingEx = computeEx(0.0);
    if (out) {
      out->ok            = true;
      out->finalSteps    = finalSteps;
      out->stepsRequired = delta;
      out->targetEx      = 1.0;
      out->resultingEx   = resultingEx;
    }
    if (_reportMoves)
      _printMoveReport("Xgoto", t0, startSteps, finalSteps, delta, 1.0, resultingEx);
    return true;
  }

  if (magnitude >= _geo.maxEx) {
    // Error lines print regardless of the reporting flag.
    _io.print("Error: |Ex| out of range [1.0, ");
    _io.print(_geo.maxEx, 3);
    _io.println(")");
    if (out) out->ok = false;
    return false;
  }

  const double posTheta = findTheta(magnitude);   // always positive
  if (isnan(posTheta)) {
    _io.println("Error: no valid \xce\xb8 found for that targetEx");
    if (out) out->ok = false;
    return false;
  }

  const bool   cw = (signedEx >= 0.0);
  const double angle = cw ? posTheta : -posTheta;

  const long   delta       = _rotate(angle, /*quiet=*/true);
  const long   finalSteps  = _driver.currentPosition();
  // CCW has no Eₓ<1 domain in the model, so echo the commanded magnitude.
  const double resultingEx = cw ? computeEx(angle) : magnitude;

  if (out) {
    out->ok            = true;
    out->finalSteps    = finalSteps;
    out->stepsRequired = delta;
    out->targetEx      = magnitude;
    out->resultingEx   = resultingEx;
  }
  if (_reportMoves)
    _printMoveReport("Xgoto", t0, startSteps, finalSteps, delta, magnitude,
                     resultingEx, cw ? "cw" : "ccw");
  return true;
}

void IrisStretcher::goToZero(MoveResult* out) {
  const unsigned long t0 = millis();
  const long startSteps = _driver.currentPosition();
  if (out) { *out = MoveResult{}; out->startSteps = startSteps; }

  const long   delta       = _rotate(0.0, /*quiet=*/true);
  const long   finalSteps  = _driver.currentPosition();
  const double resultingEx = computeEx(0.0);

  if (out) {
    out->ok            = true;
    out->finalSteps    = finalSteps;
    out->stepsRequired = delta;
    out->targetEx      = 1.0;
    out->resultingEx   = resultingEx;
  }
  if (_reportMoves)
    _printMoveReport("Xzero", t0, startSteps, finalSteps, delta, 1.0, resultingEx);
}

void IrisStretcher::setZeroHere() {
  const unsigned long t0 = millis();
  const long previous = _driver.currentPosition();
  _driver.setCurrentPosition(0);
  // XsetZero resets the counter without moving the motor, so the report
  // carries only the before/after counts (no expansion / steps fields).
  if (_reportMoves) {
    report::title(_io, "XsetZero");
    report::field(_io, F("uptime_ms"));         _io.println(t0);
    report::field(_io, F("previous_position")); _io.println(previous);
    report::field(_io, F("new_position"));      _io.println(0L);
    report::field(_io, F("blade_speed_cm_s"));  _io.println(_bladeSpeedCmPerSec, 4);
    report::blank(_io);
  }
}

void IrisStretcher::calibrate() {
  if (_calibCb) {
    _calibCb(*this, _calibUser);
    return;
  }
  defaultCalibration();
}

void IrisStretcher::defaultCalibration() {
  // Calibration calls setZeroHere() repeatedly as internal bookkeeping;
  // suppress its per-call report so they don't clutter the routine's own
  // narrative. Restore the previous setting afterwards.
  const bool prevReport = _reportMoves;
  _reportMoves = false;

  _io.println("Starting calibration...");
  rotateThetaRadians(0.7);
  setZeroHere();
  _io.println("Moving back in by -0.290 rad...");
  rotateThetaRadians(-0.287);
  setZeroHere();
  _io.println("Setting Zero");
  setZeroHere();
  _io.println("Calibration complete.");

  _reportMoves = prevReport;
}

unsigned long IrisStretcher::setBladeSpeed(double cmPerSec, bool quiet) {
  _bladeSpeedCmPerSec = cmPerSec;
  const double omega = cmPerSec / _geo.bladeRadiusCm;
  unsigned long delayUs = 0;
  if (omega > 0.0) {
    delayUs = (unsigned long)lround(
      (M_PI * 1e6) / (omega * double(_geo.pulsesPerRev) * _geo.g));
  }
  _driver.setStepHalfPeriodUs(delayUs);
  if (!quiet) {
    _io.print("Step pulse half\xe2\x80\x90period delay set to ");
    _io.print(delayUs);
    _io.println(" \xc2\xb5s");
  }
  return delayUs;
}

} // namespace iris
} // namespace kisley
