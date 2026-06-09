#pragma once
#include <Arduino.h>
#include "IrisGeometry.h"
#include "IrisStepperDriver.h"
#include "IrisKinematics.h"

namespace kisley {
namespace iris {

// High-level facade combining geometry + kinematics + stepper.
//
// Typical lab use:
//   IrisStretcher stretcher(/*STEP=*/12, /*DIR=*/13);
//   stretcher.begin();
//   stretcher.gotoExpansion(1.5);
//
class IrisStretcher {
public:
  using CalibrationRoutine = void(*)(IrisStretcher& self, void* user);

  IrisStretcher(uint8_t stepPin,
                uint8_t dirPin,
                const IrisGeometry& geo = IrisGeometry{},
                Stream& io = Serial);

  void begin();

  // Captured outcome of a single move, for callers (e.g. the serial
  // console) that want to report it. Filled when a MoveResult* is passed
  // to gotoExpansion()/goToZero(); see those methods.
  struct MoveResult {
    bool   ok            = false;  // false if the move was rejected
    long   startSteps    = 0;      // signed absolute step count before the move
    long   finalSteps    = 0;      // signed absolute step count after the move
    long   stepsRequired = 0;      // finalSteps - startSteps (signed; dir in sign)
    double targetEx      = 1.0;    // commanded magnitude (1.0 = center)
    double resultingEx   = 1.0;    // computeEx() of the final commanded angle
  };

  // ---- High-level motion ----
  // Both moves emit the structured key=value report (uptime_ms,
  // positions, target/resulting expansion, steps, blade speed) to the
  // serial stream by default — so the same report appears whether the
  // command came from the serial console or the LCD GUI. Pass `out` to
  // also capture the result programmatically. Disable the printed report
  // with setMoveReporting(false) (the experiment runner does this so the
  // per-waypoint moves don't pollute the CSV stream). Error lines always
  // print regardless of the reporting flag.
  bool gotoExpansion(double targetEx, MoveResult* out = nullptr);  // false if out of range
  void goToZero(MoveResult* out = nullptr);                        // drive to θ = 0
  void setZeroHere();                   // reset position counter to 0 (emits an XsetZero report)
  void calibrate();                     // runs default or registered routine
  // Updates the step half-period and returns it (µs). `quiet` suppresses
  // the confirmation line.
  unsigned long setBladeSpeed(double cmPerSec, bool quiet = false);

  // Toggle the intrinsic command reports printed by gotoExpansion/goToZero
  // (Xgoto/Xzero) and setZeroHere (XsetZero). Default on. Turned off during
  // experiment runs and the calibration routine to keep their output clean.
  void setMoveReporting(bool on)       { _reportMoves = on; }
  bool moveReporting() const           { return _reportMoves; }

  // ---- Extension seams ----
  void setCalibrationRoutine(CalibrationRoutine cb, void* user = nullptr) {
    _calibCb = cb; _calibUser = user;
  }
  void onEachStep(IrisStepperDriver::StepCallback cb, void* user = nullptr) {
    _driver.onEachStep(cb, user);
  }
  void setStepLogging(bool enabled, uint16_t every = 1) {
    _driver.setStepLogging(&_io, enabled, every);
  }

  // ---- Low-level escape hatches ----
  void   rotateThetaRadians(double theta);
  long   currentSteps()                { return _driver.currentPosition(); }
  double currentTheta()                { return _driver.currentTheta(_geo); }
  double computeEx(double theta) const { return IrisKinematics::computeEx(_geo, theta); }
  double findTheta(double targetEx) const { return IrisKinematics::findTheta(_geo, targetEx); }

  // ---- Accessors ----
  // Last blade speed (cm/s) set via setBladeSpeed(); also the geometry
  // default applied at begin().
  double bladeSpeedCmPerSec() const    { return _bladeSpeedCmPerSec; }
  const IrisGeometry& geometry() const { return _geo; }
  IrisGeometry&       geometry()       { return _geo; }
  IrisStepperDriver&  driver()         { return _driver; }
  Stream&             io()             { return _io; }

private:
  void defaultCalibration();
  // Core rotate; `quiet` suppresses the "Moved … steps" line. Returns the
  // signed step delta moved.
  long _rotate(double theta, bool quiet);
  // Emit the structured key=value move report to _io. `direction` is the
  // commanded "cw"/"ccw" token (Xgoto); pass nullptr to omit the row
  // (Xzero / center moves have no expansion direction).
  void _printMoveReport(const char* cmd, unsigned long uptimeMs,
                        long startSteps, long finalSteps, long stepsRequired,
                        double targetEx, double resultingEx,
                        const char* direction = nullptr);

  IrisGeometry        _geo;
  IrisStepperDriver   _driver;
  Stream&             _io;
  CalibrationRoutine  _calibCb   = nullptr;
  void*               _calibUser = nullptr;
  double              _bladeSpeedCmPerSec = 0.0;  // set in begin()/setBladeSpeed()
  bool                _reportMoves = true;        // intrinsic gotoExpansion/goToZero report
};

} // namespace iris
} // namespace kisley
