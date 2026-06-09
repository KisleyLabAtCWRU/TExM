#pragma once
#include <Arduino.h>
#include <AccelStepper.h>
#include "IrisGeometry.h"

namespace kisley {
namespace iris {

// Blocking STEP/DIR bit-bang motion driver.
// AccelStepper is retained purely as the position counter.
class IrisStepperDriver {
public:
  using StepCallback = void(*)(long stepIndex, void* user);

  IrisStepperDriver(uint8_t stepPin, uint8_t dirPin);

  // Configures pinMode for STEP/DIR.
  void begin();

  // Executes a blocking move to absolute angle `theta` (radians) using the
  // geometry's gear ratio + pulses-per-rev. Returns signed step delta moved.
  long rotateThetaRadians(const IrisGeometry& geo, double theta);

  // Microseconds spent in each STEP high/low half. Default 2000 µs.
  void setStepHalfPeriodUs(unsigned long us) { _stepHalfPeriodUs = us; }
  unsigned long stepHalfPeriodUs() const     { return _stepHalfPeriodUs; }

  // Position bookkeeping. Not const because AccelStepper::currentPosition()
  // is not declared const upstream.
  long currentPosition()                 { return _stepper.currentPosition(); }
  void setCurrentPosition(long position) { _stepper.setCurrentPosition(position); }
  double currentTheta(const IrisGeometry& geo);

  // Optional per-step callback invoked from inside the move loop.
  void onEachStep(StepCallback cb, void* user = nullptr) {
    _stepCb = cb;
    _stepUser = user;
  }

  // Optional per-step "Steps, <i>" line written to `stream`. Use `every` to
  // throttle (every=10 prints every 10th step). Pass stream=nullptr to disable.
  void setStepLogging(Stream* stream, bool enabled, uint16_t every = 1) {
    _logStream  = stream;
    _logEnabled = enabled;
    _logEvery   = every ? every : 1;
  }

  uint8_t stepPin() const { return _stepPin; }
  uint8_t dirPin()  const { return _dirPin;  }

private:
  uint8_t _stepPin;
  uint8_t _dirPin;
  unsigned long _stepHalfPeriodUs = 2000;
  AccelStepper _stepper;
  StepCallback _stepCb = nullptr;
  void*        _stepUser = nullptr;
  Stream*      _logStream = nullptr;
  bool         _logEnabled = false;
  uint16_t     _logEvery = 1;
};

} // namespace iris
} // namespace kisley
