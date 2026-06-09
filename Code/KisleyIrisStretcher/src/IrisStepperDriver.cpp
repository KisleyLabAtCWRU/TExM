#include "IrisStepperDriver.h"
#include <math.h>

namespace kisley {
namespace iris {

IrisStepperDriver::IrisStepperDriver(uint8_t stepPin, uint8_t dirPin)
  : _stepPin(stepPin),
    _dirPin(dirPin),
    _stepper(AccelStepper::DRIVER, stepPin, dirPin) {}

void IrisStepperDriver::begin() {
  pinMode(_stepPin, OUTPUT);
  pinMode(_dirPin,  OUTPUT);
  digitalWrite(_stepPin, LOW);
  digitalWrite(_dirPin,  LOW);
}

long IrisStepperDriver::rotateThetaRadians(const IrisGeometry& geo, double theta) {
  const double stepsPerRad = (double(geo.pulsesPerRev) * geo.g) / (2.0 * M_PI);
  const long   targetSteps = (long)lround(theta * stepsPerRad);
  const long   startSteps  = _stepper.currentPosition();
  const long   deltaSteps  = targetSteps - startSteps;
  const bool   dir         = (deltaSteps >= 0);

  digitalWrite(_dirPin, dir ? HIGH : LOW);

  const long stepsToMove = labs(deltaSteps);
  for (long i = 0; i < stepsToMove; ++i) {
    digitalWrite(_stepPin, HIGH);
    delayMicroseconds(_stepHalfPeriodUs);
    digitalWrite(_stepPin, LOW);
    delayMicroseconds(_stepHalfPeriodUs);

    if (dir) _stepper.setCurrentPosition(startSteps + (i + 1));
    else     _stepper.setCurrentPosition(startSteps - (i + 1));

    if (_stepCb) _stepCb(i, _stepUser);

    if (_logEnabled && _logStream && (i % _logEvery == 0)) {
      _logStream->print("Steps, ");
      _logStream->println(i);
    }
  }
  return deltaSteps;
}

double IrisStepperDriver::currentTheta(const IrisGeometry& geo) {
  const double stepsPerRad = (double(geo.pulsesPerRev) * geo.g) / (2.0 * M_PI);
  return (double)_stepper.currentPosition() / stepsPerRad;
}

} // namespace iris
} // namespace kisley
