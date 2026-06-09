#pragma once
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_NAU7802.h>

namespace kisley {
namespace iris {

// Wraps Adafruit_NAU7802 with the ESP32 Wire-pin restore workaround:
// Adafruit_NAU7802::begin() re-pins I2C to default SDA/SCL on ESP32, which
// breaks the LCD if it shares the bus. begin() here calls Wire.begin(sda, scl)
// again after the ADC is up.
class IrisStrainNAU7802 {
public:
  bool  begin(uint8_t sda = 3, uint8_t scl = 4);

  // Averages `averages` samples (blocking) and returns volts referenced to
  // the configured reference voltage (default 3.0 V, 24-bit ADC).
  float readVolts(uint8_t averages = 35);

  float referenceVoltage() const     { return _vref; }
  void  setReferenceVoltage(float v) { _vref = v; }

  Adafruit_NAU7802& raw() { return _nau; }

private:
  Adafruit_NAU7802 _nau;
  float _vref = 3.0f;
};

} // namespace iris
} // namespace kisley
