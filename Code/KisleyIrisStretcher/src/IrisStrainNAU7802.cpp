#include "IrisStrainNAU7802.h"

namespace kisley {
namespace iris {

bool IrisStrainNAU7802::begin(uint8_t sda, uint8_t scl) {
  if (!_nau.begin()) return false;

  // Adafruit_NAU7802::begin() calls Wire.begin() with default pins on ESP32,
  // overriding any custom pins set earlier. Re-pin here to restore.
  Wire.begin(sda, scl);

  _nau.setRate(NAU7802_RATE_10SPS);
  _nau.setGain(NAU7802_GAIN_1);
  _nau.calibrate(NAU7802_CALMOD_INTERNAL);
  return true;
}

float IrisStrainNAU7802::readVolts(uint8_t averages) {
  if (averages == 0) averages = 1;
  long sum = 0;
  for (uint8_t k = 0; k < averages; k++) {
    while (!_nau.available()) { /* spin */ }
    sum += _nau.read();
  }
  const float avgCounts = float(sum) / float(averages);
  return avgCounts * (_vref / float(1UL << 23));
}

} // namespace iris
} // namespace kisley
