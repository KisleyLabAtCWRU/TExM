#pragma once
#include <Arduino.h>

namespace kisley {
namespace iris {

class QuadEncoder {
public:
  void begin(uint8_t pinA, uint8_t pinB) {
    _a = pinA;
    _b = pinB;
    pinMode(_a, INPUT_PULLUP);
    pinMode(_b, INPUT_PULLUP);
    _last = (digitalRead(_a) << 1) | digitalRead(_b);
  }

  // Returns -1, 0, or +1 per detent.
  int8_t step() {
    uint8_t s = (digitalRead(_a) << 1) | digitalRead(_b);
    static const int8_t quad[16] = {
       0, -1, +1,  0,
      +1,  0,  0, -1,
      -1,  0,  0, +1,
       0, +1, -1,  0
    };
    int8_t d = quad[(_last << 2) | s];
    _last = s;
    if (d) _accum += d;
    if ((s == 0 || s == 3) && (_accum >= 4 || _accum <= -4)) {
      int8_t out = (_accum > 0) ? +1 : -1;
      _accum = 0;
      return out;
    }
    return 0;
  }

private:
  uint8_t _a = 0, _b = 0;
  uint8_t _last = 0;
  int8_t  _accum = 0;
};

} // namespace iris
} // namespace kisley
