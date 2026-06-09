#pragma once
#include <Arduino.h>

namespace kisley {
namespace iris {

class DebouncedButton {
public:
  DebouncedButton() = default;
  explicit DebouncedButton(uint8_t pin) : _pin(pin) {}

  void attach(uint8_t pin) { _pin = pin; }

  void begin(uint8_t mode, bool activeHigh) {
    pinMode(_pin, mode);
    _activeLevel = activeHigh;
    _stable = _raw = digitalRead(_pin);
    _lastChangeMs = millis();
  }

  // Returns 0 = no change, 1 = released edge, 2 = pressed edge.
  // NOTE: edge() consumes the transition. Call once per loop and cache.
  uint8_t edge() {
    bool x = digitalRead(_pin);
    unsigned long now = millis();
    if (x != _raw) { _raw = x; _lastChangeMs = now; }
    if (now - _lastChangeMs >= _debounceMs && _raw != _stable) {
      bool wasPressed = (_stable == _activeLevel);
      _stable = _raw;
      bool nowPressed = (_stable == _activeLevel);
      if (wasPressed && !nowPressed) return 1;
      if (!wasPressed && nowPressed) return 2;
    }
    return 0;
  }

  bool fell()       { return edge() == 2; }
  bool isPressed()  const { return _stable == _activeLevel; }

private:
  uint8_t _pin = 0;
  bool _activeLevel = true;
  bool _stable = false;
  bool _raw = false;
  unsigned long _lastChangeMs = 0;
  static constexpr unsigned long _debounceMs = 25;
};

} // namespace iris
} // namespace kisley
