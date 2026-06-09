#pragma once
#include <Arduino.h>

// Tiny helpers for emitting an aligned "key = value" report block to a
// Stream. Shared by the IrisStretcher move report (Xgoto/Xzero) and the
// IrisSerialConsole speed report (Xspeed) so they look identical.
//
// Layout:
//   === Xgoto ===
//     uptime_ms            = 1234567
//     current_position     = 0
//     ...
//   <blank line>
//
// Usage:
//   report::title(io, "Xgoto");
//   report::field(io, F("uptime_ms")); io.println(value);
//   report::blank(io);

namespace kisley {
namespace iris {
namespace report {

// Width of the key column so every "=" lines up. The longest key in use
// ("resulting_expansion", 19 chars) still leaves one separating space.
constexpr uint8_t kFieldWidth = 20;

// "=== <name> ===" header line.
inline void title(Stream& io, const char* name) {
  io.print(F("=== "));
  io.print(name);
  io.println(F(" ==="));
}

// Two-space indent, the key left-justified in a kFieldWidth column, then
// "= ". The caller prints the value immediately after (e.g. io.println(v)).
inline void field(Stream& io, const __FlashStringHelper* key) {
  io.print(F("  "));
  io.print(key);
  size_t n = strlen_P(reinterpret_cast<PGM_P>(key));
  while (n < kFieldWidth) { io.print(' '); ++n; }
  io.print(F("= "));
}

// Blank line separating consecutive reports.
inline void blank(Stream& io) { io.println(); }

} // namespace report
} // namespace iris
} // namespace kisley
