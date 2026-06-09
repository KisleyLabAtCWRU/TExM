#include "IrisSerialConsole.h"
#include "IrisExperiment.h"
#include "internal/ReportFormat.h"
#include <string.h>
#include <stdlib.h>

namespace kisley {
namespace iris {

namespace {
// Case-insensitive equality for short ASCII tokens. Arduino doesn't
// portably provide strcasecmp, so we hand-roll one.
bool ieq(const char* a, const char* b) {
  while (*a && *b) {
    char ca = (*a >= 'A' && *a <= 'Z') ? char(*a + 32) : *a;
    char cb = (*b >= 'A' && *b <= 'Z') ? char(*b + 32) : *b;
    if (ca != cb) return false;
    a++; b++;
  }
  return *a == *b;
}

// Map a plain SPS integer (as typed in "Xsamplerate 320") to the
// Adafruit_NAU7802 rate enum. Returns false for unsupported values.
bool spsToRate(long sps, NAU7802_SampleRate& out) {
  switch (sps) {
    case 10:  out = NAU7802_RATE_10SPS;  return true;
    case 20:  out = NAU7802_RATE_20SPS;  return true;
    case 40:  out = NAU7802_RATE_40SPS;  return true;
    case 80:  out = NAU7802_RATE_80SPS;  return true;
    case 320: out = NAU7802_RATE_320SPS; return true;
    default:  return false;
  }
}
} // namespace

IrisSerialConsole::IrisSerialConsole(IrisStretcher& stretcher, Stream& io)
  : _stretcher(stretcher), _io(io) {}

void IrisSerialConsole::begin() {
  printBanner();
  printHelp();
}

void IrisSerialConsole::printBanner() {
  _io.println(" ___      _       ____  _            _       _               ");
  _io.println("|_ _|_ __(_)___  / ___|| |_ _ __ ___| |_ ___| |__   ___ _ __ ");
  _io.println(" | || '__| / __| \\___ \\| __| '__/ _ \\ __/ __| '_ \\ / _ \\ '__|");
  _io.println(" | || |  | \\__ \\  ___) | |_| | |  __/ || (__| | | |  __/ |   ");
  _io.println("|___|_|  |_|___/ |____/ \\__|_|  \\___|\\__\\___|_| |_|\\___|_|   ");
  _io.println("+===============+");
  _io.println(_bannerLine);
  _io.println("+===============+");
}

void IrisSerialConsole::printHelp() {
  _io.println("Commands:");
  _io.println("________");
  _io.println(" Xgoto <Ex> [cw|ccw]  \xe2\x80\x93 move to Ex magnitude (CW default)");
  _io.println(" Xzero                \xe2\x80\x93 drive motor to zero Position");
  _io.println(" XsetZero             \xe2\x80\x93 reset position counter to 0");
  _io.println(" Xcalibrate           \xe2\x80\x93 run calibration routine");
  _io.println(" Xspeed <value> cm/s  \xe2\x80\x93 changes expansion speed (cm/s approx)");
  _io.println(" Xstrain              \xe2\x80\x93 toggle 9-ADC strain CSV streaming");
  _io.println(" XsignalAverage <n>   \xe2\x80\x93 samples averaged per ADC per row [1..256]");
  _io.println(" Xsamplerate <sps>    \xe2\x80\x93 NAU7802 rate (10/20/40/80/320 SPS)");
  _io.println(" XlogEveryN <n>       \xe2\x80\x93 log one motion row per N steps [1..99]");
  _io.println(" Xrun <name>          \xe2\x80\x93 run a registered experiment (Xrun list)");
  _io.println(" Xabort               \xe2\x80\x93 abort a running experiment");
  _io.println(" Xhelp                \xe2\x80\x93 this message");
  for (uint8_t i = 0; i < _customCount; i++) {
    if (_custom[i].help) {
      _io.print(" X");
      _io.print(_custom[i].name);
      _io.print("  \xe2\x80\x93 ");
      _io.println(_custom[i].help);
    }
  }
}

void IrisSerialConsole::attachRunner(IrisExperimentRunner& runner) {
  _runner = &runner;
}

void IrisSerialConsole::attachStrain(IrisStrainArray& strain) {
  _strain = &strain;
}

void IrisSerialConsole::printSpeedReport(uint32_t uptimeMs, double bladeSpeedCmPerSec,
                                         unsigned long stepHalfPeriodUs, long currentSteps) {
  report::title(_io, "Xspeed");
  report::field(_io, F("uptime_ms"));           _io.println(uptimeMs);
  report::field(_io, F("blade_speed_cm_s"));    _io.println(bladeSpeedCmPerSec, 4);
  report::field(_io, F("step_half_period_us")); _io.println(stepHalfPeriodUs);
  report::field(_io, F("current_position"));    _io.println(currentSteps);
  report::blank(_io);
}

bool IrisSerialConsole::registerCommand(const char* name,
                                        const char* helpLine,
                                        CommandCallback cb,
                                        void* user) {
  if (!name || !cb) return false;
  if (_customCount >= MAX_CUSTOM_CMDS) return false;
  _custom[_customCount++] = {name, helpLine, cb, user};
  return true;
}

void IrisSerialConsole::update() {
  while (_io.available() && !_ready) {
    char c = (char)_io.read();
    if (c == '\n' || c == '\r') {
      if (_idx > 0) {
        _buf[_idx] = '\0';
        _ready = true;
      }
    } else if (_idx < MAX_CMD_LEN - 1) {
      _buf[_idx++] = c;
    }
  }
  if (_ready) {
    parseCommand();
    _idx = 0;
    _ready = false;
  }
}

void IrisSerialConsole::parseCommand() {
  if (_buf[0] != 'X') {
    _io.println("Invalid prefix. Commands must start with 'X'.");
    return;
  }
  char* p   = _buf + 1;
  char* cmd = strtok(p, " ");
  if (!cmd) return;

  if (dispatchBuiltin(cmd, nullptr)) return;

  // Custom commands
  for (uint8_t i = 0; i < _customCount; i++) {
    if (strcmp(cmd, _custom[i].name) == 0) {
      char* rest = strtok(nullptr, "");        // remainder of line
      _custom[i].cb(rest ? rest : "", _custom[i].user);
      return;
    }
  }

  _io.print("Unknown command: ");
  _io.println(cmd);
}

bool IrisSerialConsole::dispatchBuiltin(const char* cmd, char* /*tokState*/) {
  if (strcmp(cmd, "goto") == 0) {
    char* arg = strtok(nullptr, " ");
    if (!arg) { _io.println("Usage: Xgoto <Ex> [cw|ccw]"); return true; }
    const double magnitude = atof(arg);

    // Optional direction token; defaults to CW.
    char* dirArg = strtok(nullptr, " ");
    bool cw = true;
    if (dirArg) {
      if      (ieq(dirArg, "ccw")) cw = false;
      else if (ieq(dirArg, "cw"))  cw = true;
      else {
        _io.print("Unknown direction: "); _io.println(dirArg);
        _io.println("Usage: Xgoto <Ex> [cw|ccw]");
        return true;
      }
    }

    // gotoExpansion prints the structured report itself (intrinsic to the
    // command, so the LCD path produces the same output).
    _stretcher.gotoExpansion(cw ? magnitude : -magnitude);
    return true;
  }
  if (strcmp(cmd, "zero") == 0) {
    _stretcher.goToZero();   // prints the structured report intrinsically
    return true;
  }
  if (strcmp(cmd, "setZero") == 0) {
    _stretcher.setZeroHere();
    return true;
  }
  if (strcmp(cmd, "help") == 0) {
    printHelp();
    return true;
  }
  if (strcmp(cmd, "calibrate") == 0) {
    _stretcher.calibrate();
    return true;
  }
  if (strcmp(cmd, "strain") == 0) {
    if (!_runner) {
      _io.println(F("Error: no IrisExperimentRunner attached"));
      return true;
    }
    const bool wasOn = _runner->isStreamingStrain();
    _runner->setStreamStrain(!wasOn);
    _io.print(F("# Strain streaming: "));
    _io.println(_runner->isStreamingStrain() ? F("ON") : F("OFF"));
    return true;
  }
  if (strcmp(cmd, "run") == 0) {
    if (!_runner) {
      _io.println(F("Error: no IrisExperimentRunner attached"));
      return true;
    }
    char* arg = strtok(nullptr, " ");
    if (!arg) {
      _io.println(F("Usage: Xrun <name> | Xrun list"));
      return true;
    }
    if (ieq(arg, "list")) {
      _io.print(F("# Registered experiments ("));
      _io.print(_runner->experimentCount()); _io.println(F("):"));
      for (uint8_t i = 0; i < _runner->experimentCount(); i++) {
        const IrisExperiment* e = _runner->experiment(i);
        if (e) { _io.print(F("#   ")); _io.println(e->name); }
      }
      return true;
    }
    if (!_runner->requestRunByName(arg)) {
      _io.print(F("Error: no experiment named ")); _io.println(arg);
    }
    return true;
  }
  if (strcmp(cmd, "abort") == 0) {
    if (!_runner) {
      _io.println(F("Error: no IrisExperimentRunner attached"));
      return true;
    }
    _runner->requestAbort();
    _io.println(F("# Abort requested"));
    return true;
  }
  if (strcmp(cmd, "speed") == 0) {
    char* arg = strtok(nullptr, " ");
    if (!arg) { _io.println("Usage: Xspeed <bladeSpeed_cm/s>"); return true; }
    const double sb = atof(arg);
    if (sb <= 0.0) {
      _io.println("Error: speed must be > 0");
    } else {
      const uint32_t t0 = millis();
      const unsigned long halfUs = _stretcher.setBladeSpeed(sb, /*quiet=*/true);
      printSpeedReport(t0, sb, halfUs, _stretcher.currentSteps());
    }
    return true;
  }
  // Strain signal-averaging — mirrors the LCD's XsignalAverage screen.
  if (strcmp(cmd, "signalAverage") == 0) {
    if (!_strain) { _io.println(F("Error: no IrisStrainArray attached")); return true; }
    char* arg = strtok(nullptr, " ");
    if (!arg) { _io.println(F("Usage: XsignalAverage <1..256>")); return true; }
    long n = atol(arg);
    if (n < 1) n = 1;
    if (n > IrisStrainArray::MAX_AVG) n = IrisStrainArray::MAX_AVG;
    _strain->setSignalAveraging((uint16_t)n);
    _io.print(F("# Signal averaging set to ")); _io.println(n);
    return true;
  }
  // NAU7802 conversion rate — mirrors the LCD's Xsamplerate screen.
  if (strcmp(cmd, "samplerate") == 0) {
    if (!_strain) { _io.println(F("Error: no IrisStrainArray attached")); return true; }
    char* arg = strtok(nullptr, " ");
    if (!arg) { _io.println(F("Usage: Xsamplerate <10|20|40|80|320>")); return true; }
    NAU7802_SampleRate rate;
    if (!spsToRate(atol(arg), rate)) {
      _io.print(F("Error: unsupported rate ")); _io.print(arg);
      _io.println(F(" (use 10/20/40/80/320)"));
      return true;
    }
    _strain->setSampleRate(rate);
    _strain->applySampleRateLive();
    _io.print(F("# Sample rate set to ")); _io.print(arg); _io.println(F(" SPS"));
    return true;
  }
  // Motion-log step gate — mirrors the LCD's XlogEveryN screen.
  if (strcmp(cmd, "logEveryN") == 0) {
    if (!_runner) { _io.println(F("Error: no IrisExperimentRunner attached")); return true; }
    char* arg = strtok(nullptr, " ");
    if (!arg) { _io.println(F("Usage: XlogEveryN <1..99>")); return true; }
    long n = atol(arg);
    if (n < 1)  n = 1;
    if (n > 99) n = 99;
    _runner->setMotionLogEveryNSteps((uint16_t)n);
    _io.print(F("# Motion log every ")); _io.print(n); _io.println(F(" steps"));
    return true;
  }
  return false;
}

} // namespace iris
} // namespace kisley
