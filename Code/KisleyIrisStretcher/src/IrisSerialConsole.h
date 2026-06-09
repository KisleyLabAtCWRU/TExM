#pragma once
#include <Arduino.h>
#include "IrisStretcher.h"
#include "IrisStrainArray.h"

namespace kisley {
namespace iris {

class IrisExperimentRunner;   // fwd

// Reads "X<command> [args]" lines from a Stream and dispatches them.
// Built-in commands:  Xgoto <Ex>, Xzero, XsetZero, Xcalibrate,
//                     Xspeed <cm/s>, Xhelp.
//
// Labs can append their own commands via registerCommand().
class IrisSerialConsole {
public:
  using CommandCallback = void(*)(const char* args, void* user);

  IrisSerialConsole(IrisStretcher& stretcher, Stream& io = Serial);

  // Prints banner + help. Does NOT start the underlying Stream — the sketch
  // is responsible for Serial.begin() / USB.begin() before this is called.
  void begin();

  // Call from loop(). Non-blocking.
  void update();

  // Append a custom command. `name` is the bare command (no "X" prefix);
  // sending "Xfoo bar baz" with name="foo" calls cb with args="bar baz".
  bool registerCommand(const char* name,
                       const char* helpLine,
                       CommandCallback cb,
                       void* user = nullptr);

  // Attach an experiment runner so the console exposes Xstrain (toggle
  // strain streaming), Xrun <name> (start a registered experiment),
  // Xrun list (dump registered names), and Xabort (interrupt a run).
  void attachRunner(IrisExperimentRunner& runner);

  // Attach an IrisStrainArray so the console exposes the strain-tuning
  // commands that mirror the LCD edit screens:
  //   XsignalAverage <n>  — samples averaged per ADC per CSV row [1..256]
  //   Xsamplerate <sps>   — NAU7802 conversion rate (10/20/40/80/320)
  // Without this attachment those commands report "no strain attached".
  void attachStrain(IrisStrainArray& strain);

  void printBanner();
  void printHelp();

  // Override the single mid-banner identification line (default "KisleyLab V1.0").
  void setBannerLine(const char* line) { _bannerLine = line; }

private:
  void parseCommand();
  bool dispatchBuiltin(const char* cmd, char* tokenizerState);

  // Structured "key=value" report for Xspeed (no move). The Xgoto/Xzero
  // move report is intrinsic to IrisStretcher so it appears from the LCD
  // path too. `uptimeMs` is millis() at command receipt — same clock as
  // the strain/experiment CSV t_ms column.
  void printSpeedReport(uint32_t uptimeMs, double bladeSpeedCmPerSec,
                        unsigned long stepHalfPeriodUs, long currentSteps);

  static constexpr uint8_t MAX_CMD_LEN     = 64;
  static constexpr uint8_t MAX_CUSTOM_CMDS = 8;

  struct CustomCmd {
    const char*     name;
    const char*     help;
    CommandCallback cb;
    void*           user;
  };

  IrisStretcher& _stretcher;
  Stream&        _io;

  char     _buf[MAX_CMD_LEN];
  uint8_t  _idx = 0;
  bool     _ready = false;

  CustomCmd _custom[MAX_CUSTOM_CMDS];
  uint8_t   _customCount = 0;

  const char* _bannerLine = "|KisleyLab V1.0 |";

  IrisExperimentRunner* _runner = nullptr;
  IrisStrainArray*      _strain = nullptr;
};

} // namespace iris
} // namespace kisley
