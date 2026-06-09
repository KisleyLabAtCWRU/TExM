#pragma once
#include <Arduino.h>
#include <Wire.h>
#include <hd44780.h>
#include <hd44780ioClass/hd44780_I2Cexp.h>
#include "IrisStretcher.h"
#include "internal/DebouncedButton.h"
#include "internal/QuadEncoder.h"

namespace kisley {
namespace iris {

class IrisExperimentRunner;   // fwd; declared in IrisExperiment.h
class IrisStrainArray;        // fwd; declared in IrisStrainArray.h

// "About" screen metadata. Defaults come from library build; override with
// setAboutInfo() in your sketch to display the sketch's build info instead.
struct IrisAboutInfo {
  const char* fwName    = "Iris Stretcher";
  const char* fwVersion = "v1.0.0";
  const char* labLine1  = "Kisley Lab";
  const char* labLine2  = "CWRU";
  const char* creator   = "Tejasvin Shrikanth";
  const char* buildDate = __DATE__;
  const char* buildTime = __TIME__;
};

class IrisMenuUI {
public:
  struct Pins {
    uint8_t btnMenu;
    uint8_t btnDown;
    uint8_t btnAccept;
    uint8_t encA;
    uint8_t encB;
    uint8_t encSW;
    uint8_t sda = 3;
    uint8_t scl = 4;
  };

  using ActionCallback = void(*)(IrisMenuUI& ui, void* user);

  IrisMenuUI(IrisStretcher& stretcher, const Pins& pins, Stream& io = Serial);

  void begin();    // initializes I2C, LCD, buttons, encoder, splash screen
  void update();   // call from loop() — fully non-blocking

  void setInvertEncoder(bool on)         { _invertEncoder = on; }
  void setUseInternalPulldown(bool on)   { _useInternalPulldown = on; }

  // When on (default), every GUI-triggered action echoes the equivalent
  // serial command line to the console stream — e.g. committing the Xgoto
  // edit screen prints "Xgoto 1.350 cw". The echoed lines use the exact
  // X-command syntax IrisSerialConsole accepts, so they are re-runnable.
  // Turn off to keep the serial log clean (e.g. to avoid one extra line in
  // an active experiment CSV stream when aborting from the LCD).
  void setEchoCommands(bool on)          { _echoCommands = on; }
  void setAboutInfo(const IrisAboutInfo& info) { _about = info; }
  void setShowSplash(bool on)            { _showSplash = on; }
  void setDetectedLcdAddress(uint8_t a)  { _lcdAddress = a; }

  uint8_t lcdAddress() const             { return _lcdAddress; }

  // Append a custom menu item. Returns false if MAX_MENU_ITEMS reached.
  bool registerMenuItem(const char* label,
                        ActionCallback cb,
                        void* user = nullptr);

  // Attach an IrisExperimentRunner so the LCD gets a top-level
  // "Experiments" entry that pushes into a submenu listing every
  // experiment registered on the runner. The runner is polled during
  // run for status (current step, target, motor state) so the LCD can
  // render progress. The MENU button mid-run requests an abort.
  void attachRunner(IrisExperimentRunner& runner);

  // Attach an IrisStrainArray so the LCD's "Xsignalavg" edit screen
  // can read and write the current signal-averaging value on the
  // strain array. Without this attachment, selecting Xsignalavg
  // briefly displays "No strain attached" instead.
  void attachStrain(IrisStrainArray& strain);

  // Public for advanced sketches that want direct LCD access.
  hd44780_I2Cexp& lcd() { return _lcd; }

  // Write a two-line message to the LCD without changing UI state.
  // Useful for "Initialising the ADCs..." style status during setup().
  // line1 may be nullptr.
  void showLcdMessage(const char* line0, const char* line1 = nullptr);

  // Re-render whatever UI state is currently active. Call this after
  // showLcdMessage() to bring the menu back, e.g.:
  //   ui.showLcdMessage("Initialising", "ADCs...");
  //   strain.begin();
  //   ui.refresh();
  void refresh();

  // Editable parameter ranges and step sizes — labs can tune these.
  float xSpeedMin       = 0.1f;
  float xSpeedMax       = 5.0f;
  float xSpeedStepFine  = 0.01f;
  float xSpeedStepCoarse = 0.10f;
  float xGotoStepFine   = 0.001f;
  float xGotoStepCoarse = 0.050f;

private:
  // ---- State ----
  enum class UiState : uint8_t {
    MENU,
    RUNNING,
    HELP_SCROLL,
    ABOUT_SCROLL,
    EDIT_XSPEED,
    EDIT_XGOTO,
    EDIT_XSIGNALAVG,
    EDIT_XSAMPLERATE,
    EDIT_XLOGEVERYN,
    EXPERIMENTS_MENU,
    RUNNING_EXPERIMENT,
  };

  enum class BuiltinKind : uint8_t {
    XsetZero, Xzero, Xcalibrate, Xspeed, Xgoto, Xsignalavg,
    Xsamplerate, XlogEveryN, Xhelp, Xabout,
    Experiments, Custom
  };

  struct MenuEntry {
    const char*    label;
    BuiltinKind    kind;
    ActionCallback callback;
    void*          user;
  };

  static constexpr uint8_t MAX_MENU_ITEMS = 16;

  // ---- Helpers ----
  uint8_t scanI2CForLCD();
  void drawMenu();
  void drawStatus(const char* msg);
  void drawEditXspeed();
  void drawEditXgoto();
  void drawExperimentsMenu();
  void drawRunningExperiment();
  void drawEditXsignalavg();
  void drawEditXsamplerate();
  void drawEditXlogEveryN();
  void printFloatFixed(float v, uint8_t places);
  // Echo a no-argument equivalent serial command (e.g. F("XsetZero")).
  // No-op when _echoCommands is false. Parametrised echoes (Xgoto/Xspeed/…)
  // are emitted inline at their commit sites.
  void echoCommand(const __FlashStringHelper* line);
  void runCurrentItem();
  void enterHelp();
  void enterAbout();
  void tickHelp();
  void tickAbout();

  // ---- Refs ----
  IrisStretcher&  _stretcher;
  Stream&         _io;
  Pins            _pins;

  // ---- Devices ----
  hd44780_I2Cexp  _lcd;
  uint8_t         _lcdAddress = 0x00;

  DebouncedButton _btnMenu;
  DebouncedButton _btnDown;
  DebouncedButton _btnAccept;
  DebouncedButton _btnEncSW;
  QuadEncoder     _enc;

  // ---- Menu ----
  MenuEntry _items[MAX_MENU_ITEMS];
  uint8_t   _itemCount = 0;
  uint8_t   _menuIndex = 0;

  // ---- UI state ----
  UiState        _ui = UiState::MENU;
  unsigned long  _statusUntilMs = 0;
  bool           _invertEncoder = false;
  bool           _useInternalPulldown = true;
  bool           _showSplash = true;
  bool           _echoCommands = true;  // echo GUI actions as serial commands

  // ---- Edit values ----
  float _xSpeedValue = 0.1f;   // seeded from geometry default in begin()
  float _xGotoValue  = 1.000f;   // magnitude, range [1.0, maxEx]
  bool  _xSpeedFineMode = false;
  bool  _xGotoFineMode  = false;
  bool  _xGotoDirCw     = true;  // CW default; toggled by DOWN button in edit screen

  // ---- Help/About paging ----
  IrisAboutInfo  _about;
  uint8_t        _scrollFrame = 0;
  unsigned long  _scrollNextMs = 0;

  // ---- Experiments submenu ----
  IrisExperimentRunner* _runner    = nullptr;
  uint8_t               _expIndex  = 0;   // index into runner's registered experiments
  uint32_t              _lastRunStatusMs = 0;

  // ---- Xsignalavg edit ----
  IrisStrainArray*  _strain               = nullptr;
  uint16_t          _xSignalAvgValue      = 4;
  bool              _xSignalAvgFineMode   = true;   // fine = ±1, coarse = ±10

  // ---- Xsamplerate edit ----
  // Index into the 5-entry rate table (10 / 20 / 40 / 80 / 320 SPS). The
  // NAU7802_SampleRate enum is non-contiguous (320 SPS = 7, not 4), so
  // we don't store the enum value directly.
  uint8_t           _xSampleRateIndex     = 4;     // 4 = 320 SPS (chip max)

  // ---- XlogEveryN edit ----
  uint16_t          _xLogEveryNValue      = 1;
  bool              _xLogEveryNFineMode   = true;  // fine = ±1, coarse = ±10
};

} // namespace iris
} // namespace kisley
