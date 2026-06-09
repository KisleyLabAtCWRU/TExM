// IrisModule2.0 — canonical Kisley rig firmware + extended LCD menu.
//
// Identical wiring and core behaviour to IrisModule1.0, but the library's
// LCD menu now exposes two additional runtime-editable settings:
//   - Xsamplerate: cycle the NAU7802 conversion rate (10 / 20 / 40 /
//     80 / 320 SPS) without rebooting. ACCEPT live-applies the new rate
//     to every initialised chip in ~50 ms.
//   - XlogEveryN: edit IrisExperimentRunner::setMotionLogEveryNSteps()
//     at runtime, range [1, 99]. Lets you dial the step-modulo filter
//     for motion-CSV emission from the LCD instead of recompiling.
//
// Boot defaults: sample rate 320 SPS (chip max), logEveryN = 1 (every
// step is eligible). Adjust from the LCD as needed during the session.
//
// Hardware: ESP32-S3, HD44780 I2C LCD, rotary encoder, 3 buttons,
// STEP/DIR stepper driver, 9 NAU7802 strain ADCs behind two TCA9548A
// muxes (5 on 0x71 ch0..4, 4 on 0x73 ch0..3).

#include <USB.h>
#include <KisleyIrisStretcher.h>

using namespace kisley::iris;

// ---- Hardware pin map ----
constexpr uint8_t PIN_STEP   = 12;
constexpr uint8_t PIN_DIR    = 13;

constexpr uint8_t PIN_MENU   = A0;
constexpr uint8_t PIN_DOWN   = A1;
constexpr uint8_t PIN_ACCEPT = A2;
constexpr uint8_t PIN_ENC_SW = A3;
constexpr uint8_t PIN_ENC_B  = A4;
constexpr uint8_t PIN_ENC_A  = A5;
constexpr uint8_t PIN_SDA    = 3;
constexpr uint8_t PIN_SCL    = 4;

// ---- Core library components ----
IrisStretcher     stretcher(PIN_STEP, PIN_DIR);
IrisStrainArray   strain;
IrisMenuUI        ui(stretcher, IrisMenuUI::Pins{
                    PIN_MENU, PIN_DOWN, PIN_ACCEPT,
                    PIN_ENC_A, PIN_ENC_B, PIN_ENC_SW,
                    PIN_SDA, PIN_SCL});
IrisSerialConsole    console(stretcher);
IrisExperimentRunner runner(stretcher, strain);

// =====================================================================
// EXPERIMENTS — add new entries here and register them in setup().
// Each `targets` array holds signed Ex values (positive = CW, negative
// = CCW, |Ex| ≤ 1 returns to center). Hold time between waypoints is
// the runner's global setting (default 2000 ms).
// =====================================================================

const float kExp1Targets[] = { 1.0f, 3.4f, 1.0f };
const IrisExperiment kExp1 = { "Exp1", kExp1Targets, 3 };

// Exp2: stepped ramp — 1x → 2x → 1x → 3x → 1x → 3.4x.
// Each pair of waypoints holds for the runner's global hold time
// (default 2 s) before advancing.
const float kExp2Targets[] = { 1.0f, 2.0f, 1.0f, 3.0f, 1.0f, 3.4f };
const IrisExperiment kExp2 = { "Exp2", kExp2Targets, 6 };

void setup() {
  delay(500);
  USB.begin();
  Serial.begin(115200);
  while (!Serial && millis() < 2000) {}

  stretcher.begin();
  stretcher.setStepLogging(false);   // step spam disabled — Xstrain is the data path

  IrisAboutInfo info;
  info.fwName    = "Iris Stretcher";
  info.fwVersion = "v2.1.0";
  info.labLine1  = "Kisley Lab";
  info.labLine2  = "CWRU";
  info.creator   = "Tejasvin Shrikanth";
  info.buildDate = __DATE__;
  info.buildTime = __TIME__;
  ui.setAboutInfo(info);

  ui.setInvertEncoder(false);
  ui.setUseInternalPulldown(true);
  ui.begin();                        // brings up Wire (SDA=3, SCL=4) + LCD

  // Speed/noise balance tuned for the Kisley rig:
  //   - I2C bus at 50 kHz (5× the safe-default 10 kHz; still well below the
  //     bus's 100 kHz failure point on this wiring)
  //   - NAU7802 conversion rate explicitly pinned at 320 SPS — the chip's
  //     documented maximum. Library already defaults to this; the explicit
  //     call documents the intent and is the right place to dial down if
  //     a future rig needs less noise per sample (e.g. RATE_80SPS).
  //   - mux settle wait set to 0 — TCA9548A switches sub-µs and the
  //     conversion-ready poll is already skipped in the read path, so the
  //     post-route delay buys nothing on this rig. Raise to ~50 µs if you
  //     see occasional bad first-reads after a mux switch.
  //   - 2 raw samples averaged per ADC per CSV row — keeps acquireRow under
  //     ~90 ms so it doesn't visibly stall the motor between log rows
  //     during gotoExpansion. Per-row Bessel stdev is ~2× larger than at
  //     N=8 but motion is smooth. Bump back to 8 for static drift/noise
  //     captures where motion smoothness doesn't matter.
  // Dial setI2cClock(10000) to fall back to safe default if you see I2C errors.
  strain.setI2cClock(50000);
  strain.setSampleRate(NAU7802_RATE_320SPS);
  strain.setMuxSettleMicros(0);
  strain.setSignalAveraging(2);
  ui.showLcdMessage("Initialising the", "ADCs...");
  strain.begin();                    // discovers 9 NAU7802s, runs host-side tare
  ui.refresh();                      // restore the main menu on the LCD

  // Register experiments here. Each call appends to the runner's list
  // and (via attachRunner below) makes it appear in the LCD submenu.
  runner.registerExperiment(kExp1);
  runner.registerExperiment(kExp2);

  // Tell the UI and serial console about the runner so the
  // "Experiments" menu and Xstrain/Xrun/Xabort serial commands work.
  ui.attachRunner(runner);
  console.attachRunner(runner);

  // Lets the Xsignalavg edit screen read + write the strain array's
  // signal-averaging value at runtime.
  ui.attachStrain(strain);
  // Lets the serial console expose XsignalAverage/Xsamplerate (mirrors of
  // the LCD edit screens), so LCD-echoed commands are also re-runnable.
  console.attachStrain(strain);

  // Lets the runner refresh the LCD during motion (gotoExpansion blocks
  // ui.update(), so the runner's step callback is the only opportunity
  // to repaint the current target/step on the screen mid-motion).
  runner.attachUi(ui);

  // Boot with the step gate disabled (N=1, every step is eligible) and
  // the time gate at 0 (no minimum interval between rows). This is the
  // most permissive setting — emitRow fires on every motor step, motor
  // effectively ~10 steps/sec at sigAvg=2.
  //
  // Adjust from the LCD at runtime:
  //   - XlogEveryN  → step-modulo filter (N=2 → 1st/3rd/5th step, etc.)
  //   - Xsamplerate → NAU7802 conversion rate (10 / 20 / 40 / 80 / 320 SPS)
  //   - XsignalAverage → samples averaged per chip per CSV row
  //
  // For sparser captures on long moves, dial XlogEveryN to N=2/4/10
  // from the LCD; motor speed scales linearly with N until acquireRow
  // stops being the bottleneck.
  runner.setMotionLogPeriodMs(0);

  console.setBannerLine("|KisleyLab V2.2 |");
  console.begin();
}

void loop() {
  ui.update();
  console.update();
  runner.update();   // drives experiment state machine + strain streaming
}
