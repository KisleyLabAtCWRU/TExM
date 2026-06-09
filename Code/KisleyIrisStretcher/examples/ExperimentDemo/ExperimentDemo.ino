// ExperimentDemo — defining and running iris-stretcher experiments.
//
// This sketch shows the minimum scaffolding for a lab to add its own
// experiments. The pattern is:
//   1. Define a `const float targets[]` array (signed Ex values).
//   2. Wrap it in an `IrisExperiment` descriptor with a name.
//   3. `runner.registerExperiment(myExp)` once in setup().
//   4. The entry automatically appears in the LCD "Experiments"
//      submenu and is runnable via `Xrun <name>` over serial.
//
// Three demo experiments are included below. Add your own anywhere
// between Exp3 and the registerExperiment calls in setup().

#include <USB.h>
#include <KisleyIrisStretcher.h>

using namespace kisley::iris;

// ---- Hardware pin map (Kisley rig defaults) ----
constexpr uint8_t PIN_STEP=12, PIN_DIR=13;
constexpr uint8_t PIN_MENU=A0, PIN_DOWN=A1, PIN_ACCEPT=A2;
constexpr uint8_t PIN_ENC_SW=A3, PIN_ENC_B=A4, PIN_ENC_A=A5;
constexpr uint8_t PIN_SDA=3, PIN_SCL=4;

IrisStretcher        stretcher(PIN_STEP, PIN_DIR);
IrisStrainArray      strain;
IrisMenuUI           ui(stretcher, IrisMenuUI::Pins{
                       PIN_MENU, PIN_DOWN, PIN_ACCEPT,
                       PIN_ENC_A, PIN_ENC_B, PIN_ENC_SW,
                       PIN_SDA, PIN_SCL});
IrisSerialConsole    console(stretcher);
IrisExperimentRunner runner(stretcher, strain);

// ===== Experiment definitions =====

// Exp1: simple expansion-and-return.
const float kExp1Targets[] = { 1.0f, 3.4f, 1.0f };
const IrisExperiment kExp1 = { "Exp1", kExp1Targets, 3 };

// Exp2: bidirectional round-trip (CW then CCW).
const float kExp2Targets[] = { 1.0f, 2.0f, 1.0f, -2.0f, 1.0f };
const IrisExperiment kExp2 = { "Exp2", kExp2Targets, 5 };

// Exp3: stepped ramp out and back.
const float kExp3Targets[] = { 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 2.5f, 2.0f, 1.5f, 1.0f };
const IrisExperiment kExp3 = { "Exp3", kExp3Targets, 9 };

// ===== Optional: a "customRun" experiment for non-step-array profiles.
// Shows how a lab could write an arbitrary motion routine while still
// using the runner's row-emission and abort plumbing. ============
void runOscillation(IrisExperimentRunner& r, void* /*user*/) {
  // 5 oscillations between 1.2x and 2.0x, with the runner emitting rows.
  for (int i = 0; i < 5; i++) {
    r.emitRow('M');   // an 'M' row at start of each cycle
    // (In a real custom run, you'd interleave gotoExpansion calls
    //  and emitRow('M'/'H') yourself. Kept minimal here for brevity.)
  }
}
const IrisExperiment kOscillation = { "Oscillate", nullptr, 0,
                                      runOscillation, nullptr };

void setup() {
  delay(500);
  USB.begin();
  Serial.begin(115200);
  while (!Serial && millis() < 2000) {}

  stretcher.begin();

  ui.setUseInternalPulldown(true);
  ui.begin();
  ui.showLcdMessage("Initialising the", "ADCs...");
  strain.begin();
  ui.refresh();

  // Tune the runner if you want different timing.
  runner.setHoldMs(2000);                // dwell at each waypoint, ms
  runner.setMotionLogPeriodMs(200);      // CSV row every 200 ms while moving
  runner.setHoldLogPeriodMs(100);        // CSV row every 100 ms while holding

  // Register experiments. Each appears as a menu entry.
  runner.registerExperiment(kExp1);
  runner.registerExperiment(kExp2);
  runner.registerExperiment(kExp3);
  runner.registerExperiment(kOscillation);

  ui.attachRunner(runner);
  console.attachRunner(runner);
  ui.attachStrain(strain);      // enables the LCD Xsignalavg edit screen
  console.attachStrain(strain); // enables XsignalAverage/Xsamplerate serial verbs
  runner.attachUi(ui);          // lets the LCD refresh during motion

  console.setBannerLine("|KisleyLab V2.1 |");
  console.begin();
}

void loop() {
  ui.update();
  console.update();
  runner.update();
}
