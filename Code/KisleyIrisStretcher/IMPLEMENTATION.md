# KisleyIrisStretcher — Implementation Plan

> **Decisions locked in (2026-05-12):**
> - Library name: **`KisleyIrisStretcher`**. The correct lab spelling is
>   **Kisley Lab**.
> - **`FW_LABLINE1 "Kisely Lab"` in the current sketch is a typo** and will
>   be corrected to `"Kisley Lab"` during the port.
> - New **top-level GitHub repo** (separate from `Blanchedubois/IrisStretcher`).
> - **Keep AccelStepper** as the position counter — do not drop the dep.

---

## 1. Goals

1. Refactor `IrisModule1.0.ino` (794 lines, monolithic) into a reusable Arduino
   library called **`KisleyIrisStretcher`** that other labs can drop into their
   `~/Arduino/libraries/` folder and use without copying source files around.
2. Expose **simple, high-level calls** for the things a lab actually does:
   move to expansion, zero, calibrate, set speed, attach a strain ADC.
3. Ship a **default example sketch** (`examples/IrisModule1.0/IrisModule1.0.ino`)
   that **behaves identically to the current monolithic sketch** — same LCD
   menu, same encoder feel, same serial commands, same splash/banner, same
   geometry. A user flashing it should not be able to tell the difference.
4. Leave **clean extension seams** (callbacks, registerable menu items,
   registerable serial commands) so a follow-up "Experiments" subsystem can
   plug in without re-touching library internals. *Do not implement that
   subsystem yet — just leave the seams.*

## 2. Non-goals (for this pass)

- No experiments framework yet. Hooks only.
- No new motion features (acceleration profiles, homing, limit switches).
- No protocol changes — serial commands stay byte-identical.
- No board-portability work beyond ESP32-S3 (the target HW). Pin defaults
  remain ESP32-S3.

---

## 3. Library layout

```
KisleyIrisStretcher/
├── library.properties              # Arduino IDE metadata
├── keywords.txt                    # syntax highlighting in IDE
├── README.md                       # quick-start for lab users
├── LICENSE
├── src/
│   ├── KisleyIrisStretcher.h       # umbrella header — single include
│   ├── IrisGeometry.h              # POD struct of geometry constants
│   ├── IrisStepperDriver.h/.cpp    # STEP/DIR bit-bang + position book-keeping
│   ├── IrisKinematics.h/.cpp       # computeEx + findTheta (pure math, no HW)
│   ├── IrisStretcher.h/.cpp        # main facade — combines kinematics+stepper
│   ├── IrisMenuUI.h/.cpp           # LCD + encoder + buttons state machine
│   ├── IrisSerialConsole.h/.cpp    # "X*" command parser
│   ├── IrisStrainNAU7802.h/.cpp    # optional ADC adapter
│   └── internal/
│       ├── DebouncedButton.h
│       └── QuadEncoder.h
└── examples/
    ├── IrisModule1.0/              # default — full parity replacement
    │   └── IrisModule1.0.ino
    ├── MinimalSerialOnly/          # headless: serial-only, no LCD/encoder
    │   └── MinimalSerialOnly.ino
    └── CustomGeometry/             # shows overriding geometry constants
        └── CustomGeometry.ino
```

`library.properties` minimum:

```
name=KisleyIrisStretcher
version=1.0.0
author=Tejasvin Shrikanth <tejasvin.shrikanth@gmail.com>
maintainer=Kisely Lab, CWRU
sentence=Drive the Kisely Lab iris stretcher: motion, kinematics, UI, strain.
paragraph=High-level API + drop-in example that reproduces IrisModule1.0.
category=Device Control
url=https://github.com/Blanchedubois/IrisStretcher
architectures=esp32
depends=AccelStepper,hd44780,Adafruit NAU7802
```

---

## 4. Public API

### 4.1 `IrisGeometry` (POD)

Holds the constants currently sprinkled at the top of the .ino. A lab with
different physical hardware overrides the struct; the defaults match the
current rig.

```cpp
struct IrisGeometry {
  float r0     = 7.1f;    // mm
  float rp     = 5.25f;   // mm
  float Y0     = -6.1f;   // must be negative
  float rPin   = 0.5f;    // pin offset, mm
  float g      = 15.0f;   // gear ratio
  float maxEx  = 4.2f;    // safety clamp on Xgoto
  uint16_t pulsesPerRev = 1600;
  float E0() const { return r0 - rp; }
};
```

### 4.2 `IrisStretcher` (main facade)

```cpp
class IrisStretcher {
public:
  IrisStretcher(uint8_t stepPin, uint8_t dirPin,
                const IrisGeometry& geo = IrisGeometry{});

  void begin();

  // High-level moves — what 95% of lab code will call.
  bool gotoExpansion(float targetEx);   // false if unreachable / out of range
  void goToZero();                      // drive θ -> 0
  void setZeroHere();                   // reset internal counter
  void calibrate();                     // current canned routine
  void setBladeSpeed(float cmPerSec);   // sets stepDelayUs internally

  // Low-level escape hatches.
  void  rotateThetaRadians(float theta);
  long  currentSteps() const;
  float currentTheta() const;
  float computeEx(float theta) const;
  float findTheta(float targetEx) const; // returns NaN if no solution

  // Extension seams — see §8.
  using StepCallback = void(*)(long stepIndex, void* user);
  void onEachStep(StepCallback cb, void* user = nullptr);

  // Diagnostic logging toggle (replaces the global `strainData` bool).
  void setStepLogging(bool on);
};
```

### 4.3 `IrisMenuUI`

```cpp
class IrisMenuUI {
public:
  struct Pins {
    uint8_t btnMenu, btnDown, btnAccept;
    uint8_t encA, encB, encSW;
    uint8_t sda = 3, scl = 4;
  };

  IrisMenuUI(IrisStretcher& stretcher, const Pins& pins);

  void begin();    // initializes LCD, scans I2C, sets pinModes
  void update();   // call every loop() — non-blocking

  void setInvertEncoder(bool on);
  void setUseInternalPulldown(bool on);  // default true

  // Future experiments hook — see §8.
  // Adds a custom item to the rotating menu list.
  using ActionCallback = void(*)(void* user);
  bool registerMenuItem(const char* label,
                        ActionCallback cb, void* user = nullptr);
};
```

### 4.4 `IrisSerialConsole`

```cpp
class IrisSerialConsole {
public:
  IrisSerialConsole(IrisStretcher& stretcher, Stream& io = Serial);

  void begin(unsigned long baud = 115200);
  void update();   // non-blocking line reader + dispatcher

  // Future experiments hook.
  using CommandCallback = void(*)(const char* args, void* user);
  bool registerCommand(const char* name,        // "myexp"  -> "Xmyexp"
                       const char* helpLine,
                       CommandCallback cb,
                       void* user = nullptr);
};
```

### 4.5 `IrisStrainNAU7802` (optional)

Wraps Adafruit_NAU7802 and bakes in the **Wire pin-reset workaround** noted in
prior memory (`Adafruit_NAU7802::begin()` re-pins ESP32 I2C; we restore them
after).

```cpp
class IrisStrainNAU7802 {
public:
  bool begin(uint8_t sda = 3, uint8_t scl = 4);   // restores pins after
  float readVolts(uint8_t averages = 35);
  float referenceVoltage() const { return 3.0f; }
};
```

### 4.6 Umbrella header

`KisleyIrisStretcher.h` simply re-includes the per-class headers so user
sketches need only `#include <KisleyIrisStretcher.h>`.

---

## 5. The default example sketch

`examples/IrisModule1.0/IrisModule1.0.ino` must **reproduce the current
sketch's behavior exactly.** Target shape:

```cpp
#include <KisleyIrisStretcher.h>

IrisStretcher  stretcher(/*STEP=*/12, /*DIR=*/13);
IrisMenuUI     ui(stretcher, { /*MENU*/A0, /*DOWN*/A1, /*ACCEPT*/A2,
                               /*ENC_A*/A5, /*ENC_B*/A4, /*ENC_SW*/A3 });
IrisSerialConsole console(stretcher);

void setup() {
  console.begin(115200);
  stretcher.begin();
  ui.begin();
}

void loop() {
  ui.update();
  console.update();
}
```

That's the whole sketch — ~15 lines vs 794. Everything else lives in the
library.

---

## 6. Behavior-parity checklist

Every item below is currently observable in `IrisModule1.0.ino` and must
remain observable after the refactor. **Tick these off during the port.**

| # | Behavior | Source line(s) | Notes |
|---|---|---|---|
| 1 | Splash screen "Iris Stretcher" + "Kisely Lab" 1500 ms | 416–423 | |
| 2 | LCD I2C auto-scan 0x20–0x3F, prefer 0x27/0x3F | 302–351 | |
| 3 | Show "LCD @ 0xXX" for 1000 ms after splash | 426–436 | |
| 4 | ASCII-art banner + command help on serial | 452–467 | |
| 5 | Menu items in this exact order: XsetZero, Xzero, Xcalibrate, Xspeed, Xgoto, Xhelp, Xabout | 99–100 | |
| 6 | Encoder rotates menu and edits values | 696–778 | |
| 7 | DOWN button advances menu, ACCEPT runs item, MENU toggles backlight | 711–713 | |
| 8 | Edit modes show " F"/" C" + "[SW]" suffix | 236–254 | |
| 9 | Xspeed fine=0.01, coarse=0.10; range [0.1, 5.0] | 104–107 | |
| 10 | Xgoto fine=0.001, coarse=0.050; range [1.001, maxEx] | 111–117 | |
| 11 | SW toggles fine/coarse | 722–725, 751–754 | **bug — see §7.1** |
| 12 | Xhelp scrolls 6 frames @ 1200 ms each | 256–264 | |
| 13 | Xabout scrolls 6 frames | 265–285 | |
| 14 | Serial commands: Xgoto, Xzero, XsetZero, Xcalibrate, Xspeed, Xhelp | 645–693 | |
| 15 | `θ` solver: bisection 50 iter then Newton 50 iter, NaN/Inf revert | 508–561 | |
| 16 | `rotateThetaRadians` bit-bangs STEP at stepDelayUs each half | 563–596 | |
| 17 | Position bookkeeping via AccelStepper.currentPosition() | 72, 591–592 | Internal — labs don't see |
| 18 | `omega = bladeSpeed / 13.6`, delay formula at 632–637 | 632–642 | |
| 19 | Pulldown vs INPUT config flag | 401–411 | Expose via `setUseInternalPulldown` |
| 20 | Encoder invert flag | 25, 698–700 | Expose via `setInvertEncoder` |
| 21 | NAU7802 disabled by default, USE_NAU7802 path compiles | 11–16, 577–587 | Becomes optional include |
| 22 | strainData=true prints "Steps, i" per step | 50, 577–590 | Expose via `setStepLogging` |
| 23 | Default stepDelayUs=2000 µs | 68 | |

A simple validation harness: after porting, compare serial output of a fresh
`Xcalibrate` run on both firmwares byte-for-byte (excluding timestamps).

---

## 7. Bugs to fix during the refactor

These are pre-existing bugs in `IrisModule1.0.ino` that the library version
should fix (and **call out in the changelog** so a lab updating from the
monolithic sketch knows about behavior changes).

### 7.1 Double `fell()` consumption in edit screens

Lines 722+738 (Xspeed) and 751+768 (Xgoto) both call `btnEncSW.fell()`.
`DebouncedButton::edge()` consumes the state transition on the first call —
so the commit branch can never fire from the encoder switch press; only the
ACCEPT button commits. The mode-toggle branch always wins.

**Fix:** cache the edge once per `update()`:
```cpp
uint8_t encEdge = btnEncSW.edge();
bool encPressed = (encEdge == 2);
```
then use `encPressed` in both branches.

### 7.2 `float`/`double` mixing in kinematics

`computeEx` takes `float`, `findThetaHighPrec` takes `double`, the Newton
loop's `tolF=1e-10`/`tolTheta=1e-10` are below float epsilon. Either lift
everything to `double` or drop tolerances to ~1e-6. Recommend **all double**
since ESP32-S3 has hardware FP for both.

### 7.3 NAU7802 re-pinning I2C

If `USE_NAU7802` is enabled, `nau.begin()` re-pins I2C and breaks the LCD.
The library version should call `Wire.begin(sda, scl)` again after every
NAU init call. (Already in saved memory `project_iris_wire_pin_reset`.)

### 7.4 Blocking `delay()` in drawHelp/drawAbout

~7–8 s of `delay()` freezes the encoder and serial reader. Replace with a
non-blocking page sequencer in `IrisMenuUI` that advances frames in `update()`
based on `millis()`.

### 7.5 Serial flood during `Xgoto`

`Serial.print("Steps, ")` every microstep at 250 steps/s saturates the link
for long moves. Library default: log every Nth step (N defaulting to 1 for
parity, but exposed via `setStepLogging(bool, uint16_t every = 1)`).

### 7.6 `AccelStepper` retained as the position counter

It's never `.run()`-driven (we bit-bang STEP/DIR directly), but we are
**keeping** the AccelStepper dependency per the lab's decision — it stays as
the canonical position counter inside `IrisStepperDriver` exactly as it is
today. Not a bug; noted here so the refactor doesn't accidentally drop it.

---

## 8. Extension seams for the future Experiments feature

Do **not** implement experiments yet, but leave these hooks in place so they
can be added without re-touching library internals:

| Seam | What it enables later | Where it lives |
|---|---|---|
| `IrisMenuUI::registerMenuItem(label, cb)` | An experiment can add itself as a menu entry | `IrisMenuUI` |
| `IrisSerialConsole::registerCommand(name, help, cb)` | An experiment defines `Xmyexp` | `IrisSerialConsole` |
| `IrisStretcher::onEachStep(cb)` | Strain logging, force feedback, datalog | `IrisStretcher` |
| `IrisStrainNAU7802` as a separate class | Experiments can sample without owning motion | `IrisStrainNAU7802` |
| `IrisGeometry` as POD parameter | Rigs with different geometry work without forking | `IrisGeometry` |

Keep callback signatures `void(*)(args, void* user)` (not `std::function`) to
stay friendly to AVR/SAMD ports later if any.

---

## 9. Migration steps (suggested order)

Each step is a self-contained change; the sketch should still compile and run
after every step.

1. **Scaffold the library folder** with `library.properties`, empty
   per-class headers, and a `KisleyIrisStretcher.h` umbrella header.
2. **Move pure-math code** (`computeEx`, `findThetaHighPrec`) into
   `IrisKinematics.cpp`. No HW dependency. Unit-testable on host.
3. **Move stepper bit-bang** into `IrisStepperDriver`. Drop AccelStepper.
4. **Build `IrisStretcher`** facade that composes the two above. Port
   `setAngularSpeed`, `goToZero`, `calibrate`, `setZeroHere`.
5. **Move LCD + encoder + buttons** into `IrisMenuUI`. Lift the I2C scanner.
   Fix bug 7.1, 7.4 along the way.
6. **Move serial parser** into `IrisSerialConsole`. Keep command names
   identical.
7. **Move NAU7802** into `IrisStrainNAU7802` with the Wire-pin restore.
8. **Write the default example** — must compile and produce parity with
   §6 checklist.
9. **Write `MinimalSerialOnly` and `CustomGeometry` examples** to validate
   the seams.
10. **README.md** with wiring diagram reference, the 15-line sketch, and
    a table of public API.
11. **Tag v1.0.0**, push to `github.com/Blanchedubois/IrisStretcher` (per
    saved memory `reference_iris_stretcher_repo`) as a subdirectory or a
    sibling repo — open question, see §10.

## 10. Open questions

Resolved (2026-05-12):

- ~~Repo layout~~ → new top-level repo `KisleyIrisStretcher`.
- ~~Kisley vs Kisely~~ → **Kisley** is correct; firmware string is a typo.
- ~~AccelStepper retention~~ → **keep it**.

Still open:

1. **License** — MIT? BSD-3? Lab default?
2. **`maxEx` as compile-time vs runtime.** Currently `const float`. Move to
   `IrisGeometry` (runtime) so labs with longer travel don't recompile?
3. **Calibration routine.** Current `stepperMotorCalibration` has hardcoded
   `0.7` / `−0.287` rad moves. Keep as default but let labs override via
   `stretcher.setCalibrationRoutine(cb)`?

---

## 11. Definition of done

- [ ] Library installs via "Add .ZIP Library…" in Arduino IDE.
- [ ] `examples/IrisModule1.0/IrisModule1.0.ino` compiles and runs.
- [ ] All §6 checklist items verified on hardware.
- [ ] All §7 bugs fixed; changelog entry written.
- [ ] README.md has the 15-line sketch and an API summary table.
- [ ] `keywords.txt` covers every public symbol.
- [ ] Library passes `arduino-cli lint` with no errors.
