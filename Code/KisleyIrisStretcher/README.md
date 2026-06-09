# KisleyIrisStretcher

Arduino library for the **Kisley Lab iris stretcher** (Case Western
Reserve University) — a stepper-driven mechanical iris that radially
stretches a sample to a programmable expansion ratio while reading
nine NAU7802 strain-gauge ADCs for force feedback.

The library refactors what used to be a ~800-line monolithic sketch
(`IrisModule1.0.ino`) into seven composable C++ classes. It runs the
canonical rig with a single ~80-line sketch, lets other labs swap in
different geometry/wiring without touching library source, and ships
an "Experiments" framework so a researcher defines an expansion
sequence as a `float` array and the firmware drives the motor through
it while streaming synchronised CSV data.

---

## What it does

| Capability | API entry point |
|---|---|
| Drive the stepper to a target expansion ratio (CW or CCW) | `IrisStretcher::gotoExpansion(signedEx)` |
| Solve inverse kinematics (Ex → crank angle θ) | `IrisKinematics::findTheta(geo, Ex)` |
| LCD + encoder + buttons menu with editable Xspeed/Xgoto screens | `IrisMenuUI::begin/update` |
| Serial command parser (`Xgoto`, `Xrun`, `Xstrain`, …) | `IrisSerialConsole::begin/update` |
| Read 9 NAU7802 strain ADCs across 2 TCA9548A muxes with signal averaging + Bessel-corrected stdev | `IrisStrainArray::acquireRow` |
| Define + run an ordered expansion sequence with synchronised CSV output | `IrisExperimentRunner::registerExperiment` |

---

## Hardware

- **ESP32-S3** (Arduino-ESP32 core). `library.properties` declares
  `architectures=esp32` — other architectures are not supported.
- **HD44780 16×2 I²C LCD** (any backpack at 0x20–0x3F; auto-detected).
- **Rotary encoder** with detent, plus **MENU / DOWN / ACCEPT** buttons.
- **STEP/DIR stepper driver** (DRV8825, TMC2208, etc.).
- **9× NAU7802 strain ADCs** behind two **TCA9548A I²C muxes** at
  `0x71` and `0x73`. Default wiring: 5 ADCs on Mux A channels 0..4,
  4 ADCs on Mux B channels 0..3. Override with
  `IrisStrainArray::setLayout()`.

Default pinout (configurable):

| Function | ESP32-S3 pin |
|---|---|
| STEP / DIR | GPIO 12 / 13 |
| I²C SDA / SCL | GPIO 3 / 4 |
| MENU / DOWN / ACCEPT button | A0 / A1 / A2 |
| Encoder switch / B / A | A3 / A4 / A5 |

Buttons use `INPUT_PULLDOWN` (active-HIGH) by default. Set
`ui.setUseInternalPulldown(false)` for boards with external pulldowns.

---

## Install

**Arduino IDE → Sketch → Include Library → Add .ZIP Library…**, point
at this repo zipped, or symlink the repo into
`~/Documents/Arduino/libraries/`.

Dependencies (auto-installed via Library Manager or `library.properties`):

- `AccelStepper`
- `hd44780` (Bill Perry)
- `Adafruit NAU7802`
- `Adafruit BusIO` (transitive)

---

## Quick start

The shortest sketch that brings up the full rig and exposes one
experiment:

```cpp
#include <USB.h>
#include <KisleyIrisStretcher.h>
using namespace kisley::iris;

IrisStretcher        stretcher(/*STEP=*/12, /*DIR=*/13);
IrisStrainArray      strain;
IrisMenuUI           ui(stretcher, IrisMenuUI::Pins{A0, A1, A2, A5, A4, A3, 3, 4});
IrisSerialConsole    console(stretcher);
IrisExperimentRunner runner(stretcher, strain);

// One experiment: expand to 3.4× and return to center.
const float kExp1Targets[] = { 1.0f, 3.4f, 1.0f };
const IrisExperiment kExp1 = { "Exp1", kExp1Targets, 3 };

void setup() {
  USB.begin();
  Serial.begin(115200);
  while (!Serial && millis() < 2000) {}

  stretcher.begin();
  ui.begin();
  strain.begin();
  runner.registerExperiment(kExp1);
  ui.attachRunner(runner);
  console.attachRunner(runner);
  console.begin();
}

void loop() {
  ui.update();
  console.update();
  runner.update();
}
```

That's everything — the LCD now has all the built-in menu items plus
an **Experiments** entry containing `Exp1`, the serial port accepts
every `X…` command, and 9 strain channels are live.

`examples/IrisModule1.0/IrisModule1.0.ino` is the canonical Kisley-rig
version of this sketch with full about-screen / banner customisation.

---

## Library architecture

```
                            +---------------------+
                            |   loop() of sketch  |
                            +----------+----------+
                                       |
              +------------------------+------------------------+
              |                        |                        |
              v                        v                        v
       IrisMenuUI               IrisSerialConsole       IrisExperimentRunner
       (LCD + buttons)          (X-command parser)      (motion + logging
              |                        |                state machine)
              |                        |                        |
              +------------+-----------+--------+---------------+
                           |                    |
                           v                    v
                    IrisStretcher         IrisStrainArray
                    (facade)              (9 NAU7802s
                       |                   across 2 muxes)
              +--------+--------+
              v        v        v
        IrisStepper  IrisGeo  IrisKinematics
        Driver       metry    (computeEx,
        (STEP/DIR             findTheta)
        bit-bang)
```

The seven classes split cleanly along these lines:

| Class | Responsibility | Calls |
|---|---|---|
| `IrisGeometry` | POD of mechanical constants (`r0`, `rp`, `Y0`, gear ratio, `maxEx`, …) | — |
| `IrisKinematics` | Pure-math forward/inverse maps between θ and Eₓ (double precision; bisection + Newton-Raphson) | reads `IrisGeometry` |
| `IrisStepperDriver` | Blocking STEP/DIR bit-bang motion; keeps `AccelStepper` as the position counter | — |
| `IrisStretcher` | High-level facade: `gotoExpansion`, `goToZero`, `calibrate`, `setBladeSpeed`. Composes `IrisGeometry + IrisKinematics + IrisStepperDriver`. | — |
| `IrisMenuUI` | LCD + encoder + buttons state machine. Menu, edit screens (Xspeed/Xgoto), Help/About scrolling, Experiments submenu. | `IrisStretcher`, optional `IrisExperimentRunner` |
| `IrisSerialConsole` | `X<command>` line parser with built-ins + lab-registered commands. | `IrisStretcher`, optional `IrisExperimentRunner` |
| `IrisStrainArray` | Multi-mux NAU7802 reader. Round-robin signal averaging, host-side tare, CSV emit. | `Adafruit_NAU7802` |
| `IrisExperiment` / `IrisExperimentRunner` | POD descriptor + cooperative state machine that drives motion + logging. | `IrisStretcher`, `IrisStrainArray` |

Each class is also usable standalone — for example, a headless rig can
skip `IrisMenuUI` entirely and drive everything from `IrisSerialConsole`,
or a sketch that doesn't have strain ADCs simply doesn't instantiate
`IrisStrainArray`.

---

## Driving the motor

### Signed-Ex convention

`IrisStretcher::gotoExpansion(double signedEx)` is the one motion
primitive. Sign of the argument picks rotation direction:

| Input | Direction | Effect |
|---|---|---|
| `gotoExpansion(1.35)` | CW | `θ = +findTheta(1.35)` |
| `gotoExpansion(-1.35)` | CCW | `θ = -findTheta(1.35)` |
| `gotoExpansion(1.0)` (any sign) | center | `θ = 0` (returns to zeroed step) |
| `\|Ex\| ≥ maxEx` | error | logged to Serial, no motion |

The motor's step counter is **signed and absolute** — `currentSteps()`
goes up while moving CW, down while moving CCW. Round-tripping
`1.3 CW → 1.0 → 1.3 CW` lands on byte-identical step positions, and
`1.3 CW → 1.3 CCW` lands on exactly mirrored steps. See
`BIDIRECTIONAL_XGOTO.md` for the kinematic rationale.

### LCD UX

**Command echo.** Every action committed from the LCD/encoder GUI also
prints the equivalent serial command to the console — committing the
Xgoto edit screen emits `Xgoto 1.350 cw`, the Xspeed screen emits
`Xspeed 0.10`, running an experiment emits `Xrun Exp1`, and so on. The
echoed lines use the exact `X…` syntax `IrisSerialConsole` accepts, so a
serial log of a hands-on LCD session is a replayable command script.
Disable with `ui.setEchoCommands(false)` (e.g. to avoid one extra line in
an active experiment CSV stream when aborting from the LCD). The three
strain/logging edit screens echo `XsignalAverage <n>`, `Xsamplerate
<sps>`, and `XlogEveryN <n>` — now also accepted as serial commands (see
the command table above; the strain ones need `console.attachStrain`).

The `Xgoto` edit screen lets you scrub a magnitude in `[1.0, maxEx]`
with the encoder while the **DOWN** button toggles CW ↔ CCW, **SW**
toggles fine (0.001) / coarse (0.05) step size, and **ACCEPT** commits.
Display reads `"1.350 CW  [SW]"` or `"1.350 CCW [SW]"`.

There's also an **XsignalAverage** menu item that edits the strain array's
signal-averaging value (samples per ADC per CSV row) at runtime —
encoder ±1 in fine mode (SW toggles), ±10 in coarse, range `[1, 256]`,
ACCEPT commits. Only available when `ui.attachStrain(strain)` was called
in setup; without it the menu briefly shows "No strain attached".

**Xsamplerate** edits the NAU7802 conversion rate. The chip has five
discrete rates; the encoder cycles through them (10 / 20 / 40 / 80 /
320 SPS). ACCEPT commits — `setSampleRate()` stores the value, then
`applySampleRateLive()` pushes it to every initialised chip in the
array without a full `begin()`/re-tare (~50 ms total). Requires
`ui.attachStrain(strain)`.

**XlogEveryN** edits `IrisExperimentRunner::setMotionLogEveryNSteps()`
— the step-modulo filter on motion CSV emission. Encoder ±1 in fine
mode (SW toggles), ±10 in coarse, range `[1, 99]`, ACCEPT commits.
Setting N=2 logs only on the 1st, 3rd, 5th, … step of each
`gotoExpansion`. Requires `ui.attachRunner(runner)`.

### Serial commands

| Command | Effect |
|---|---|
| `Xgoto <Ex> [cw\|ccw]` | Move to magnitude `Ex` in `[1.0, maxEx]`. Direction defaults to `cw`. `Ex ≤ 1.0` returns to center. |
| `Xzero` | Drive the motor back to θ = 0 |
| `XsetZero` | Reset the position counter to 0 at the current pose |
| `Xcalibrate` | Run the calibration routine |
| `Xspeed <cm/s>` | Set blade speed (recomputes step delay) |
| `Xstrain` | Toggle continuous 9-ADC CSV streaming |
| `XsignalAverage <n>` | Samples averaged per ADC per row `[1..256]` (needs `console.attachStrain`) |
| `Xsamplerate <sps>` | NAU7802 conversion rate: `10/20/40/80/320` (needs `console.attachStrain`) |
| `XlogEveryN <n>` | Log one motion row per N steps `[1..99]` (needs `console.attachRunner`) |
| `Xrun <name>` | Run a registered experiment (`Xrun list` enumerates) |
| `Xabort` | Abort a running experiment (honoured between motion segments) |
| `Xhelp` | Print this list |

#### Structured reports — `Xgoto` / `Xzero` / `XsetZero` / `Xspeed`

These commands emit an aligned `key = value` block instead of free-form
chatter. `uptime_ms` is `millis()` at the start of the command — the same
clock as the strain/experiment CSV `t_ms`, so it lines up with the stream.

The `Xgoto` / `Xzero` / `XsetZero` reports are **intrinsic to
`IrisStretcher`** — they print whether the command came from the serial
console *or* the LCD GUI (`gotoExpansion` / `goToZero` / `setZeroHere` own
them). The experiment runner and the calibration routine suppress them via
`stretcher.setMoveReporting(false)` so internal moves don't corrupt the CSV
stream / clutter the routine; call `setMoveReporting(true)` to re-enable, or
`false` to silence them yourself. The `Xspeed` report is produced by the
serial console (the LCD's Xspeed edit screen still shows its own
confirmation).

```
Xgoto 1.35 cw     ->   === Xgoto ===
                         uptime_ms            = 1234567
                         current_position     = 0
                         target_expansion     = 1.350
                         resulting_expansion  = 1.350
                         direction            = cw
                         steps_required       = 1234
                         current_step_count   = 0
                         final_step_count     = 1234
                         blade_speed_cm_s     = 0.1000

XsetZero          ->   === XsetZero ===
                         uptime_ms            = 1234567
                         previous_position    = 4123
                         new_position         = 0
                         blade_speed_cm_s     = 0.1000

Xspeed 0.1        ->   === Xspeed ===
                         uptime_ms            = 1234567
                         blade_speed_cm_s     = 0.1000
                         step_half_period_us  = 21027
                         current_position     = 0
```

All reports share one formatter (`src/internal/ReportFormat.h`), so the
`key = value` columns stay aligned and every block looks identical. The
`=` column is parse-friendly: split on the first `=` and trim.

In the move report, `current_position` and `current_step_count` are the
same quantity (steps before the move) and `steps_required` is signed
(`final − start`), so `current_step_count + steps_required ==
final_step_count`. `direction` (`cw`/`ccw`) echoes the commanded `Xgoto`
token and appears only for expansion moves — not for `Xzero` or the
return-to-centre case. `XsetZero` only resets the counter (no motion), so it
reports just `previous_position` → `new_position`. `Xspeed` omits the
move-only fields since the motor doesn't move. Out-of-range / unsolvable
`Xgoto` still prints its error line instead of a report.

---

## Reading strain — `IrisStrainArray`

```cpp
IrisStrainArray strain;       // default = Kisley rig 9-slot layout

void setup() {
  strain.begin();             // discover muxes, init each NAU7802, host-side tare
}

void loop() {
  IrisStrainArray::Row row;
  strain.acquireRow(row);     // round-robin N samples, mean + stdev per chip
  strain.printCsvRow(Serial, row);
}
```

**Row contents** for each chip:

- `row.mean[i]` — round-half-away-from-zero integer mean of N raw samples
  minus the boot-time baseline (raw counts)
- `row.std[i]` — Bessel-corrected sample standard deviation (float)
- `row.present[i]` — false if the chip wasn't detected at `begin()`;
  CSV emits `NaN,NaN` for missing chips

**Configuration knobs** (call before `begin()`):

```cpp
strain.setSignalAveraging(8);     // N samples per chip per row (default 2)
strain.setTareSamples(32);        // baseline samples at boot (default 16)
strain.setI2cClock(10000);        // bus clock Hz (default 10 kHz — see Design notes)
strain.setSdaScl(3, 4);           // I²C pins
strain.setLayout(myLayout, 6);    // override the 9-slot Kisley default
strain.setSampleRate(NAU7802_RATE_320SPS);   // chip sample rate (default 320 SPS)
strain.setMuxSettleMicros(100);   // settle delay after mux channel-select (default 100 µs)
strain.setWaitForReadyOnRead(true); // poll conversion-ready bit before each read (default OFF)
```

**Sample rate options**: `NAU7802_RATE_10SPS` (lowest noise) through
`NAU7802_RATE_320SPS` (fastest, library default). Going from 10 to
320 SPS gives 32× faster per-chip conversion — the practical throughput
ceiling becomes the I²C clock instead. Dial back if you see excessive
sample-to-sample jitter.

CSV header / row format:

```
t_ms,ADC1_mean,ADC1_std,ADC2_mean,ADC2_std,…,ADC9_mean,ADC9_std
12345,−12,1.21,−8,0.95,…
```

---

## Experiments

An **experiment** is a named sequence of signed Ex targets. Hold time
between waypoints is a global setting on the runner (default 2 s).

### Defining

```cpp
const float kExp1Targets[] = { 1.0f, 3.4f, 1.0f };
const IrisExperiment kExp1 = { "Exp1", kExp1Targets, 3 };

const float kRoundTrip[] = { 1.0f, 2.0f, 1.0f, -2.0f, 1.0f };
const IrisExperiment kExp2 = { "Exp2", kRoundTrip, 5 };
```

Signed Ex semantics are the same as `gotoExpansion`:
positive = CW, negative = CCW, `|Ex| ≤ 1.0` = center.

### Registering

```cpp
runner.registerExperiment(kExp1);
runner.registerExperiment(kExp2);
ui.attachRunner(runner);          // adds an "Experiments" submenu to the LCD
console.attachRunner(runner);     // wires Xrun / Xabort / Xstrain
```

### Running

- **From the LCD**: main menu → `Experiments` → ACCEPT → scroll to
  the experiment → ACCEPT. Mid-run the LCD shows `Exp1 2/3` on row 0
  and `2.00 CW M` (or `H`) on row 1. **MENU** during run requests an
  abort.
- **From serial**: `Xrun Exp1`. Stream `Xrun list` to enumerate
  registered names. `Xabort` interrupts.

### Pre-run tare

Every `Xrun` and every `Xstrain` ON toggle re-runs the strain array's
host-side tare before the first CSV row is emitted. The rig should be
undisturbed at the moment you start the run — the captured baseline
becomes the zero for every row in that stream. (Boot-time tare in
`strain.begin()` is unchanged; the per-run tare is additive.)

### Output

One CSV stream per run, emitted to `Serial`. Header:

```
exp,t_ms,steps,target_ex,state,ADC1_mean,ADC1_std,…,ADC9_mean,ADC9_std
```

Per row:

```
Exp1,12345,4123,3.400,M,1234,1.21,−567,0.98,…
Exp1,12545,4280,3.400,M,1240,1.19,…
Exp1,14600,4525,3.400,H,1244,0.85,…
```

Columns:

- `exp` — experiment name, or `Xstrain` for continuous-streaming mode
- `t_ms` — `millis()` at sample acquisition
- `steps` — motor's signed absolute step count
- `target_ex` — magnitude of the current commanded Ex
- `state` — `M` (motor moving) / `H` (holding at waypoint) /
  `S` (continuous strain stream, no experiment)
- `ADCk_mean / ADCk_std` — what `IrisStrainArray::acquireRow` produced

### Timing knobs

```cpp
runner.setHoldMs(2000);                 // dwell at each waypoint
runner.setMotionLogPeriodMs(200);       // CSV row cadence while moving (time gate)
runner.setHoldLogPeriodMs(100);         // CSV row cadence while holding
runner.setMotionLogEveryNSteps(2);      // step gate: log only every Nth motor step
                                        // N=2 → 1st, 3rd, 5th, … step of each move
```

`setMotionLogEveryNSteps` is a **step-modulo** gate that combines with
the time gate (`setMotionLogPeriodMs`). A row is emitted only when both
gates pass — `stepIdx % N == 0` AND the time interval has elapsed.
For purely step-based emission, set `setMotionLogPeriodMs(0)`. Default
`N=1` disables the step gate (every step is eligible). The phase is
per-move: `stepIdx` resets to 0 at the start of each `gotoExpansion`,
so the first step of every waypoint move always passes.

### Custom run functions

For ramps, oscillations, or anything not expressible as a step array,
supply a `customRun` callback instead of `targets`:

```cpp
void runRamp(IrisExperimentRunner& r, void* /*user*/) {
  for (float ex = 1.1f; ex <= 3.0f; ex += 0.05f) {
    // call stretcher / emitRow / millis() yourself…
  }
}
const IrisExperiment kRamp = { "Ramp", nullptr, 0, runRamp };
```

See `examples/ExperimentDemo/` for a working `customRun` skeleton.

---

## Extending the UI / serial

Beyond experiments, the library accepts arbitrary menu items and
serial commands without library edits:

```cpp
void doThing(IrisMenuUI&, void*) {
  // your code
}

void cmdSweep(const char* args, void* /*user*/) {
  for (float ex = 1.1f; ex <= 3.0f; ex += 0.1f) stretcher.gotoExpansion(ex);
}

void setup() {
  /* usual init */
  ui.registerMenuItem("DoThing", doThing);
  console.registerCommand("sweep", "Run 1.1→3.0 sweep", cmdSweep);
}
```

The LCD menu grows automatically; `Xsweep` becomes a valid serial
verb. Up to 16 menu items and 8 custom commands by default
(`MAX_MENU_ITEMS` / `MAX_CUSTOM_CMDS` in their respective headers).

---

## Tuning geometry

If your rig has different link lengths or gear ratio, supply your own
`IrisGeometry` at construction:

```cpp
IrisGeometry myGeo;
myGeo.r0   = 6.5f;
myGeo.rp   = 5.0f;
myGeo.Y0   = -6.0f;
myGeo.g    = 20.0f;
myGeo.maxEx = 5.0f;
IrisStretcher stretcher(12, 13, myGeo);
```

`IrisKinematics::findTheta` automatically adapts — no code change.

---

## Design notes and gotchas

These are the non-obvious decisions baked into the library.
Several were dearly bought during debugging.

1. **I²C clock pinned at 10 kHz** for the strain array. The Kisley rig
   bus is signal-marginal at 100 kHz (long unshielded jumpers, weak
   downstream pullups on the muxes). 10 kHz works reliably; bus speed
   for the LCD on the same wires is unaffected.

2. **`Adafruit_I2CDevice::begin()` resets the Wire clock.** Internally
   it calls `Wire.begin()` with no args, which on ESP32 resets the
   clock to default 100 kHz. `IrisStrainArray` restores 10 kHz after
   every `nau.begin()` call.

3. **Deselect every mux before selecting any channel.** If you skip
   this, transitioning from Mux A ch4 → Mux B ch0 leaves Mux A still
   routing ch4 — two NAU7802s end up on the bus simultaneously and
   register reads collide. The library tracks the currently-active
   mux so the deselect pass is skipped when consecutive reads stay on
   the same mux (the dominant case in the round-robin sample loop);
   the deselect still happens on every cross-mux switch.

4. **Host-side tare, no `calibrate(OFFSET)`.** Adafruit's
   `Adafruit_NAU7802::calibrate()` has an inverted wait-loop
   (`while (!cal_start.read())` exits immediately because the bit was
   just written 1). The function returns before calibration completes,
   leaving the chip stuck mid-cal. `IrisStrainArray::tare()`
   averages N raw samples on the host side and subtracts.

5. **Kinematics in double precision.** The original sketch mixed
   `float`/`double` with `1e-10` Newton-Raphson tolerances that fell
   below float epsilon (~1.2e-7). Free on ESP32-S3 FPU; meaningful
   tolerances.

6. **Step counter is signed and absolute.** Every `gotoExpansion`
   computes `targetSteps` from absolute θ rather than as a delta, so
   repeated `1.3 → 1.0 → 1.3` round-trips land on identical step
   positions. Necessary for measurement repeatability.

7. **Experiments are cooperative.** Motion (one `gotoExpansion` per
   waypoint) is blocking, but hold time is non-blocking — the main
   `loop()` keeps polling UI + serial during holds. Aborts are
   detected between motion segments, not mid-stride. Future work:
   integrate an abort check inside `IrisStepperDriver::rotateThetaRadians`
   for instant abort.

---

## Repository contents

```
KisleyIrisStretcher/
├── library.properties               Arduino metadata (v2.0.0)
├── keywords.txt                     IDE syntax highlighting
├── README.md                        this file
├── BIDIRECTIONAL_XGOTO.md           why Xgoto accepts signed values
├── EXPERIMENTS_PLAN.md              design rationale for the experiments subsystem
├── SERIAL_OUTPUTS.md                catalogue of every serial output the firmware emits
├── SERIAL_CSV_FORMAT.md             parsing spec for the CSV data streams
├── src/
│   ├── KisleyIrisStretcher.h        umbrella header (one include for all)
│   ├── IrisGeometry.h
│   ├── IrisKinematics.h / .cpp
│   ├── IrisStepperDriver.h / .cpp
│   ├── IrisStretcher.h / .cpp
│   ├── IrisMenuUI.h / .cpp
│   ├── IrisSerialConsole.h / .cpp
│   ├── IrisStrainNAU7802.h / .cpp   single-chip ADC adapter
│   ├── IrisStrainArray.h / .cpp     9-chip strain array (NEW in v2)
│   ├── IrisExperiment.h / .cpp      experiment descriptor + runner (NEW in v2)
│   └── internal/
│       ├── DebouncedButton.h
│       └── QuadEncoder.h
├── examples/
│   ├── IrisModule1.0/               canonical Kisley rig firmware (boots Exp1 ready)
│   ├── ExperimentDemo/              shows Exp1 / Exp2 / Exp3 + customRun
│   ├── MinimalSerialOnly/           headless, no LCD
│   └── CustomGeometry/              geometry override demo
└── extras/
    └── StrainArray9/                standalone strain-array reference sketch
```

---

## License

MIT — see `LICENSE`.

---

## Lab

Built for the [Kisley Lab](https://engineering.case.edu/lab/kisely-lab),
Case Western Reserve University. The strain rig hardware verification
and most of the design feedback came from rig-side debugging sessions —
the "Design notes and gotchas" section is the post-mortem.

---

## Credits

Designed and developed with [Claude Code](https://claude.com/claude-code).
