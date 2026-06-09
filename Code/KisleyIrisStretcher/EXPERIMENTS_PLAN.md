# Experiments + Strain Logging — Implementation Plan

> **Goal.** Bring the working `StrainArray9` sketch into the
> `KisleyIrisStretcher` library as first-class capability and add an
> "Experiments" subsystem so labs can define ordered expansion profiles
> that run end-to-end, with synchronised motor + 9-channel strain data
> streaming to USB serial.

## Locked decisions (2026-05-12)

1. **Sampling cadence during motion**: time-based, every 200 ms.
   Configurable via `runner.setMotionLogPeriodMs(uint32_t)`.
2. **Hold time is a single global value**, not per-waypoint. Default
   2000 ms, settable via `runner.setHoldMs(uint32_t)`. Therefore the
   `ExpansionStep` struct collapses to just `float targetEx`, and an
   experiment's `steps` is effectively a `const float* targets`.
3. **Step column = motor cumulative step count**, signed (positive when
   the motor rotated CW, negative when CCW). Preserves the
   absolute-position invariant the lab already relies on.
4. **`Xstrain` and motion commands coexist.** State column (`M`/`H`)
   reflects current motor activity. Streaming is independent of motion.
5. **Experiments live in a submenu.** Main menu gets a new
   "Experiments" entry that pushes into a `UiState::EXPERIMENTS_MENU`
   state listing all registered experiments. MENU button pops back.

Decisions I'm making solo (yell if any are wrong):

6. CSV header emitted **once per stream** (per `Xstrain`/`Xrun` start).
7. Default experiments live in the **example sketch only**, not in the
   library.
8. Default strain-array layout = verified Kisley rig (5 on 0x71, 4 on 0x73).
9. Both `IrisStrainArray` (multi-mux, new) and `IrisStrainNAU7802`
   (single-chip, existing) stay around.
10. `customRun` callback on `IrisExperiment` kept for ramps/oscillations;
    untested code paths but zero-cost when null.

---

## 1. What the lab will get

Two new capabilities exposed through the library API:

1. **Continuous strain logging** — `IrisStrainArray` reads all 9 NAU7802
   ADCs through the two TCA9548A muxes and streams CSV rows on demand.
   Reachable from a serial command (`Xstrain`), an LCD menu item
   ("Strain"), or directly from sketch code.
2. **Experiments** — a lab defines a named sequence of `(targetEx,
   holdMs)` waypoints. Selecting an experiment from the LCD menu (or
   sending `Xrun <name>` over serial) runs the motor through the
   sequence while every motor step + every settled hold sample is
   logged as a CSV row with `t_ms`, motor `steps`, target `Ex`, and all
   9 ADCs' mean + stdev.

Both reuse the same underlying `IrisStrainArray::acquireRow()`
machinery — exactly the round-robin signal-averaging logic from the
current `StrainArray9.ino`, just promoted into a class.

---

## 2. New components

### 2.1 `IrisStrainArray` — multi-mux strain reader

Port of `StrainArray9` into a library class. Lives at
`src/IrisStrainArray.h` / `.cpp`.

```cpp
class IrisStrainArray {
public:
  static constexpr uint8_t  MAX_ADCS = 16;
  static constexpr uint16_t MAX_AVG  = 256;

  struct Slot { uint8_t muxAddr; uint8_t ch; const char* label; };

  struct Row {
    uint32_t t_ms;
    int32_t  mean[MAX_ADCS];
    float    std [MAX_ADCS];
    bool     present[MAX_ADCS];
  };

  IrisStrainArray(TwoWire& wire = Wire, Stream& io = Serial);

  // -- Configuration: call before begin() --
  void setLayout(const Slot* slots, uint8_t n);   // default = 9-slot Kisley rig
  void setI2cClock(uint32_t hz);                  // default 10 kHz
  void setSignalAveraging(uint16_t n);            // default 4
  void setTareSamples(uint16_t n);                // default 16
  void setSdaScl(uint8_t sda, uint8_t scl);       // default 3, 4

  // -- Lifecycle --
  bool begin();           // Wire.begin, discover muxes, init+tare every chip
  bool rescan();          // re-init + re-tare without rebooting
  void tare();             // re-tare only (motor + sample must be at rest)

  // -- Acquisition --
  void acquireRow(Row& out);   // round-robin N samples, mean + Bessel stdev

  // -- Inspection --
  uint8_t       adcCount() const;
  bool          isPresent(uint8_t i) const;
  const char*   label(uint8_t i) const;
  int32_t       baseline(uint8_t i) const;
  uint16_t      signalAveraging() const;

  // -- Emitters --
  void printCsvHeader(Stream& out) const;
  void printCsvRow(Stream& out, const Row& row) const;
  void printStatus(Stream& out) const;
};
```

Defaults match the verified rig: muxes at 0x71/0x73, 5 ADCs on Mux A
ch0..4, 4 ADCs on Mux B ch0..3, 10 kHz I²C clock, host-side tare,
no on-device OFFSET cal (Adafruit's `calibrate()` is broken — see
saved memory `feedback_adafruit_nau7802_calibrate_bug`).

### 2.2 `IrisExperiment` — POD describing one experiment

```cpp
struct ExpansionStep {
  float    targetEx;   // signed: positive = CW, negative = CCW
                       // |targetEx| ∈ [1.0, maxEx]; |Ex| ≤ 1 = center
  uint32_t holdMs;     // dwell time at this Ex before advancing
};

struct IrisExperiment {
  const char*           name;          // shown on LCD, used by `Xrun <name>`
  const ExpansionStep*  steps;
  uint8_t               nSteps;

  // Optional alternative: a custom run function for non-step profiles
  // (ramps, oscillations, etc.). If non-null, this is called instead
  // of walking through `steps[]`.
  using RunFn = void(*)(class IrisExperimentRunner& r, void* user);
  RunFn   customRun = nullptr;
  void*   customUser = nullptr;
};
```

A lab writes a static array of `ExpansionStep` and an `IrisExperiment`
descriptor that points at it. Example:

```cpp
const ExpansionStep kExp1Steps[] = {
  {1.0f,    500},   // start at center
  {3.4f,    2000},  // expand to 3.4x CW, hold 2 s
  {1.0f,    500},   // back to center
};
const IrisExperiment kExp1 = {"Exp1", kExp1Steps, 3};
```

Adding a new experiment is one new array + one descriptor + one
registration call — no library edits needed.

### 2.3 `IrisExperimentRunner` — orchestrates one run

```cpp
class IrisExperimentRunner {
public:
  IrisExperimentRunner(IrisStretcher& stretcher,
                       IrisStrainArray& strain,
                       Stream& io = Serial);

  // Logging cadence
  void setLogEveryNSteps(uint32_t n);   // 0 = don't log during motion
  void setHoldLogPeriodMs(uint32_t ms); // 0 = don't log during hold

  // Run an experiment. Blocks until done or interrupted.
  // Returns true on completion, false on interrupt.
  bool run(const IrisExperiment& exp);

  // Called from interrupt context / button handler to abort
  void requestAbort();
};
```

During a run it:
1. Prints a CSV header.
2. For each `ExpansionStep`:
   a. Prints `# step k/N: targetEx = X (CW|CCW), hold Yms`
   b. Calls `stretcher.gotoExpansion(targetEx)` with an `onEachStep`
      callback that emits a CSV row every `logEveryNSteps` steps.
   c. Starts a hold timer for `holdMs`; while held, emits a CSV row
      every `holdLogPeriodMs`.
3. Prints `# experiment complete`.

The user's existing step-count invariant (the motor's signed absolute
step counter, so multiple expansions land on identical positions) is
preserved end-to-end — every CSV row carries `stretcher.currentSteps()`.

### 2.4 `IrisMenuUI` additions

```cpp
// Add to IrisMenuUI:
bool registerExperiment(const IrisExperiment& exp,
                        IrisExperimentRunner& runner);
```

Internally calls `registerMenuItem(exp.name, runExperimentCallback,
&context)`. The context bundles the experiment + runner pointers. From
the LCD menu, scrolling to "Exp1" and pressing ACCEPT starts it; the
MENU button mid-run requests an abort.

During a run the LCD shows:
- Row 0: `Exp1 3/5` (experiment name + which step we're on)
- Row 1: `1.50 CW H` (target Ex + direction + state: M=moving, H=holding)

### 2.5 `IrisSerialConsole` additions

Two new built-in commands:

- `Xstrain` — toggle continuous strain streaming on/off (independent of
  motor; useful for static-load characterisation).
- `Xrun <name>` — start the experiment whose `.name` matches (case-
  sensitive). `Xrun list` prints every registered experiment.
- `Xabort` — abort a running experiment.

Implemented by registering them via the same `registerCommand` seam
that already exists.

---

## 3. CSV output format

One header line at start of each stream:

```
exp,t_ms,steps,target_ex,state,ADC1_mean,ADC1_std,ADC2_mean,ADC2_std,...,ADC9_mean,ADC9_std
```

Rows:

```
Exp1,12345,4123,3.400,M,1234,1.21,-567,0.98,...
Exp1,12380,4200,3.400,M,1240,1.19,-562,1.02,...
Exp1,14500,4525,3.400,H,1244,0.85,-560,0.78,...
```

- `exp`     — name from `IrisExperiment.name`; or `Xstrain` for the
              continuous-streaming mode (no experiment).
- `t_ms`    — `millis()` at the moment the row was acquired.
- `steps`   — motor's signed absolute step count.
- `target_ex` — magnitude of the current commanded Ex (always positive).
- `state`   — `M` = motor moving toward target, `H` = holding at target.
- `ADCk_mean/std` — round-half-away-from-zero mean and Bessel-corrected
              sample stdev over `signalAveraging` raw samples.

Missing ADCs emit `NaN,NaN` so the column count stays constant.

---

## 4. Data-flow during a step

```
gotoExpansion(targetEx)
     │
     │  (drives motor; per-step onEachStep callback)
     │       │
     │       └─► every N motor steps:
     │              strain.acquireRow(row)
     │              print "M" row
     │
     │  (motor arrives at target)
     │
hold for holdMs
     │
     └─► every holdLogPeriodMs:
            strain.acquireRow(row)
            print "H" row
```

Sampling during motion is throttled by `logEveryNSteps` because the
motor pulses fast (250 steps/s at default `stepDelayUs = 2000` µs) but
`strain.acquireRow()` takes ~`SIGNAL_AVERAGING × 100 ms` at 10 SPS —
sampling every motor step would be impossible. Default
`logEveryNSteps = 50` gives a row every ~200 ms during motion.

During hold, the motor is parked and `acquireRow` runs back-to-back,
spaced by `holdLogPeriodMs`. Default 100 ms = 10 Hz.

---

## 5. File-by-file change summary

| File | Change |
|---|---|
| `src/IrisStrainArray.h` / `.cpp` | **NEW** — multi-mux strain reader, ported from `StrainArray9.ino` |
| `src/IrisExperiment.h` / `.cpp` | **NEW** — `ExpansionStep`, `IrisExperiment`, `IrisExperimentRunner` |
| `src/IrisMenuUI.h` / `.cpp` | add `registerExperiment(...)` convenience wrapper; add running-experiment LCD render path |
| `src/IrisSerialConsole.h` / `.cpp` | add `Xstrain`, `Xrun <name>`, `Xabort` built-ins; add a list of known experiments for `Xrun list` |
| `src/KisleyIrisStretcher.h` | include the two new headers |
| `library.properties` | bump version to 2.0.0; depends gets `Adafruit BusIO, Adafruit Unified Sensor` if not already pulled in |
| `examples/IrisModule1.0/IrisModule1.0.ino` | **UPDATE** — defines and registers `Exp1` (1x→3.4x→1x) so the canonical Kisley rig firmware ships with the example experiment selectable from the LCD. Otherwise unchanged. |
| `examples/ExperimentDemo/ExperimentDemo.ino` | **NEW** — defines `Exp1` (1x→3.4x→1x) and `Exp2` (1x→2.0x→1x→-2.0x→1x), registers both, demonstrates the pattern for labs adding their own |
| `extras/StrainArray9/StrainArray9.ino` | unchanged — kept as a standalone reference sketch that doesn't depend on the rest of the library |
| `EXPERIMENTS_PLAN.md` | this file |
| `README.md` | add an Experiments section with the 4-line "how to add one" recipe |

Nothing existing changes signature. `IrisStrainNAU7802` (the single-chip
adapter) stays around for users who only have one ADC.

---

## 6. The "how to add an experiment" UX

This is the central UX promise — must stay this short:

```cpp
// 1. Define the waypoints (Ex value, hold time in ms).
const ExpansionStep kMyExpSteps[] = {
  {1.0f,  500},
  {2.5f, 1500},
  {1.0f,  500},
  {-2.5f, 1500},   // negative = CCW
  {1.0f,  500},
};
const IrisExperiment kMyExp = { "MyExp", kMyExpSteps, 5 };

// 2. Register in setup().
ui.registerExperiment(kMyExp, runner);
```

That's it. New entry appears in the LCD menu after "Xabout".

For experiments that need ramps, oscillations, or any control flow more
complex than step-and-hold, the lab supplies a `customRun` function
instead of the `steps` array.

---

## 7. Memory & concurrency considerations

- All experiment data is built from `const` arrays; no heap allocation.
- Menu items max out at 16 (7 built-in + 9 user). If labs want more,
  bump `MAX_MENU_ITEMS` in `IrisMenuUI.h`.
- The runner is **blocking** — it owns the loop while running. The LCD
  poll, serial poll, and `requestAbort` are checked in a tight loop
  during hold time; abort is detected within one row interval.
- The button-debouncer state in `IrisMenuUI` is checked once per row
  during hold, so up to ~100 ms latency on abort.

---

## 8. Validation plan

After implementation, smoke tests:

1. **Strain-only**: `Xstrain` → rows stream at the configured averaging
   rate. `Xstrain` again → stops.
2. **Exp1 dry run**: register `Exp1` from §2.6, select from LCD, watch
   serial. Should see motion (M-state rows) and 3 hold periods (H-state
   rows). Final motor position back to step 0.
3. **CCW experiment**: register Exp2 with a CCW waypoint, run, confirm
   motor reverses direction physically and CSV target_ex shows magnitude
   only while state shows direction implicit in motor steps.
4. **Round-trip step count**: run Exp1 twice, confirm both runs end at
   identical step counts (already guaranteed by `IrisStretcher`, but
   worth verifying).
5. **Abort**: start a long experiment, press MENU mid-motion. Run should
   stop within ~1 second, motor stops where it is, menu returns.

---

## 9. Open questions for you

Need your answers on these before I start coding.

1. **Sampling cadence during motion.** I'm defaulting to "emit a row
   every 50 motor steps." Is that ok, or do you want it time-based
   (e.g., every 200 ms regardless of step rate)? Or both, configurable?

2. **Hold time semantics.** Each `ExpansionStep` has its own `holdMs`.
   Is that the right granularity, or do you want a single global "hold
   between waypoints" time? My proposal: per-step, with `holdMs = 0`
   meaning "don't hold, go straight to next waypoint."

3. **Step number column meaning.** Two candidates: motor's cumulative
   step counter (signed, persists across the experiment — this is what
   matches your "step count acts as a counter on the expansion size"
   note) vs. experiment-internal step index (1..N within this run). I
   propose **motor steps** because it's the absolute-position invariant
   you've been relying on. Confirm.

4. **CSV header re-emit during long runs.** Should it print once per
   experiment, or every K rows so a re-attached host can find a header?
   Default: once per run.

5. **`Xstrain` interaction with motion**. Can the user issue motion
   commands (`Xgoto`, etc.) while `Xstrain` is streaming? My proposal:
   yes — the motion commands keep working and rows continue to stream
   with `state` = `M`/`H` reflecting motor state. Or do you want strain
   streaming to lock out motion commands?

6. **Experiment selection UI**. Inside the LCD menu, experiments appear
   appended after the built-ins (XsetZero, Xzero, Xcalibrate, Xspeed,
   Xgoto, Xhelp, Xabout, then Exp1, Exp2, ...). Or do you want a
   separate "Experiments" submenu that you enter and then scroll
   through? Submenu is more work but cleaner if you have many.

7. **Default experiments shipped with the library.** Should `Exp1`
   (1x→3.4x→1x) live in the example sketch only, or be a built-in that
   ships with the library so a fresh install works out of the box?

8. **Strain array layout default**. The library will hardcode the
   verified 9-slot Kisley layout (5 on 0x71, 4 on 0x73) as the default
   for `IrisStrainArray`. Labs with different wiring call `setLayout()`
   before `begin()`. Sound right?

9. **Naming**. I've called the class `IrisStrainArray` to distinguish
   from the existing single-chip `IrisStrainNAU7802`. Both will live
   in the library. OK with both names, or prefer renaming
   `IrisStrainNAU7802` to something like `IrisStrainSingle` for
   symmetry?

10. **Custom `runFn` users.** The `IrisExperiment::customRun` callback
    lets labs write arbitrary motion profiles in C++. Worth keeping, or
    is the step-array-only model enough for what you actually need? I'd
    keep it because it costs nothing and unlocks ramps, but I won't put
    documentation effort into it unless you'll use it.

Answer those (terse answers are fine — "1: time-based 200 ms; 2: per-
step; 3: motor; 4: once; 5: yes; 6: appended; 7: example only; 8: yes;
9: keep both names; 10: keep") and I'll implement.

---

## 10. Estimated effort

After §9 answers:

- `IrisStrainArray` port: ~1 hour (the logic is already proven in
  StrainArray9.ino, just wrapping it).
- `IrisExperiment` + `IrisExperimentRunner`: ~1 hour.
- `IrisMenuUI` integration: ~45 min.
- `IrisSerialConsole` integration: ~30 min.
- Example sketch + README: ~30 min.
- Hardware verification: a session, can't shortcut.

Total ~3.5 hours of implementation + your verification time.
