# Serial Outputs — full catalogue

Every kind of line the `KisleyIrisStretcher` firmware can print to the
serial stream, grouped by category, with the exact text, the class that
emits it, and when it fires.

- **Bus**: 115200 baud, 8-N-1, line ending `\n`, ASCII/UTF-8 (a few lines
  contain `θ`, `µ`, `–`).
- **Destination**: every class writes to the same `Stream` (`Serial` by
  default; injectable). The serial console *reads* commands from that same
  stream.
- For a machine **parsing spec** of the CSV streams specifically, see
  [`SERIAL_CSV_FORMAT.md`](SERIAL_CSV_FORMAT.md). This document is the
  human-facing "what can appear" reference.

## Line-shape cheat sheet

| Shape | Meaning | Example |
|---|---|---|
| `=== Name ===` then `  key = value` rows then blank line | **Structured report** | `=== Xgoto ===` |
| Starts with `#` | **Informational comment** (status, logs, meta) | `# Tare complete.` |
| CSV header + comma rows (no prefix) | **Data stream** | `Xstrain,12345,0,0.000,S,…` |
| Plain line, no prefix | Banner / help / errors / confirmations / echoes | `Error: speed must be > 0` |

A parser can route every line by testing, in order: `===` prefix →
report; `#` prefix → comment; matches a CSV header seen earlier → data
row; else → free-form text.

---

## 1. Boot block (one-time, at power-on)

Emitted once during `setup()` as the sketch brings subsystems up.

### 1a. LCD bring-up + I²C scan — `IrisMenuUI::begin` / `scanI2CForLCD`
```
Initializing LCD with library auto-detection...
LCD initialized successfully!
Scanning I2C bus to identify LCD address...
Scanning I2C bus for LCD...
I2C device found at address 0x27
  -> LCD detected at 0x27
Found 1 I2C device(s)
Using LCD address: 0x27
Splash screen displayed
```
Variants: `LCD initialization failed! Status: <n>` + `Check I2C
connections and LCD address`; `Unknown error at address 0x<NN>`; `No I2C
devices found!`; `WARNING: No I2C LCD detected in scan`.

### 1b. Console banner — `IrisSerialConsole::printBanner`
```
 ___      _       ____  _            _       _
|_ _|_ __(_)___  / ___|| |_ _ __ ___| |_ ___| |__   ___ _ __
 | || '__| / __| \___ \| __| '__/ _ \ __/ __| '_ \ / _ \ '__|
 | || |  | \__ \  ___) | |_| | |  __/ || (__| | | |  __/ |
|___|_|  |_|___/ |____/ \__|_|  \___|\__\___|_| |_|\___|_|
+===============+
|KisleyLab V1.0 |
+===============+
```
The middle line is `setBannerLine(...)` (default `|KisleyLab V1.0 |`).

### 1c. Command help — `IrisSerialConsole::printHelp` (also on `Xhelp`)
```
Commands:
________
 Xgoto <Ex> [cw|ccw]  – move to Ex magnitude (CW default)
 Xzero                – drive motor to zero Position
 XsetZero             – reset position counter to 0
 Xcalibrate           – run calibration routine
 Xspeed <value> cm/s  – changes expansion speed (cm/s approx)
 Xstrain              – toggle 9-ADC strain CSV streaming
 XsignalAverage <n>   – samples averaged per ADC per row [1..256]
 Xsamplerate <sps>    – NAU7802 rate (10/20/40/80/320 SPS)
 XlogEveryN <n>       – log one motion row per N steps [1..99]
 Xrun <name>          – run a registered experiment (Xrun list)
 Xabort               – abort a running experiment
 Xhelp                – this message
 X<name>  – <help line>          (one per registered custom command)
```

### 1d. Strain array bring-up — `IrisStrainArray::begin`
```
# IrisStrainArray begin
# I2C clock: 10000 Hz
# Mux 0x71: ACK
# Mux 0x73: ACK
# Init ADC1 (mux 0x71 ch0): OK (first read = -12)
# Init ADC6 (mux 0x73 ch0): NACK (no chip)
# Initialised 9 / 9
# Taring 16 samples per chip...
# IMPORTANT: keep the rig undisturbed for the next few seconds.
#   ADC1 baseline = -12  (n=16)
# Tare complete.
```
Per-chip init variants: `NACK (no chip)`, `nau.begin FAILED`, `OK begin
but no sample after cal`.

---

## 2. Structured command reports

Aligned `key = value` blocks from the shared formatter
(`src/internal/ReportFormat.h`). Intrinsic to the command, so they print
identically whether triggered from the **serial console or the LCD GUI**.
A trailing blank line separates consecutive reports.

### 2a. `Xgoto` / `Xzero` — `IrisStretcher::gotoExpansion` / `goToZero`
```
=== Xgoto ===
  uptime_ms            = 1234567
  current_position     = 0
  target_expansion     = 1.350
  resulting_expansion  = 1.350
  direction            = cw
  steps_required       = 1234
  current_step_count   = 0
  final_step_count     = 1234
  blade_speed_cm_s     = 0.1000
```
`uptime_ms` is `millis()` at the start of the move — same clock as the CSV
`t_ms`. `steps_required` is signed (`final − start`).

`direction` is the commanded rotation: `cw` or `ccw`, echoing the token
from `Xgoto <Ex> [cw|ccw]` (the sign of the signed expansion). It appears
**only for `Xgoto` expansion moves** — `Xzero` and the `Xgoto`-to-center
case omit the row (returning to centre has no expansion direction). So
`Xzero` is otherwise identical with header `=== Xzero ===` and no
`direction` line.

### 2b. `XsetZero` — `IrisStretcher::setZeroHere`
```
=== XsetZero ===
  uptime_ms            = 1234567
  previous_position    = 4123
  new_position         = 0
  blade_speed_cm_s     = 0.1000
```
Resets the counter without moving, so it reports only `previous_position`
→ `new_position`.

### 2c. `Xspeed` — `IrisSerialConsole::printSpeedReport` (serial only)
```
=== Xspeed ===
  uptime_ms            = 1234567
  blade_speed_cm_s     = 0.1000
  step_half_period_us  = 21027
  current_position     = 0
```

> **Suppression**: all three intrinsic reports (2a/2b) are gated by
> `IrisStretcher::setMoveReporting(bool)`. The experiment runner and the
> calibration routine turn them **off** so internal moves don't pollute
> the CSV stream / clutter the routine.

---

## 3. Short command confirmations (`#`-prefixed) — `IrisSerialConsole`

| Command | Output |
|---|---|
| `Xstrain` | `# Strain streaming: ON` / `# Strain streaming: OFF` |
| `XsignalAverage <n>` | `# Signal averaging set to <n>` |
| `Xsamplerate <sps>` | `# Sample rate set to <sps> SPS` |
| `XlogEveryN <n>` | `# Motion log every <n> steps` |
| `Xabort` | `# Abort requested` |
| `Xrun list` | `# Registered experiments (2):` then `#   Exp1` … |

---

## 4. Errors & usage messages (no prefix)

### 4a. Parser / console — `IrisSerialConsole`
```
Invalid prefix. Commands must start with 'X'.
Unknown command: <cmd>
Unknown direction: <dir>
Usage: Xgoto <Ex> [cw|ccw]
Usage: Xspeed <bladeSpeed_cm/s>
Usage: Xrun <name> | Xrun list
Usage: XsignalAverage <1..256>
Usage: Xsamplerate <10|20|40|80|320>
Usage: XlogEveryN <1..99>
Error: speed must be > 0
Error: unsupported rate <x> (use 10/20/40/80/320)
Error: no experiment named <name>
Error: no IrisExperimentRunner attached
Error: no IrisStrainArray attached
```

### 4b. Motion — `IrisStretcher::gotoExpansion`
```
Error: |Ex| out of range [1.0, 4.650)
Error: no valid θ found for that targetEx
```
On error, no report block is printed.

---

## 5. CSV data streams

Two distinct CSV shapes. See `SERIAL_CSV_FORMAT.md` for the full parsing
contract. State column: `M` moving, `H` holding, `S` static stream.

### 5a. Wide stream — `Xstrain` toggle & `Xrun` — `IrisExperimentRunner`
Header (`_emitHeader`) + rows; `exp` is the run name or `Xstrain`.
```
exp,t_ms,steps,target_ex,state,ADC1_mean,ADC1_std,…,ADC9_mean,ADC9_std
Xstrain,12345,0,0.000,S,12,1.21,…
Exp1,12545,4280,3.400,M,1240,1.19,…
```

### 5b. Bare strain rows — `IrisStrainArray::printCsvHeader/Row`
Used when a sketch drives the array directly (e.g. `extras/StrainArray9`).
```
t_ms,ADC1_mean,ADC1_std,…,ADC9_mean,ADC9_std
12345,-12,1.21,-8,0.95,…
```
Missing chips emit `NaN,NaN`.

---

## 6. Experiment lifecycle (`#`-prefixed) — `IrisExperimentRunner`

Wraps the 5a stream during `Xrun`:
```

# experiment: Exp1
# steps: 3
# motion log every 200 ms
# hold 2000 ms, log every 100 ms
exp,t_ms,steps,target_ex,state,…          <- 5a header + rows here
# experiment complete
```
Ends with `# experiment complete` or `# experiment aborted`.

---

## 7. Calibration narrative (no prefix) — `IrisStretcher::defaultCalibration`
```
Starting calibration...
Moved 1234 steps, new position = 1234
Moving back in by -0.290 rad...
Moved -512 steps, new position = 722
Setting Zero
Calibration complete.
```
The per-step `Moved …` lines come from the low-level escape hatch
`rotateThetaRadians`; the `XsetZero` reports are suppressed here.

---

## 8. Low-level escape hatch (no prefix) — `IrisStretcher::rotateThetaRadians`
```
Moved 1234 steps, new position = 1234
```
Only fires when the public `rotateThetaRadians(theta)` is called directly
(e.g. from calibration). The high-level `gotoExpansion`/`goToZero` paths
move quietly and emit the §2 report instead.

---

## 9. LCD command echoes (no prefix) — `IrisMenuUI`

When `setEchoCommands(true)` (default), every action committed from the
LCD/encoder echoes the equivalent serial command **before** its report,
so a serial log of a hands-on session is a replayable command script.

```
XsetZero
Xzero
Xcalibrate
Xabort
Xgoto 1.350 cw           (or "... ccw")
Xspeed 0.10
XsignalAverage 8
Xsamplerate 320
XlogEveryN 2
Xrun Exp1
```
Disable with `ui.setEchoCommands(false)`.

---

## 10. On-demand status (`#`-prefixed) — `IrisStrainArray::printStatus`
```
# IrisStrainArray status:
#   Muxes: 0x71,0x73
#   Signal averaging: 2
#   I2C clock: 10000 Hz
#   ADCs present: 9 / 9
```
Only printed if a sketch/command calls `printStatus()`.

---

## Quick reference — what triggers what

| Trigger | Sections |
|---|---|
| Power-on / `setup()` | 1a–1d |
| `Xgoto` / `Xzero` (serial or LCD) | 2a (+ 9 if from LCD) |
| `XsetZero` | 2b (+ 9 if from LCD) |
| `Xspeed` | 2c serial / LCD shows its own confirmation |
| `Xstrain` | 3 + 5a |
| `Xrun <name>` | 6 wrapping 5a |
| `Xcalibrate` | 7 |
| `XsignalAverage`/`Xsamplerate`/`XlogEveryN` | 3 |
| `Xabort` / `Xhelp` / `Xrun list` | 3 / 1c / 3 |
| Bad input | 4 |
