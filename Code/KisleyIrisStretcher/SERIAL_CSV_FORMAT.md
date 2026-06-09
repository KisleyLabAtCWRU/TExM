# Serial CSV Format

Reference document for a Python (or any other) script that parses the
USB-serial output of the `KisleyIrisStretcher` firmware into CSV-style
rows. Covers the canonical example `IrisModule1.0` and the
`Xstrain` / `Xrun` flows.

This is a **parsing spec**, not a usage guide — see `README.md` for the
firmware itself.

---

## 1. The bus

- **Baud rate**: 115200, 8-N-1, no flow control.
- **Encoding**: ASCII / UTF-8. A few lines contain Greek (`θ`) or
  symbol (`µ`) characters; parsers can safely treat them as opaque
  bytes since they only appear in `#`-prefixed comments.
- **Line endings**: `\n` only. The firmware emits `Serial.println()`
  which on Arduino-ESP32 is `\n`.

The serial output is a continuous mixed stream of:

- **Boot text** — banner, init log, command help (one block at power-on).
- **Comment lines** — informational status emitted by the firmware
  during operation; always start with `#`.
- **CSV streams** — header line + N data rows. Triggered by `Xstrain`
  (continuous mode) or `Xrun <name>` (one stream per experiment).
- **Command response lines** — short replies to individual `X…`
  commands. Some have no consistent prefix.

A parser only needs to extract the CSV streams; everything else can be
discarded (or logged separately).

---

## 2. Line classification

Apply rules in order, first match wins:

| If the line… | Then it's a… | What to do |
|---|---|---|
| starts with `#` | comment / status line | discard (or log) |
| equals `exp,t_ms,steps,target_ex,state,...` (literal `exp,` prefix and the rest matches the column-name pattern in §3) | **CSV header** — new stream starting | switch parser to "in-stream", record column names |
| is in "in-stream" mode AND comma-count matches the active header | **CSV row** | parse and emit |
| is in "in-stream" mode but comma-count doesn't match | end of stream (probably a `#` comment or empty line follows) | leave "in-stream" mode, treat this line by re-applying rules from the top |
| is empty | blank | discard |
| anything else | irregular text (command response, banner remnants) | discard or log |

You don't need to detect the END of a CSV stream explicitly. The
firmware emits:

- `# experiment complete` at the natural end of an `Xrun` stream
- `# Strain streaming: OFF` after a second `Xstrain` toggle
- `# experiment aborted` after an `Xabort`

But these are just `#` comments. Detecting them is optional —
"in-stream" mode naturally exits the next time a `#` line arrives.

### Multiple streams in one session

Each `Xrun` and each `Xstrain` ON-cycle emits its own header line.
Treat each header as the start of a fresh stream — the column count
*can* differ between streams if `setLayout()` is called between runs,
though in practice it never will on the same firmware.

---

## 3. CSV column schema

The header is exactly:

```
exp,t_ms,steps,target_ex,state,ADC1_mean,ADC1_std,ADC2_mean,ADC2_std,...,ADC9_mean,ADC9_std
```

That's **5 metadata columns + 2 columns per ADC**. With the default
9-ADC Kisley layout, total = **23 columns**.

| # | Column | Type | Range | Meaning |
|---|---|---|---|---|
| 1 | `exp` | string | e.g. `Exp1`, `Exp2`, `Xstrain` | The experiment that's running, or the literal string `Xstrain` for continuous-streaming mode. Acts as a "stream id" — all rows in one stream share the same value. |
| 2 | `t_ms` | uint32 | 0 .. ~49 days | `millis()` at the moment `acquireRow()` started for this row. Wraps every ~49 days (firmware doesn't reboot on overflow, parser should handle wrap if you run that long). Monotonic within a session. |
| 3 | `steps` | int32 | ±2³¹ | Motor's signed absolute step count. Positive when motor has rotated CW past zero, negative for CCW. Persists across waypoints — every row carries the *current* cumulative step count. |
| 4 | `target_ex` | float | 1.000 .. `maxEx` | Magnitude of the *commanded* Ex for the current waypoint, written with 3 decimal places. For `Xstrain` rows (no active experiment) this is `0.000`. The sign of the rotation is implicit in `steps` and `state`. |
| 5 | `state` | char | `M` / `H` / `S` | Motor state at the moment this row was sampled — see §4. |
| 6 | `ADCk_mean` | int32 or `NaN` | ±2³¹ | Round-half-away-from-zero mean of `signalAveraging` raw samples for ADC channel k, minus the boot-time baseline. Zeroed counts. `NaN` if the chip was missing or all samples timed out. |
| 7 | `ADCk_std` | float (2 dp) or `NaN` | ≥ 0 | Bessel-corrected sample standard deviation across those same samples. `0.00` if N≤1. `NaN` if the chip was missing. Use as the per-row error bar on `ADCk_mean`. |
| … | (pairs repeat for k = 2..9) | | | |

### Notes on individual columns

- **`t_ms`** is the row's **sample-start** timestamp, captured inside
  `IrisStrainArray::acquireRow()` before any chips are read. Per-chip
  read timing within a row spans `t_ms` to roughly `t_ms +
  signalAveraging × 9 × ~5 ms`.
- **`steps`** is updated continuously during motion — the value
  printed on each row reflects the motor's position at the moment the
  row was emitted, not at the start of the row's sampling.
- **`target_ex`** is always positive even for CCW motion. The CCW
  intent is recoverable from the trajectory of `steps` (it decreases)
  but is *not* directly visible in `target_ex` alone. If you need the
  signed target, watch `steps` deltas around each waypoint.

---

## 4. The `state` column

Three possible values:

| Value | Meaning |
|---|---|
| `M` | **Moving** — motor is actively stepping toward `target_ex`. Rows arrive every `motionLogPeriodMs` (default 200 ms). |
| `H` | **Holding** — motor has reached the waypoint and is dwelling for the experiment's `holdMs` (default 2000 ms). Rows arrive every `holdLogPeriodMs` (default 100 ms). |
| `S` | **Static streaming** — continuous `Xstrain` mode. No experiment running; motor may or may not be moving (motion commands like `Xgoto` still work during `Xstrain` but the `state` column will be `S`, not `M`/`H`). |

A typical `Xrun Exp1` stream where Exp1 = `{1.0, 3.4, 1.0}`:

```
M  M  M  M  ...  M    ←  motor moving from current pos to step 1 (target 1.0)
H  H  H  H  H  H  ...  ← holding at step 1 for 2 s
M  M  M  M  ...  M    ←  moving to step 2 (target 3.4)
H  H  H  H  H  H  ...
M  M  M  M  ...  M    ←  moving to step 3 (target 1.0)
H  H  H  H  H  H  ...
```

Detect step transitions by `target_ex` changes or by an `M→H` edge.

---

## 5. NaN handling

The firmware emits the literal string `NaN` (no quotes, exact spelling
`N`, `a`, `N`) for any per-ADC mean or std it can't compute:

- ADC was never initialised at boot (chip missing or wiring fault).
- All `signalAveraging` reads in this row timed out (chip became
  unreachable mid-session).

Both `_mean` and `_std` for a missing ADC come through as `NaN,NaN` —
column count is always preserved, so you can use `csv.reader` /
`numpy.loadtxt` without special handling. Use
`pandas.read_csv(..., na_values=["NaN"])` and these become `pd.NA`
automatically.

`target_ex` and `steps` are **never** NaN. If parsing fails on those,
the line is malformed (almost certainly a partial line from serial
buffering).

---

## 6. Stream lifecycle

### Power-on / reset

```
# Iris Stretcher / banner ASCII art (multi-line, all without #-prefix on the art itself)
# +===============+
# |KisleyLab V2.1 |
# +===============+
# Commands: ...
# IrisStrainArray begin
# I2C clock: 50000 Hz
# Mux 0x71: ACK
# Mux 0x73: ACK
# Init ADC1 (mux 0x71 ch0): OK (first read = ...)
... (one per ADC)
# Initialised 9 / 9
# Taring 16 samples per chip...
# IMPORTANT: keep the rig undisturbed for the next few seconds.
#   ADC1 baseline = ...
... (one per ADC)
# Tare complete.
```

Everything in this block can be discarded by a parser. The
"IrisStrainArray begin" line is a reliable boot marker if you need
to detect firmware restart.

> **Edge case**: the boot banner *includes lines that don't start with
> `#`* — specifically the ASCII-art lines. They're irregular text;
> filter them out by the "doesn't match header pattern, not in stream"
> rule from §2. Or skip everything until you see the first `# Tare
> complete.` line, which always precedes the first useful CSV stream.

### `Xrun Exp1`

Every `Xrun` re-runs the host-side tare before the first CSV row,
so each stream's per-ADC means are zeroed against the rig's
pre-run baseline. The tare block (same shape as boot) appears first:

```
# Taring 16 samples per chip...
# IMPORTANT: keep the rig undisturbed for the next few seconds.
#   ADC1 baseline = -7634  (n=16)
...
# Tare complete.

# experiment: Exp1
# steps: 3
# motion log every 200 ms
# hold 2000 ms, log every 100 ms
exp,t_ms,steps,target_ex,state,ADC1_mean,ADC1_std,...,ADC9_mean,ADC9_std
Exp1,12345,4123,1.000,M,...
Exp1,12545,4280,1.000,M,...
...
Exp1,14600,4525,1.000,H,...
... etc
# experiment complete
```

A parser should treat the tare baselines emitted here as
**replacing** the boot-time baselines for analysing this stream —
they're the more relevant zero.

### `Xstrain` (toggle ON)

Toggling `Xstrain` ON also re-tares before emitting the CSV header,
for the same reason. (Toggling OFF does not re-tare.)

```
# Taring 16 samples per chip...
#   ADC1 baseline = -7634  (n=16)
...
# Tare complete.
# Strain streaming: ON
exp,t_ms,steps,target_ex,state,ADC1_mean,ADC1_std,...,ADC9_mean,ADC9_std
Xstrain,12345,0,0.000,S,...
Xstrain,12545,0,0.000,S,...
...
```

### `Xstrain` (toggle OFF) or `Xabort`

```
# Strain streaming: OFF
```

Or:

```
# experiment aborted
```

Stream stops; no closing CSV row. The last CSV row received is the
last valid row.

---

## 7. Timestamps and sampling cadence

| Mode | Row period | Source |
|---|---|---|
| Motion (state = `M`) | `motionLogPeriodMs` (default 200 ms) | `IrisExperimentRunner::setMotionLogPeriodMs` |
| Holding (state = `H`) | `holdLogPeriodMs` (default 100 ms) | `IrisExperimentRunner::setHoldLogPeriodMs` |
| `Xstrain` (state = `S`) | `motionLogPeriodMs` | same setter (reuses motion cadence) |

These are *targets*, not guarantees — each row's acquisition takes
roughly `signalAveraging × 9 × 5 ms` (~45 ms at the canonical
`signalAveraging=8` and 50 kHz I²C). If acquisition takes longer than
the configured period, rows simply arrive as fast as they can.

Compute true row rate from `t_ms` deltas, not the configured period.

---

## 8. Recommended Python parser outline

This is what I'd build for the serial-reader script; treat it as
pseudocode.

```python
import csv, math, re, serial

HEADER_PREFIX = "exp,t_ms,steps,target_ex,state,"

def parse_stream(port="/dev/cu.usbmodem...", baud=115200):
    ser = serial.Serial(port, baud, timeout=0.5)
    in_stream = False
    header = None
    n_cols = 0

    while True:
        line = ser.readline().decode("utf-8", errors="replace").strip()
        if not line:
            continue
        if line.startswith("#"):
            in_stream = False
            yield ("comment", line)
            continue
        if line.startswith(HEADER_PREFIX):
            header = line.split(",")
            n_cols = len(header)
            in_stream = True
            yield ("header", header)
            continue
        if in_stream:
            fields = line.split(",")
            if len(fields) != n_cols:
                in_stream = False
                yield ("text", line)
                continue
            row = {
                "exp":        fields[0],
                "t_ms":       int(fields[1]),
                "steps":      int(fields[2]),
                "target_ex":  float(fields[3]),
                "state":      fields[4],
            }
            for i, name in enumerate(header[5:], start=5):
                v = fields[i]
                if v == "NaN":
                    row[name] = math.nan
                elif name.endswith("_mean"):
                    row[name] = int(v)
                else:  # _std
                    row[name] = float(v)
            yield ("row", row)
        else:
            yield ("text", line)
```

The generator yields tagged tuples so the caller can route comments
to a log file, headers to a "new stream" handler, and rows to a
DataFrame / sink. Tagging is preferable to silently dropping
non-CSV lines — boot/setup info is useful diagnostically.

### One-shot: load a saved log into a DataFrame

If you've already captured serial to a file:

```python
import pandas as pd
df = pd.read_csv("session.log",
                 comment="#",
                 na_values=["NaN"],
                 on_bad_lines="skip")  # drop banner/response lines
```

That's adequate for offline analysis as long as you accept silently
dropping any malformed/non-CSV lines. The streaming parser above is
better for live capture because it tells you about the dropped lines.

---

## 9. Examples

### Full Exp1 run (abbreviated to 1 ADC for space)

```
# experiment: Exp1
# steps: 3
# motion log every 200 ms
# hold 2000 ms, log every 100 ms
exp,t_ms,steps,target_ex,state,ADC1_mean,ADC1_std
Exp1,1003,0,1.000,M,-2,0.85
Exp1,1213,142,1.000,M,5,1.20
Exp1,1421,287,1.000,H,8,0.92
Exp1,1521,287,1.000,H,7,0.88
Exp1,1621,287,1.000,H,9,1.05
... (~20 H rows over 2 s) ...
Exp1,3420,287,3.400,M,15,1.40
Exp1,3624,623,3.400,M,134,2.10
Exp1,3829,891,3.400,M,267,2.45
Exp1,4035,1024,3.400,H,290,1.95
... (~20 H rows) ...
Exp1,6101,1024,1.000,M,287,1.80
...
Exp1,8050,0,1.000,H,2,0.78
... (~20 final H rows) ...
# experiment complete
```

### Continuous Xstrain with a missing chip

```
# Strain streaming: ON
exp,t_ms,steps,target_ex,state,ADC1_mean,ADC1_std,ADC2_mean,ADC2_std,ADC3_mean,ADC3_std
Xstrain,5234,0,0.000,S,1,0.82,NaN,NaN,4,1.05
Xstrain,5434,0,0.000,S,-3,0.91,NaN,NaN,3,0.98
Xstrain,5634,0,0.000,S,2,0.87,NaN,NaN,5,1.10
```

ADC2 was missing at boot — every row carries `NaN,NaN` for it.

### Boot + first Xrun

```
 ___      _       ____  _            _       _
|_ _|_ __(_)___  / ___|| |_ _ __ ___| |_ ___| |__   ___ _ __
 | || '__| / __| \___ \| __| '__/ _ \ __/ __| '_ \ / _ \ '__|
 | || |  | \__ \  ___) | |_| | |  __/ || (__| | | |  __/ |
|___|_|  |_|___/ |____/ \__|_|  \___|\__\___|_| |_|\___|_|
+===============+
|KisleyLab V2.1 |
+===============+
Commands:
________
 Xgoto <Ex> [cw|ccw]  – move to Ex magnitude (CW default)
 ...
# IrisStrainArray begin
# I2C clock: 50000 Hz
# Mux 0x71: ACK
# Mux 0x73: ACK
# Init ADC1 (mux 0x71 ch0): OK (first read = -7634)
... (ADC2..9) ...
# Initialised 9 / 9
# Taring 16 samples per chip...
# IMPORTANT: keep the rig undisturbed for the next few seconds.
#   ADC1 baseline = -7634  (n=16)
... (one per ADC) ...
# Tare complete.

# experiment: Exp1
# steps: 3
# motion log every 200 ms
# hold 2000 ms, log every 100 ms
exp,t_ms,steps,target_ex,state,ADC1_mean,ADC1_std,...
Exp1,1003,0,1.000,M,-2,0.85,...
```

---

## 10. Edge cases and parser pitfalls

1. **Partial lines on connection**. When you open the serial port,
   you may join mid-line. Discard everything until the first `\n`.
2. **`#` lines mid-stream**. The firmware doesn't emit `#` lines
   between CSV rows of the same stream — but if you implement
   user-registered serial commands that print to Serial, those *will*
   interleave. Treat any non-CSV-row line as ending the stream
   (per §2).
3. **`NaN` capitalisation**. The firmware always emits literal `NaN`
   (capital N, lowercase a, capital N). Don't case-fold.
4. **`target_ex` precision**. Always 3 decimal places. Always
   positive — see §3's note on CCW recovery.
5. **`steps` direction**. The sign of `steps` is the *cumulative*
   motor rotation since last `XsetZero`, not since the start of the
   experiment. If you need experiment-relative steps, subtract the
   first row's `steps` value from later rows.
6. **`Xstrain` during motion**. If `Xstrain` is on and an `Xgoto` or
   `Xrun` happens concurrently, the streaming `state` stays `S` even
   while the motor moves. The experiment's own stream uses `M`/`H`.
   You'll see two interleaved streams sharing the bus — they're
   distinguishable by the `exp` column.
7. **Buffer overruns**. The ESP32 USB serial buffer is 256 bytes by
   default. At ~250 char rows × 5 Hz = ~1250 B/s, well within. At
   high-cadence experiments (10-20 Hz) you'd want to read promptly
   on the host or you'll drop rows.
8. **No checksums or framing**. The firmware does not include row
   sequence numbers, CRCs, or frame markers. If integrity matters,
   add them in a `customRun` callback or post-hoc by checking `t_ms`
   monotonicity.

---

## 11. Format version

This document describes the format emitted by `KisleyIrisStretcher`
**v2.0.x**. Future firmware versions may add columns to the right of
`ADC9_std` (e.g., temperature) — parsers should iterate `header[5:]`
in pairs rather than hardcoding 18 ADC columns.

If columns are ever added in the middle, the firmware will bump the
major version and the new schema will be documented here.
