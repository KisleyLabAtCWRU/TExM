// StrainArray9 — clean-room rebuild.
//
// Reads 9 NAU7802 strain ADCs across two TCA9548A I2C muxes and streams
// a CSV row per acquisition to USB serial. No external library wrapper;
// uses Adafruit_NAU7802 directly. Every choice in here was validated by
// the diagnostic sketches (I2CScanner, ThoroughI2C, TwoChipTest):
//
//   - Mux A at 0x71, channels 0..4 hold ADC1..ADC5
//   - Mux B at 0x73, channels 0..3 hold ADC6..ADC9
//   - I2C clock pinned at 10 kHz (bus is signal-marginal at 100 kHz)
//   - ALWAYS deselect both muxes before selecting a channel — otherwise
//     two NAU7802s end up on the bus simultaneously and collide
//   - Re-set the I2C clock after every nau.begin(), because the Adafruit
//     library calls Wire.begin() internally which resets it to 100 kHz

#include "USB.h"
#include <Wire.h>
#include <Adafruit_NAU7802.h>
#include <math.h>

// ====================== Configuration ======================
static constexpr int       PIN_SDA      = 3;
static constexpr int       PIN_SCL      = 4;
static constexpr uint32_t  SERIAL_BAUD  = 115200;
static constexpr uint32_t  I2C_CLOCK_HZ = 10000;
static constexpr uint8_t   MUX_A        = 0x71;
static constexpr uint8_t   MUX_B        = 0x73;
static constexpr uint8_t   NAU_ADDR     = 0x2A;
static constexpr uint16_t  SAMPLE_TIMEOUT_MS = 200;
static constexpr uint16_t  ROW_PERIOD_MS = 100;     // ~10 Hz row rate

struct Slot { uint8_t mux; uint8_t ch; const char *label; };

static const Slot LAYOUT[] = {
  {MUX_A, 0, "ADC1"},
  {MUX_A, 1, "ADC2"},
  {MUX_A, 2, "ADC3"},
  {MUX_A, 3, "ADC4"},
  {MUX_A, 4, "ADC5"},
  {MUX_B, 0, "ADC6"},
  {MUX_B, 1, "ADC7"},
  {MUX_B, 2, "ADC8"},
  {MUX_B, 3, "ADC9"},
};
static constexpr uint8_t N_ADCS = sizeof(LAYOUT) / sizeof(LAYOUT[0]);

Adafruit_NAU7802 nau;
bool    g_present[N_ADCS]  = {false};
int32_t g_baseline[N_ADCS] = {0};      // captured at boot; subtracted from every stream value

static constexpr uint8_t  TARE_SAMPLES     = 16;   // samples averaged per ADC at boot
static constexpr uint16_t SIGNAL_AVERAGING = 4;    // samples per ADC per output row (1..256)

// One CSV row's worth of acquired data.
struct Row {
  uint32_t t_ms;
  int32_t  mean[N_ADCS];
  float    std [N_ADCS];
  bool     present[N_ADCS];
};

// ====================== I2C primitives ======================

static inline void restoreClock() { Wire.setClock(I2C_CLOCK_HZ); }

static bool i2cAck(uint8_t addr) {
  Wire.beginTransmission(addr);
  return Wire.endTransmission() == 0;
}

static void muxDeselectAll() {
  Wire.beginTransmission(MUX_A);
  Wire.write((uint8_t)0x00);
  Wire.endTransmission();
  Wire.beginTransmission(MUX_B);
  Wire.write((uint8_t)0x00);
  Wire.endTransmission();
}

static void muxRoute(uint8_t mux, uint8_t ch) {
  // Belt and suspenders: always start from a fully-deselected state.
  muxDeselectAll();
  delay(2);     // let the muxes settle into deselected state
  Wire.beginTransmission(mux);
  Wire.write((uint8_t)(1u << ch));
  Wire.endTransmission();
  delay(5);     // let the new channel come up
}

static bool waitSample(uint16_t timeoutMs) {
  uint32_t t0 = millis();
  while (!nau.available()) {
    if (millis() - t0 > timeoutMs) return false;
  }
  return true;
}

// ====================== Per-chip init ======================

static bool initOneChip(uint8_t i) {
  const Slot &s = LAYOUT[i];
  Serial.print(F("# Init "));
  Serial.print(s.label); Serial.print(F(" (mux 0x"));
  Serial.print(s.mux, HEX); Serial.print(F(" ch"));
  Serial.print(s.ch); Serial.print(F("): "));

  restoreClock();
  muxRoute(s.mux, s.ch);

  if (!i2cAck(NAU_ADDR)) {
    Serial.println(F("NACK (no chip on this channel)"));
    return false;
  }

  if (!nau.begin(&Wire)) {
    restoreClock();
    Serial.println(F("nau.begin FAILED"));
    return false;
  }
  restoreClock();   // Adafruit lib just called Wire.begin internally

  nau.setLDO(NAU7802_3V0);
  nau.setGain(NAU7802_GAIN_128);
  nau.setRate(NAU7802_RATE_10SPS);

  // Internal calibration — zeroes the chip's intrinsic input offset.
  // Adafruit's calibrate() is buggy (its wait-loop checks the bit in the
  // wrong direction and returns before cal completes). Add our own delay
  // to give the chip time to actually finish — 200 ms is plenty at 10 SPS.
  nau.calibrate(NAU7802_CALMOD_INTERNAL);
  delay(200);

  // Drain any stale sample sitting in the data register from before cal.
  if (waitSample(800)) (void)nau.read();

  // Skip OFFSET calibration entirely for now — we just want data flowing.
  // Subtract baselines on the host side. If we need on-device tare later,
  // we'll revisit with a manual register-poke that waits properly.

  if (!waitSample(800)) {
    Serial.println(F("OK begin but no sample after internal cal"));
    return false;
  }
  int32_t first = nau.read();
  Serial.print(F("OK (first read = "));
  Serial.print(first); Serial.println(F(")"));
  return true;
}

// ====================== Per-chip read ======================

// Reads one raw sample (no baseline subtraction). Used during tare.
static bool readOneChipRaw(uint8_t i, int32_t &out) {
  if (!g_present[i]) return false;
  const Slot &s = LAYOUT[i];
  restoreClock();
  muxRoute(s.mux, s.ch);
  if (!waitSample(SAMPLE_TIMEOUT_MS)) return false;
  out = nau.read();
  return true;
}

// Reads one zeroed sample (raw − stored baseline). Used during streaming.
static bool readOneChip(uint8_t i, int32_t &out) {
  int32_t raw;
  if (!readOneChipRaw(i, raw)) return false;
  out = raw - g_baseline[i];
  return true;
}

// ====================== Signal-averaged acquisition ======================
//
// Round-robin: each pass visits every present ADC once, total
// SIGNAL_AVERAGING passes. Spacing samples across channels rather than
// reading one chip N times then the next keeps the N-sample window roughly
// co-temporal between channels. Tracks Σx and Σx² per channel in int64 so
// we can compute Bessel-corrected sample stdev without intermediate overflow.
static void acquireRow(Row &row) {
  row.t_ms = millis();

  int64_t  sum  [N_ADCS] = {0};
  int64_t  sumsq[N_ADCS] = {0};
  uint16_t count[N_ADCS] = {0};

  for (uint16_t pass = 0; pass < SIGNAL_AVERAGING; pass++) {
    for (uint8_t i = 0; i < N_ADCS; i++) {
      if (!g_present[i]) continue;
      int32_t raw;
      if (readOneChipRaw(i, raw)) {
        const int64_t zeroed = (int64_t)raw - (int64_t)g_baseline[i];
        sum[i]   += zeroed;
        sumsq[i] += zeroed * zeroed;
        count[i] += 1;
      }
    }
  }

  for (uint8_t i = 0; i < N_ADCS; i++) {
    row.present[i] = g_present[i] && (count[i] > 0);
    if (!row.present[i]) {
      row.mean[i] = 0;
      row.std[i]  = 0.0f;
      continue;
    }

    // Mean with round-half-away-from-zero.
    const int64_t  s = sum[i];
    const uint16_t n = count[i];
    int64_t q = s / (int64_t)n;
    int64_t r = s - q * (int64_t)n;
    if (r >  (int64_t)n / 2)        q += 1;
    else if (r < -(int64_t)n / 2)   q -= 1;
    row.mean[i] = (int32_t)q;

    // Bessel-corrected sample stdev:  var = (Σx² − (Σx)²/n) / (n−1)
    if (n <= 1) {
      row.std[i] = 0.0f;
    } else {
      const double dS  = (double)s;
      const double dSS = (double)sumsq[i];
      const double dN  = (double)n;
      double var = (dSS - (dS * dS) / dN) / (dN - 1.0);
      if (var < 0.0) var = 0.0;            // FP noise floor
      row.std[i] = (float)sqrt(var);
    }
  }
}

// ====================== Host-side tare ======================
//
// Captures TARE_SAMPLES per chip and stores the mean as the baseline.
// Subsequent stream reads have this subtracted. Replaces the broken
// on-device calibrate(OFFSET) call.
static void tareAllAdcs() {
  Serial.print(F("# Taring "));
  Serial.print(TARE_SAMPLES);
  Serial.println(F(" samples per chip..."));
  Serial.println(F("# IMPORTANT: keep the rig undisturbed for the next few seconds."));

  for (uint8_t i = 0; i < N_ADCS; i++) {
    if (!g_present[i]) { g_baseline[i] = 0; continue; }
    int64_t sum = 0;
    uint8_t got = 0;
    for (uint8_t k = 0; k < TARE_SAMPLES; k++) {
      int32_t s;
      if (readOneChipRaw(i, s)) { sum += s; got++; }
    }
    g_baseline[i] = (got > 0) ? (int32_t)(sum / got) : 0;
    Serial.print(F("#   "));
    Serial.print(LAYOUT[i].label);
    Serial.print(F(" baseline = "));
    Serial.print(g_baseline[i]);
    Serial.print(F("  (n="));
    Serial.print(got); Serial.println(F(")"));
  }
  Serial.println(F("# Tare complete."));
}

// ====================== Setup / loop ======================

void setup() {
  USB.begin();
  Serial.begin(SERIAL_BAUD);
  unsigned long t0 = millis();
  while (!Serial && millis() - t0 < 2000) {}
  delay(200);

  Wire.begin(PIN_SDA, PIN_SCL);
  restoreClock();

  Serial.println();
  Serial.println(F("# StrainArray9 v1.0 (clean-room rebuild)"));
  Serial.print  (F("# Build: ")); Serial.print(F(__DATE__));
  Serial.print  (' ');             Serial.println(F(__TIME__));
  Serial.print  (F("# I2C clock: ")); Serial.print(I2C_CLOCK_HZ); Serial.println(F(" Hz"));
  Serial.print  (F("# SDA=GPIO")); Serial.print(PIN_SDA);
  Serial.print  (F("  SCL=GPIO")); Serial.println(PIN_SCL);

  // Sanity: confirm both muxes ACK before touching anything else.
  muxDeselectAll();
  delay(5);
  Serial.print(F("# Mux A (0x71): ")); Serial.println(i2cAck(MUX_A) ? F("ACK") : F("MISSING"));
  Serial.print(F("# Mux B (0x73): ")); Serial.println(i2cAck(MUX_B) ? F("ACK") : F("MISSING"));

  Serial.println(F("# Initialising 9 ADCs (host-side tare happens after init)..."));

  for (uint8_t i = 0; i < N_ADCS; i++) {
    g_present[i] = initOneChip(i);
  }

  uint8_t okCount = 0;
  for (uint8_t i = 0; i < N_ADCS; i++) if (g_present[i]) okCount++;
  Serial.print(F("# Initialised "));
  Serial.print(okCount); Serial.print(F(" / ")); Serial.println(N_ADCS);

  if (okCount == 0) {
    Serial.println(F("# FATAL: no ADCs came up. Halting."));
    while (true) delay(1000);
  }

  // Capture baseline for every working chip and store. Stream values will
  // be raw minus baseline so each column sits near zero at rest.
  tareAllAdcs();

  Serial.print  (F("# Signal averaging: ")); Serial.print(SIGNAL_AVERAGING);
  Serial.println(F(" samples per chip per row"));
  Serial.println(F("# Streaming CSV (zeroed, mean + sample stdev per ADC)..."));
  Serial.print(F("t_ms"));
  for (uint8_t i = 0; i < N_ADCS; i++) {
    Serial.print(',');
    Serial.print(LAYOUT[i].label);
    Serial.print(F("_mean,"));
    Serial.print(LAYOUT[i].label);
    Serial.print(F("_std"));
  }
  Serial.println();
}

void loop() {
  uint32_t rowStart = millis();

  Row row;
  acquireRow(row);

  Serial.print(row.t_ms);
  for (uint8_t i = 0; i < N_ADCS; i++) {
    Serial.print(',');
    if (!row.present[i]) {
      Serial.print(F("NaN,NaN"));
    } else {
      Serial.print(row.mean[i]);
      Serial.print(',');
      Serial.print(row.std[i], 2);
    }
  }
  Serial.println();

  // Hold a steady row period. Acquisition already takes ~SIGNAL_AVERAGING ×
  // 100 ms at 10 SPS; if that's already longer than ROW_PERIOD_MS, this
  // delay is a no-op and rows just stream as fast as the chips produce data.
  uint32_t elapsed = millis() - rowStart;
  if (elapsed < ROW_PERIOD_MS) delay(ROW_PERIOD_MS - elapsed);
}
