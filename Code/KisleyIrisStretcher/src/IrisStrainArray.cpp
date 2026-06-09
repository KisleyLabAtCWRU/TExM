#include "IrisStrainArray.h"
#include <math.h>

namespace kisley {
namespace iris {

namespace IrisStrainArrayDefaults {
  // Verified Kisley rig: Mux A (0x71) ch0..4, Mux B (0x73) ch0..3
  const IrisStrainArray::Slot DEFAULT_LAYOUT[9] = {
    {0x71, 0, "ADC1"},
    {0x71, 1, "ADC2"},
    {0x71, 2, "ADC3"},
    {0x71, 3, "ADC4"},
    {0x71, 4, "ADC5"},
    {0x73, 0, "ADC6"},
    {0x73, 1, "ADC7"},
    {0x73, 2, "ADC8"},
    {0x73, 3, "ADC9"},
  };
}

IrisStrainArray::IrisStrainArray(TwoWire& wire, Stream& io)
  : _wire(&wire), _io(&io) {
  // Seed with default layout.
  setLayout(IrisStrainArrayDefaults::DEFAULT_LAYOUT,
            IrisStrainArrayDefaults::DEFAULT_LAYOUT_SIZE);
  for (uint8_t i = 0; i < MAX_ADCS; i++) {
    _present[i]  = false;
    _baseline[i] = 0;
  }
}

// ============================ Configuration ============================

void IrisStrainArray::setLayout(const Slot* slots, uint8_t n) {
  if (n > MAX_ADCS) n = MAX_ADCS;
  for (uint8_t i = 0; i < n; i++) _layout[i] = slots[i];
  _nSlots = n;
  _computeUniqueMuxes();
}

void IrisStrainArray::setSdaScl(uint8_t sda, uint8_t scl) { _sda = sda; _scl = scl; }
void IrisStrainArray::setI2cClock(uint32_t hz)            { _i2cClockHz = hz; }
void IrisStrainArray::setSignalAveraging(uint16_t n) {
  if (n < 1)       n = 1;
  if (n > MAX_AVG) n = MAX_AVG;
  _signalAvg = n;
}
void IrisStrainArray::setTareSamples(uint16_t n)          { _tareSamples = n ? n : 1; }
void IrisStrainArray::setSampleTimeoutMs(uint16_t ms)     { _sampleTimeoutMs = ms; }
void IrisStrainArray::setSampleRate(NAU7802_SampleRate r) { _rate = r; }
void IrisStrainArray::setMuxSettleMicros(uint16_t us)     { _muxSettleUs = us; }
void IrisStrainArray::setWaitForReadyOnRead(bool on)      { _waitForReadyOnRead = on; }

uint8_t IrisStrainArray::applySampleRateLive() {
  uint8_t ok = 0;
  for (uint8_t i = 0; i < _nSlots; i++) {
    if (!_present[i]) continue;
    const Slot& s = _layout[i];
    _muxRoute(s.muxAddr, s.ch);
    _nau.setRate(_rate);
    ok++;
  }
  return ok;
}

void IrisStrainArray::_computeUniqueMuxes() {
  _nMuxes = 0;
  for (uint8_t i = 0; i < _nSlots; i++) {
    bool seen = false;
    for (uint8_t m = 0; m < _nMuxes; m++) {
      if (_muxAddrs[m] == _layout[i].muxAddr) { seen = true; break; }
    }
    if (!seen && _nMuxes < MAX_MUXES) {
      _muxAddrs[_nMuxes++] = _layout[i].muxAddr;
    }
  }
}

// ============================ I2C primitives ============================

void IrisStrainArray::_restoreClock() { _wire->setClock(_i2cClockHz); }

bool IrisStrainArray::_i2cAck(uint8_t addr) {
  _wire->beginTransmission(addr);
  return _wire->endTransmission() == 0;
}

void IrisStrainArray::_muxDeselectAll() {
  for (uint8_t m = 0; m < _nMuxes; m++) {
    _wire->beginTransmission(_muxAddrs[m]);
    _wire->write((uint8_t)0x00);
    _wire->endTransmission();
  }
  _activeMuxAddr = 0xFF;
}

void IrisStrainArray::_muxRoute(uint8_t muxAddr, uint8_t ch) {
  // Only deselect the OTHER muxes when we're switching FROM a different
  // mux. Within a run of reads that stay on the same mux (the dominant
  // case in the round-robin loop — 7 of 9 chip-switches per pass on the
  // default Kisley layout), the other muxes are already deselected from
  // the last cross-mux switch, so the deselect transactions are pure
  // waste. Skipping them cuts ~7 I²C transactions per signal-averaging
  // pass.
  if (muxAddr != _activeMuxAddr) {
    for (uint8_t m = 0; m < _nMuxes; m++) {
      if (_muxAddrs[m] == muxAddr) continue;
      _wire->beginTransmission(_muxAddrs[m]);
      _wire->write((uint8_t)0x00);
      _wire->endTransmission();
    }
    _activeMuxAddr = muxAddr;
  }
  _wire->beginTransmission(muxAddr);
  _wire->write((uint8_t)(1u << ch));
  _wire->endTransmission();
  // Settle after writing the channel select. TCA9548A switches in <1 µs;
  // the NAU7802 is already converting continuously; the old delay(1) was
  // paranoia at the cost of 1 ms × (9 chips × signalAvg) per row.
  if (_muxSettleUs) delayMicroseconds(_muxSettleUs);
}

bool IrisStrainArray::_waitSample(uint16_t timeoutMs) {
  uint32_t t0 = millis();
  while (!_nau.available()) {
    if (millis() - t0 > timeoutMs) return false;
  }
  return true;
}

// ============================ Per-chip init ============================

bool IrisStrainArray::_initOneChip(uint8_t i) {
  const Slot& s = _layout[i];
  _io->print(F("# Init "));
  _io->print(s.label);
  _io->print(F(" (mux 0x"));
  _io->print(s.muxAddr, HEX);
  _io->print(F(" ch"));
  _io->print(s.ch);
  _io->print(F("): "));

  _restoreClock();
  _muxRoute(s.muxAddr, s.ch);

  if (!_i2cAck(0x2A)) {
    _io->println(F("NACK (no chip)"));
    return false;
  }

  if (!_nau.begin(_wire)) {
    _restoreClock();
    _io->println(F("nau.begin FAILED"));
    return false;
  }
  _restoreClock();   // Adafruit lib reset the clock internally

  _nau.setLDO(NAU7802_3V0);
  _nau.setGain(NAU7802_GAIN_128);
  _nau.setRate(_rate);

  // Internal cal — zeroes the ADC's intrinsic input offset.
  // Adafruit's calibrate() doesn't actually wait for completion (loop
  // inverted), so we add a fixed delay long enough at 10 SPS.
  _nau.calibrate(NAU7802_CALMOD_INTERNAL);
  delay(200);
  if (_waitSample(800)) (void)_nau.read();   // drain stale sample

  // No on-device OFFSET cal — see header comment. Host-side tare runs
  // after every chip is initialised.

  if (!_waitSample(800)) {
    _io->println(F("OK begin but no sample after cal"));
    return false;
  }
  int32_t first = _nau.read();
  _io->print(F("OK (first read = "));
  _io->print(first);
  _io->println(F(")"));
  return true;
}

bool IrisStrainArray::_readOneChipRaw(uint8_t i, int32_t& out) {
  if (!_present[i]) return false;
  const Slot& s = _layout[i];
  // _restoreClock() omitted from the steady-state read path: the bus
  // clock is set in begin()/rescan() (right after the only callers that
  // reset it — Adafruit_I2CDevice::begin() inside _initOneChip) and is
  // not perturbed by any operation in the read loop. Calling setClock()
  // every chip read was harmless but added ~µs × 9 chips × signalAvg
  // per row.
  _muxRoute(s.muxAddr, s.ch);
  // Skip the conversion-ready poll by default. In the round-robin loop
  // the chip has been converting continuously for ≥ 9 × per-chip-cost
  // (≫ 1/sampleRate) since we last read it, so the data register always
  // holds a fresh conversion. Opt back in with setWaitForReadyOnRead(true)
  // for single-chip layouts that read faster than the chip's sample rate.
  if (_waitForReadyOnRead && !_waitSample(_sampleTimeoutMs)) return false;
  out = _nau.read();
  return true;
}

// ============================ Lifecycle ============================

bool IrisStrainArray::begin() {
  _wire->begin(_sda, _scl);
  _restoreClock();

  _io->println();
  _io->println(F("# IrisStrainArray begin"));
  _io->print  (F("# I2C clock: ")); _io->print(_i2cClockHz); _io->println(F(" Hz"));

  _muxDeselectAll();
  delay(5);

  // Confirm every unique mux ACKs.
  for (uint8_t m = 0; m < _nMuxes; m++) {
    _io->print(F("# Mux 0x"));
    _io->print(_muxAddrs[m], HEX);
    _io->print(F(": "));
    _io->println(_i2cAck(_muxAddrs[m]) ? F("ACK") : F("MISSING"));
  }

  for (uint8_t i = 0; i < _nSlots; i++) {
    _present[i] = _initOneChip(i);
  }

  uint8_t okCount = 0;
  for (uint8_t i = 0; i < _nSlots; i++) if (_present[i]) okCount++;
  _io->print(F("# Initialised "));
  _io->print(okCount);
  _io->print(F(" / "));
  _io->println(_nSlots);

  if (okCount == 0) return false;

  tare();
  return true;
}

bool IrisStrainArray::rescan() {
  for (uint8_t i = 0; i < MAX_ADCS; i++) _present[i] = false;
  return begin();
}

void IrisStrainArray::tare() {
  _io->print(F("# Taring "));
  _io->print(_tareSamples);
  _io->println(F(" samples per chip..."));
  _io->println(F("# IMPORTANT: keep the rig undisturbed for the next few seconds."));

  for (uint8_t i = 0; i < _nSlots; i++) {
    if (!_present[i]) { _baseline[i] = 0; continue; }
    int64_t sum = 0;
    uint16_t got = 0;
    for (uint16_t k = 0; k < _tareSamples; k++) {
      int32_t s;
      if (_readOneChipRaw(i, s)) { sum += s; got++; }
    }
    _baseline[i] = (got > 0) ? (int32_t)(sum / got) : 0;
    _io->print(F("#   "));
    _io->print(_layout[i].label);
    _io->print(F(" baseline = "));
    _io->print(_baseline[i]);
    _io->print(F("  (n="));
    _io->print(got);
    _io->println(F(")"));
  }
  _io->println(F("# Tare complete."));
}

// ============================ Acquisition ============================

void IrisStrainArray::acquireRow(Row& out) {
  out.t_ms = millis();

  int64_t  sum  [MAX_ADCS] = {0};
  int64_t  sumsq[MAX_ADCS] = {0};
  uint16_t count[MAX_ADCS] = {0};

  for (uint16_t pass = 0; pass < _signalAvg; pass++) {
    for (uint8_t i = 0; i < _nSlots; i++) {
      if (!_present[i]) continue;
      int32_t raw;
      if (_readOneChipRaw(i, raw)) {
        const int64_t zeroed = (int64_t)raw - (int64_t)_baseline[i];
        sum[i]   += zeroed;
        sumsq[i] += zeroed * zeroed;
        count[i] += 1;
      }
    }
  }

  for (uint8_t i = 0; i < _nSlots; i++) {
    out.present[i] = _present[i] && (count[i] > 0);
    if (!out.present[i]) { out.mean[i] = 0; out.std[i] = 0.0f; continue; }

    const int64_t  s = sum[i];
    const uint16_t n = count[i];
    int64_t q = s / (int64_t)n;
    int64_t r = s - q * (int64_t)n;
    if (r >  (int64_t)n / 2)        q += 1;
    else if (r < -(int64_t)n / 2)   q -= 1;
    out.mean[i] = (int32_t)q;

    if (n <= 1) {
      out.std[i] = 0.0f;
    } else {
      const double dS  = (double)s;
      const double dSS = (double)sumsq[i];
      const double dN  = (double)n;
      double var = (dSS - (dS * dS) / dN) / (dN - 1.0);
      if (var < 0.0) var = 0.0;
      out.std[i] = (float)sqrt(var);
    }
  }
}

// ============================ Emitters ============================

void IrisStrainArray::printCsvHeader(Stream& out) const {
  out.print(F("t_ms"));
  printCsvHeaderTrailing(out);
  out.println();
}

void IrisStrainArray::printCsvHeaderTrailing(Stream& out) const {
  for (uint8_t i = 0; i < _nSlots; i++) {
    out.print(',');
    out.print(_layout[i].label);
    out.print(F("_mean,"));
    out.print(_layout[i].label);
    out.print(F("_std"));
  }
}

void IrisStrainArray::printCsvRow(Stream& out, const Row& row) const {
  out.print(row.t_ms);
  printCsvRowTrailing(out, row);
  out.println();
}

void IrisStrainArray::printCsvRowTrailing(Stream& out, const Row& row) const {
  for (uint8_t i = 0; i < _nSlots; i++) {
    out.print(',');
    if (!row.present[i]) {
      out.print(F("NaN,NaN"));
    } else {
      out.print(row.mean[i]);
      out.print(',');
      out.print(row.std[i], 2);
    }
  }
}

void IrisStrainArray::printStatus(Stream& out) const {
  out.println(F("# IrisStrainArray status:"));
  out.print(F("#   Muxes: "));
  for (uint8_t m = 0; m < _nMuxes; m++) {
    out.print(F("0x"));
    if (_muxAddrs[m] < 0x10) out.print('0');
    out.print(_muxAddrs[m], HEX);
    if (m + 1 < _nMuxes) out.print(',');
  }
  out.println();
  out.print(F("#   Signal averaging: ")); out.println(_signalAvg);
  out.print(F("#   I2C clock: ")); out.print(_i2cClockHz); out.println(F(" Hz"));
  uint8_t okCount = 0;
  for (uint8_t i = 0; i < _nSlots; i++) if (_present[i]) okCount++;
  out.print(F("#   ADCs present: ")); out.print(okCount);
  out.print(F(" / ")); out.println(_nSlots);
}

} // namespace iris
} // namespace kisley
