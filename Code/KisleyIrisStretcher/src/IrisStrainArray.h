#pragma once
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_NAU7802.h>

namespace kisley {
namespace iris {

// Reads up to MAX_ADCS NAU7802 ADCs distributed across multiple TCA9548A
// I²C multiplexers. Performs round-robin signal averaging and emits a
// Bessel-corrected sample stdev as the per-row error bar.
//
// Defaults match the verified Kisley rig: 9 ADCs across two muxes
// (0x71 channels 0..4, 0x73 channels 0..3) on SDA=GPIO 3, SCL=GPIO 4.
// Override with setLayout() / setSdaScl() before begin().
//
// Non-obvious bus rules baked in (don't change unless you know why):
//   - I²C clock pinned at 10 kHz (bus is signal-marginal at 100 kHz)
//   - Deselect EVERY mux before selecting any channel, to avoid two
//     NAU7802s ending up on the bus and colliding
//   - Re-set Wire.setClock() after every nau.begin() call, since
//     Adafruit_I2CDevice::begin() invokes Wire.begin() internally and
//     resets the bus clock to the default 100 kHz on ESP32
//   - Skip Adafruit's calibrate(OFFSET); its completion wait-loop is
//     inverted and never blocks, leaving the chip stuck mid-cal.
//     Tare is done host-side by averaging N raw samples at begin().
//
// See feedback_adafruit_nau7802_calibrate_bug and
// reference_strain_array_rig in saved memory for the full backstory.
class IrisStrainArray {
public:
  static constexpr uint8_t  MAX_ADCS  = 16;
  static constexpr uint8_t  MAX_MUXES = 4;
  static constexpr uint16_t MAX_AVG   = 256;

  struct Slot {
    uint8_t     muxAddr;
    uint8_t     ch;
    const char* label;
  };

  // One acquired row: mean + sample stdev per ADC.
  struct Row {
    uint32_t t_ms;
    int32_t  mean[MAX_ADCS];
    float    std [MAX_ADCS];
    bool     present[MAX_ADCS];
  };

  IrisStrainArray(TwoWire& wire = Wire, Stream& io = Serial);

  // ---- Configuration (call before begin()) ----
  void setLayout(const Slot* slots, uint8_t n);
  void setSdaScl(uint8_t sda, uint8_t scl);
  void setI2cClock(uint32_t hz);
  void setSignalAveraging(uint16_t n);
  void setTareSamples(uint16_t n);
  void setSampleTimeoutMs(uint16_t ms);
  // NAU7802 conversion rate per chip: 10/20/40/80/320 SPS. Higher = faster
  // streaming, noisier per-sample. Default 320 SPS for max throughput;
  // dial down if you see excessive sample noise. Call before begin().
  void setSampleRate(NAU7802_SampleRate r);
  // Microseconds to settle after writing a mux channel-select. Default
  // 100 µs — TCA9548A switches in < 1 µs and the NAU7802 is already
  // converting continuously, so the wait only needs to cover I²C
  // pin/level settle. Raise if you see occasional bad reads on the
  // first sample after a mux switch.
  void setMuxSettleMicros(uint16_t us);
  // When true, every per-chip read polls the NAU7802's "conversion ready"
  // status bit before reading the data register. Defaults to FALSE because
  // the round-robin acquireRow loop gives each chip ≥ 9 × per-chip-cost
  // between successive reads (far longer than the chip's sample cycle at
  // 320 SPS), so a fresh conversion is always waiting in the data register
  // by the time we route back. Set to TRUE if you have a single-chip
  // layout and read faster than the chip's sample rate — then you genuinely
  // need to wait for the next conversion to avoid duplicating the last value.
  void setWaitForReadyOnRead(bool on);

  // ---- Lifecycle ----
  // Scans for the unique mux addresses declared in the layout, initialises
  // every present NAU7802 (internal cal + first reading), then runs a
  // host-side tare. Returns true iff at least one ADC came up.
  bool begin();

  // Re-runs initialise + tare without rebooting. Use after physically
  // reconnecting or to recapture the zero baseline.
  bool rescan();

  // Re-runs the tare pass only (chips stay initialised).
  void tare();

  // ---- Acquisition ----
  // Round-robin sampling: signalAveraging passes, one sample per ADC per
  // pass. Output mean is zeroed (raw − baseline). std is sample stdev
  // (Bessel-corrected; 0 when only one sample taken).
  void acquireRow(Row& out);

  // ---- Inspection ----
  uint8_t            adcCount()           const { return _nSlots; }
  bool               isPresent(uint8_t i) const { return i < _nSlots && _present[i]; }
  const char*        label(uint8_t i)     const { return i < _nSlots ? _layout[i].label : ""; }
  int32_t            baseline(uint8_t i)  const { return i < _nSlots ? _baseline[i] : 0; }
  uint16_t           signalAveraging()    const { return _signalAvg; }
  NAU7802_SampleRate sampleRate()         const { return _rate; }

  // Push the current sample rate (set via setSampleRate) to every already-
  // initialised chip without re-running begin() / re-taring. Use after a
  // runtime setSampleRate() call — e.g. from the LCD's Xsamplerate edit —
  // to make the new rate take effect on chips that were initialised earlier
  // at a different rate. Returns the count of chips actually updated.
  uint8_t applySampleRateLive();

  // ---- Emitters ----
  // "t_ms,LABEL1_mean,LABEL1_std,LABEL2_mean,LABEL2_std,..."
  void printCsvHeader(Stream& out) const;
  void printCsvRow(Stream& out, const Row& row) const;

  // Just the data portion, no t_ms (for embedding in a wider row).
  void printCsvHeaderTrailing(Stream& out) const;
  void printCsvRowTrailing(Stream& out, const Row& row) const;

  void printStatus(Stream& out) const;

private:
  TwoWire* _wire;
  Stream*  _io;

  Slot     _layout[MAX_ADCS];
  uint8_t  _nSlots = 0;
  uint8_t  _muxAddrs[MAX_MUXES];   // unique mux addresses from layout
  uint8_t  _nMuxes = 0;

  uint8_t  _sda = 3, _scl = 4;
  uint32_t _i2cClockHz       = 10000;
  uint16_t _signalAvg        = 2;   // 2 samples per ADC per row; raise for less noise / lower row rate
  uint16_t _tareSamples      = 16;
  uint16_t _sampleTimeoutMs  = 200;
  uint16_t _muxSettleUs      = 100;
  bool     _waitForReadyOnRead = false;
  // Library default = NAU7802's documented maximum (320 SPS = enum value 7).
  // The Adafruit_NAU7802 enum exposes 10/20/40/80/320 SPS; 320 is the ceiling.
  // Lower this only if you need quieter samples at the cost of throughput.
  NAU7802_SampleRate _rate   = NAU7802_RATE_320SPS;

  // 0xFF = no mux currently routed. Tracking this lets _muxRoute skip
  // the "deselect every other mux" pass when consecutive reads stay on
  // the same mux — the dominant case in the round-robin sample loop.
  uint8_t  _activeMuxAddr    = 0xFF;

  Adafruit_NAU7802 _nau;
  bool     _present[MAX_ADCS];
  int32_t  _baseline[MAX_ADCS];

  // Internals
  void _computeUniqueMuxes();
  void _restoreClock();
  bool _i2cAck(uint8_t addr);
  void _muxDeselectAll();
  void _muxRoute(uint8_t muxAddr, uint8_t ch);
  bool _waitSample(uint16_t timeoutMs);
  bool _initOneChip(uint8_t i);
  bool _readOneChipRaw(uint8_t i, int32_t& out);
};

// Default layout exposed so labs can reference it before overriding.
namespace IrisStrainArrayDefaults {
  extern const IrisStrainArray::Slot DEFAULT_LAYOUT[9];
  constexpr uint8_t DEFAULT_LAYOUT_SIZE = 9;
}

} // namespace iris
} // namespace kisley
