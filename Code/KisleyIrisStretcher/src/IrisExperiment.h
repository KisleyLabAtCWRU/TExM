#pragma once
#include <Arduino.h>
#include "IrisStretcher.h"
#include "IrisStrainArray.h"

namespace kisley {
namespace iris {

class IrisExperimentRunner;   // fwd
class IrisMenuUI;             // fwd; declared in IrisMenuUI.h

// One experiment = a name + a sequence of target Ex values. Hold time
// between waypoints is global to the runner, not per-step.
//
// Signed Ex convention from IrisStretcher::gotoExpansion:
//   targetEx > 0 → CW;  targetEx < 0 → CCW;  |targetEx| ≤ 1 → center.
//
// Example: 1x → 3.4x CW → 1x:
//   const float kExp1[] = { 1.0f, 3.4f, 1.0f };
//   const IrisExperiment kExp1Def = { "Exp1", kExp1, 3 };
//   runner.registerExperiment(kExp1Def);
//   ui.attachRunner(runner);
//
// For ramps/oscillations/anything not expressible as a step array,
// supply a customRun function instead. Leave `targets` null in that case.
struct IrisExperiment {
  const char*  name;
  const float* targets;
  uint8_t      nTargets;

  using RunFn = void(*)(IrisExperimentRunner& runner, void* user);
  RunFn   customRun  = nullptr;
  void*   customUser = nullptr;
};

// Drives the motor through an IrisExperiment's targets while
// synchronously streaming CSV rows to a Stream. Cooperative: motion
// itself is blocking (one IrisStretcher::gotoExpansion call per waypoint),
// but hold time is non-blocking and checks for abort each iteration.
class IrisExperimentRunner {
public:
  static constexpr uint8_t MAX_EXPERIMENTS = 12;

  IrisExperimentRunner(IrisStretcher&    stretcher,
                       IrisStrainArray&  strain,
                       Stream&           io = Serial);

  // ---- Configuration ----
  void setMotionLogPeriodMs(uint32_t ms)   { _motionLogMs = ms; }
  void setHoldLogPeriodMs(uint32_t ms)     { _holdLogMs   = ms; }
  void setHoldMs(uint32_t ms)              { _holdMs      = ms; }
  // Step-modulo filter for motion logging. Default 1 = every step is
  // eligible (still subject to the time-based motionLogPeriodMs gate).
  // Set to 2 to only log on every other step (the 1st, 3rd, 5th, … step
  // of each gotoExpansion — i.e. odd-numbered steps). Set to N to log
  // one row per N steps. Combines with the time gate: a step row is
  // emitted only if BOTH `stepIdx % N == 0` AND the time gate has expired.
  // Set the time gate to 0 (setMotionLogPeriodMs(0)) if you want purely
  // step-based emission.
  void setMotionLogEveryNSteps(uint16_t n) { _motionLogEveryN = n ? n : 1; }
  uint16_t motionLogEveryNSteps() const    { return _motionLogEveryN; }

  // ---- Experiment registry ----
  // Returns false if MAX_EXPERIMENTS reached.
  bool registerExperiment(const IrisExperiment& exp);
  uint8_t                experimentCount() const { return _nExps; }
  const IrisExperiment*  experiment(uint8_t i) const;
  const IrisExperiment*  findExperiment(const char* name) const;

  // ---- Control ----
  // Queues a run. Actual execution happens in update().
  void requestRun(const IrisExperiment& exp);
  // Same, by registered name. Returns false if name not found.
  bool requestRunByName(const char* name);
  // Mid-run abort request — honoured between motion segments.
  void requestAbort()                      { _abortRequested = true; }

  // Attach the UI so the LCD gets refreshed during motion. Without this
  // the LCD stays stuck on whatever it last drew before the (blocking)
  // gotoExpansion call took over the loop — the encoder-step callback
  // is the only opportunity to repaint mid-motion. The runner calls
  // ui.refresh() at the same cadence as motion CSV logging.
  void attachUi(IrisMenuUI& ui)            { _ui = &ui; }

  // ---- Loop hook ----
  // Drive the state machine. Call once per main loop() iteration.
  void update();

  // ---- Inspection (for UI/serial status) ----
  bool   isRunning()       const { return _state != State::IDLE; }
  bool   isHolding()       const { return _state == State::HOLDING; }
  const  IrisExperiment*   currentExperiment() const { return _curExp; }
  uint8_t currentStepIndex() const { return _curStep; }
  float  currentTargetEx() const { return _curTarget; }

  // ---- Continuous strain streaming (independent of experiments) ----
  // When true, emits CSV rows from the strain array every motionLogMs ms,
  // tagged with exp="Xstrain" and state reflecting motor activity.
  void setStreamStrain(bool on);
  bool isStreamingStrain() const { return _streamStrain; }

  // ---- Row emission (also usable from a customRun callback) ----
  // state: 'M' = motor in motion, 'H' = holding at target, 'S' = static streaming
  void emitRow(char state);

private:
  enum class State : uint8_t { IDLE, STARTING, MOVING, HOLDING, ABORTING, DONE };

  IrisStretcher&    _stretcher;
  IrisStrainArray&  _strain;
  Stream&           _io;

  // Config
  uint32_t _motionLogMs    = 200;
  uint32_t _holdLogMs      = 100;
  uint32_t _holdMs         = 2000;
  uint16_t _motionLogEveryN = 1;  // log every Nth step (1 = no skipping)

  // Registry
  const IrisExperiment* _exps[MAX_EXPERIMENTS] = {nullptr};
  uint8_t _nExps = 0;

  // Run state
  State                 _state          = State::IDLE;
  const IrisExperiment* _curExp         = nullptr;
  uint8_t               _curStep        = 0;
  float                 _curTarget      = 1.0f;
  uint32_t              _holdStartMs    = 0;
  uint32_t              _lastLogMs      = 0;
  volatile bool         _abortRequested = false;

  // Pending request set by requestRun(); update() picks it up.
  const IrisExperiment* _pendingExp     = nullptr;

  // Streaming
  bool                  _streamStrain   = false;
  uint32_t              _lastStreamMs   = 0;

  // Optional UI hook so the LCD can be repainted from inside the step
  // callback while motion blocks the main loop.
  IrisMenuUI*           _ui             = nullptr;

  // Stepper-callback bridge (used during motion to emit rows on a timer
  // even though the motor loop owns the thread). Active runner is set
  // before each gotoExpansion call.
  static IrisExperimentRunner* s_activeRunner;
  static void _stepCallback(long stepIdx, void* user);

  void _emitHeader();
  void _moveTo(float targetEx);
};

} // namespace iris
} // namespace kisley
