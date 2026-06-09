#include "IrisExperiment.h"
#include "IrisMenuUI.h"
#include <string.h>

namespace kisley {
namespace iris {

IrisExperimentRunner* IrisExperimentRunner::s_activeRunner = nullptr;

IrisExperimentRunner::IrisExperimentRunner(IrisStretcher&   stretcher,
                                           IrisStrainArray& strain,
                                           Stream&          io)
  : _stretcher(stretcher), _strain(strain), _io(io) {}

// ---- Registry ----

bool IrisExperimentRunner::registerExperiment(const IrisExperiment& exp) {
  if (_nExps >= MAX_EXPERIMENTS) return false;
  _exps[_nExps++] = &exp;
  return true;
}

const IrisExperiment* IrisExperimentRunner::experiment(uint8_t i) const {
  return (i < _nExps) ? _exps[i] : nullptr;
}

const IrisExperiment* IrisExperimentRunner::findExperiment(const char* name) const {
  if (!name) return nullptr;
  for (uint8_t i = 0; i < _nExps; i++) {
    if (_exps[i] && strcmp(_exps[i]->name, name) == 0) return _exps[i];
  }
  return nullptr;
}

// ---- Control ----

void IrisExperimentRunner::requestRun(const IrisExperiment& exp) {
  _pendingExp = &exp;
}

bool IrisExperimentRunner::requestRunByName(const char* name) {
  const IrisExperiment* e = findExperiment(name);
  if (!e) return false;
  requestRun(*e);
  return true;
}

void IrisExperimentRunner::setStreamStrain(bool on) {
  _streamStrain = on;
  if (on) {
    // Re-tare so every streaming session uses a fresh baseline. Caller
    // is responsible for the rig being undisturbed at this moment.
    _strain.tare();
    _emitHeader();
    _lastStreamMs = 0;
  }
}

// ---- CSV emission ----

void IrisExperimentRunner::_emitHeader() {
  _io.print(F("exp,t_ms,steps,target_ex,state"));
  _strain.printCsvHeaderTrailing(_io);
  _io.println();
}

void IrisExperimentRunner::emitRow(char state) {
  IrisStrainArray::Row row;
  _strain.acquireRow(row);

  _io.print(_curExp ? _curExp->name : "Xstrain");
  _io.print(','); _io.print(row.t_ms);
  _io.print(','); _io.print(_stretcher.currentSteps());
  _io.print(','); _io.print(_curTarget, 3);
  _io.print(','); _io.print(state);
  _strain.printCsvRowTrailing(_io, row);
  _io.println();
}

// ---- Motion with periodic row emission ----

void IrisExperimentRunner::_stepCallback(long stepIdx, void* /*user*/) {
  if (!s_activeRunner) return;
  IrisExperimentRunner* r = s_activeRunner;
  // Step-modulo filter. With N=1 (default) every step is eligible. With
  // N=2 only stepIdx 0, 2, 4, … pass — i.e. the 1st, 3rd, 5th, … step of
  // each gotoExpansion ("every odd-numbered step"). Phase is per-move:
  // each gotoExpansion starts with stepIdx 0, so the first step of every
  // move always passes regardless of N.
  if (r->_motionLogEveryN > 1 && (stepIdx % r->_motionLogEveryN) != 0) return;
  uint32_t now = millis();
  if (now - r->_lastLogMs >= r->_motionLogMs) {
    r->emitRow('M');
    // Repaint the LCD with the live target/state. gotoExpansion blocks
    // the main loop so this is the only place mid-motion the LCD can
    // be refreshed.
    if (r->_ui) r->_ui->refresh();
    r->_lastLogMs = now;
  }
}

void IrisExperimentRunner::_moveTo(float targetEx) {
  _curTarget = targetEx;
  _lastLogMs = millis();

  s_activeRunner = this;
  _stretcher.onEachStep(&IrisExperimentRunner::_stepCallback, this);
  _stretcher.gotoExpansion(targetEx);
  _stretcher.onEachStep(nullptr, nullptr);
  s_activeRunner = nullptr;

  // Emit one final row at the end of motion so the arrival is captured.
  emitRow('M');
}

// ---- State machine ----

void IrisExperimentRunner::update() {
  switch (_state) {
    case State::IDLE: {
      // Pick up a pending run request.
      if (_pendingExp) {
        _curExp    = _pendingExp;
        _pendingExp = nullptr;
        _abortRequested = false;
        _state = State::STARTING;
      } else if (_streamStrain) {
        // Continuous streaming, no experiment.
        const uint32_t now = millis();
        if (now - _lastStreamMs >= _motionLogMs) {
          _curTarget = 0.0f;
          emitRow('S');
          _lastStreamMs = now;
        }
      }
      break;
    }

    case State::STARTING: {
      // Suppress the stretcher's intrinsic per-move report for the duration
      // of the run — its multi-line key=value blocks would corrupt the CSV
      // stream. Restored in DONE/ABORTING.
      _stretcher.setMoveReporting(false);

      // Re-tare ADCs before every experiment so the captured strain
      // values are referenced to the rig's current pre-run baseline.
      // The motor has not started moving yet — by convention the rig
      // is undisturbed at this point.
      _strain.tare();

      _io.println();
      _io.print(F("# experiment: ")); _io.println(_curExp->name);
      _io.print(F("# steps: ")); _io.println(_curExp->nTargets);
      _io.print(F("# motion log every ")); _io.print(_motionLogMs); _io.println(F(" ms"));
      _io.print(F("# hold ")); _io.print(_holdMs); _io.print(F(" ms, log every "));
      _io.print(_holdLogMs); _io.println(F(" ms"));
      _emitHeader();
      _curStep = 0;

      // Custom run takes over end-to-end if present.
      if (_curExp->customRun) {
        _curExp->customRun(*this, _curExp->customUser);
        _state = State::DONE;
        break;
      }
      _state = State::MOVING;
      break;
    }

    case State::MOVING: {
      if (_abortRequested) { _state = State::ABORTING; break; }
      if (_curStep >= _curExp->nTargets) { _state = State::DONE; break; }
      _moveTo(_curExp->targets[_curStep]);
      _state = State::HOLDING;
      _holdStartMs = millis();
      _lastLogMs   = millis();
      break;
    }

    case State::HOLDING: {
      if (_abortRequested) { _state = State::ABORTING; break; }
      const uint32_t now = millis();
      if (now - _lastLogMs >= _holdLogMs) {
        emitRow('H');
        _lastLogMs = now;
      }
      if (now - _holdStartMs >= _holdMs) {
        _curStep++;
        _state = (_curStep >= _curExp->nTargets) ? State::DONE : State::MOVING;
      }
      break;
    }

    case State::ABORTING: {
      _stretcher.setMoveReporting(true);   // restore intrinsic move report
      _io.println(F("# experiment aborted"));
      _state = State::IDLE;
      _curExp = nullptr;
      _abortRequested = false;
      break;
    }

    case State::DONE: {
      _stretcher.setMoveReporting(true);   // restore intrinsic move report
      _io.println(F("# experiment complete"));
      _state = State::IDLE;
      _curExp = nullptr;
      break;
    }
  }
}

} // namespace iris
} // namespace kisley
