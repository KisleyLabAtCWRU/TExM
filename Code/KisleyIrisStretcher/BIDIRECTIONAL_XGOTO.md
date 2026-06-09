# Bidirectional Xgoto — Implementation Plan

> **Goal.** Extend `Xgoto` so the iris can be driven in both directions
> from a unified `Ex = 1.0` "center." Inputs above 1.0 should still
> rotate clockwise as today. Inputs below 1.0 should rotate
> counter-clockwise into compression territory.

---

## Locked decisions — revised 2026-05-12

The earlier "mirror-via-Ex-domain" scheme (where `Ex < 1` meant real
contraction) is **superseded**. New convention: input is always a
**magnitude in `[1.0, maxEx]`** plus a **direction tag**. CCW range
mirrors CW range — both go from 1.0 up to `maxEx`.

1. **Signed-value API.** `IrisStretcher::gotoExpansion(double signedEx)`
   uses the **sign** to select direction:
   - `signedEx > 0` → CW, motor θ = `+findTheta(|signedEx|)`
   - `signedEx < 0` → CCW, motor θ = `−findTheta(|signedEx|)`
   - `|signedEx| ≤ 1.0` → return to center (θ = 0)
   - `|signedEx| ≥ maxEx` → out-of-range error
2. **Kinematic story is single-sided.** `findTheta` always solves the
   CW branch (positive θ, target Eₓ > 1). The CCW move uses the same
   θ magnitude with the sign flipped — the motor rotates by the same
   physical angle in the opposite direction. The actual Eₓ the iris
   achieves on the CCW side is whatever the mechanism produces at
   that motor angle (the model has no Eₓ < 1 domain). The "Resulting
   expansion" print reflects this — verified via `computeEx` for CW,
   echoed magnitude for CCW.
3. **No `minEx` field.** Removed. Magnitude is always ≥ 1.0; both
   sides clamp to `maxEx` from above.
4. **LCD edit screen**: direction is a separate UI state (member
   `_xGotoDirCw`). Encoder edits magnitude only, in `[1.0, maxEx]`.
   - **DOWN button** toggles CW ↔ CCW.
   - SW button toggles fine/coarse (unchanged).
   - ACCEPT commits `gotoExpansion(±magnitude)` based on direction.
   - Display: `"1.350 CW  [SW]"` or `"1.350 CCW [SW]"`. F/C indicator
     stays on the top row.
5. **Serial syntax**: `Xgoto <magnitude> [cw|ccw]`. Direction token is
   case-insensitive; defaults to `cw`. Examples:
   - `Xgoto 1.35` → CW magnitude 1.35
   - `Xgoto 1.35 ccw` → CCW by the same θ magnitude
   - `Xgoto 1.0` (any direction) → return to center
6. **Step-counter invariant unchanged**: round-trips through center
   land on byte-identical absolute step positions, because every
   move computes `targetSteps` from absolute θ.

---

## 1. How `Xgoto` works today

### Three layers, top to bottom

1. **`IrisStretcher::gotoExpansion(targetEx)`** *(IrisStretcher.cpp:27)* —
   the entry point both serial and LCD UI call.
   ```cpp
   bool gotoExpansion(double targetEx) {
     const double angle = findTheta(targetEx);   // inverse kinematics
     if (targetEx <= 1.001 || targetEx >= _geo.maxEx || isnan(angle)) {
       _io.println("Error: ...");
       return false;
     }
     rotateThetaRadians(angle);                   // motion
     return true;
   }
   ```
   It hard-rejects anything `≤ 1.001`. **This is the only gate that
   stops contraction values today.**

2. **`IrisKinematics::findTheta(geo, targetEx, thetaLow=0.0001, thetaHigh=0.6)`**
   *(IrisKinematics.cpp:39)* — inverse kinematics. Bisection (50 iter)
   then Newton-Raphson (50 iter) over the **fixed bracket
   `[0.0001, 0.6] rad`**. Returns `NaN` if there's no sign change
   inside the bracket.
   - At `Theta = 0.0001` the math has a near-singularity (`cos(theta)→0`)
     because `theta = Theta − π/2`. That's why the lower bound isn't zero.
   - The bracket was tuned for expansion only.

3. **`IrisStepperDriver::rotateThetaRadians(geo, theta)`**
   *(IrisStepperDriver.cpp:19)* — motion. Computes
   `targetSteps = θ · pulsesPerRev · g / (2π)` (signed),
   then `deltaSteps = targetSteps − currentSteps`. **Direction pin is
   already determined by the sign of `deltaSteps`** — `dir = (deltaSteps ≥ 0)`.
   The AccelStepper position counter updates every step in both
   directions.

### Key observation: the motion layer is already bidirectional

`rotateThetaRadians` doesn't care about sign of `theta`. If
`findTheta` ever returned a *negative* θ, the stepper would
correctly drive CCW and the position counter would go negative.
And your absolute-position invariant ("multiple expansions and
contractions reach the same step count") is already guaranteed:
every move computes `targetSteps` from absolute θ, not deltas.

**So the entire problem reduces to two things:**
- Lift the `≤ 1.001` gate in `gotoExpansion`.
- Teach `findTheta` how to find θ for `targetEx < 1`.

---

## 2. The actual hard problem: what does the math say below 1.0?

`computeEx(Theta)` is **not** symmetric about `Theta = π/2`. Walking
through it:

| Theta input | Internal `theta = Theta − π/2` | sin(theta) | cos(theta) | Notes |
|---|---|---|---|---|
| 0.0001 | −π/2 + 0.0001 | ≈ −1 | ≈ 0 (positive) | current lower bracket — `Ex ≈ 1.0+` |
| 0.6 | −π/2 + 0.6 | ≈ −0.83 | ≈ 0.56 | current upper bracket — `Ex ≈ maxEx` |
| 0 (singular) | −π/2 | −1 | 0 | division blows up |
| −0.001 | −π/2 − 0.001 | ≈ −1 | ≈ 0 (negative) | `cos` flips sign |
| −0.6 | −π/2 − 0.6 | ≈ −0.83 | ≈ −0.56 | mirror of +0.6 across the singular line |

There are **three plausible kinematic stories** for `Ex < 1`, and only
you know which matches the physical rig:

**(A) The same equation extends through negative `Theta`.** If the
crank is symmetric, `computeEx(−Θ)` may produce `Ex < 1` for moderate
negative Θ. The fix is just to add a second search bracket like
`[−0.6, −0.0001]` and pick whichever bracket brackets the target.
No code change to `computeEx`.

**(B) Mirror symmetry: `Ex < 1` is solved using `|Ex − 1|`'s
positive-side θ, then negated.** Physically: contracting to 0.7×
uses the same mechanics as expanding to 1.3× but with the motor
spinning the other way. The fix is to compute θ for `2 − targetEx`
(or some chosen mirroring) and return `−θ`. No code change to
`computeEx`, but the meaning of `targetEx < 1` becomes a *convention*.

**(C) Different kinematic model entirely.** Some iris mechanisms
have a different link engagement below the resting position
(e.g., the crank pin hits a different cam). If that's your case, we
need a second `computeExCompression(Theta)` or a piecewise model.

**I can't tell which from the code alone — the geometry constants
don't carry that intent.** This is the first thing I need from you.

---

## 3. Notation clarification I need

You wrote:

> anything above goes clockwise (**1.35 CW**) and anything less goes
> counter clockwise (**1.24 CCW**)

The first example (1.35 CW) is clear: an `Ex` of 1.35 → CW rotation.
The second one (1.24 CCW) is ambiguous because 1.24 is **above** 1.0,
not below it. Three possible reads:

1. **Typo.** You meant `0.24` or `0.76` — i.e., `Ex` below 1.0 maps
   to CCW. Examples would be `Xgoto 0.8` (mild contraction, CCW) and
   `Xgoto 0.5` (strong contraction, CCW).
2. **"Magnitude below 1.0."** You mean a value that's `1.24` *below*
   the unit, i.e., `Ex = −0.24` or the absolute deviation is 0.24
   on the contraction side.
3. **Two ways to reach the same nominal expansion.** Some
   double-acting mechanisms can reach a given visible expansion via
   either CW or CCW rotation. `Xgoto 1.35` chooses CW; some hypothetical
   `Xgoto 1.35 ccw` or `Xgoto -1.35` chooses CCW. Unlikely but possible.

**Most likely**: read #1 (it matches "1x as the center"). I'll plan
assuming that unless you say otherwise.

---

## 4. Proposed implementation (assuming read #1 and kinematic story A or B)

### 4.1 `IrisGeometry.h` — new field

```cpp
struct IrisGeometry {
  // existing fields...
  float minEx = 0.5f;   // new — lower bound for Xgoto on the CCW side
};
```

`minEx` matches the existing `maxEx` convention. Default 0.5 is a
guess; pick whatever physically corresponds to the iris reaching
its mechanical contraction stop.

### 4.2 `IrisKinematics::findTheta` — wider/dual bracket

Two options:

**4.2a Single wider bracket** (if story A holds — same equation
straddles the singularity). Have `findTheta` automatically pick the
right bracket based on whether `targetEx > 1` or `< 1`:

```cpp
double IrisKinematics::findTheta(const IrisGeometry& geo, double targetEx) {
  if (targetEx > 1.0) {
    return findThetaInBracket(geo, targetEx, 0.0001, 0.6);   // existing
  } else if (targetEx < 1.0) {
    return findThetaInBracket(geo, targetEx, -0.6, -0.0001); // new
  } else {
    return 0.0;   // Ex = 1.0 → θ = 0 by definition
  }
}
```

Internals (the bisection+Newton body) move to a private helper
`findThetaInBracket`. Brackets are configurable via geometry if
labs need different ranges.

**4.2b Mirror approach** (if story B holds — physical symmetry).
For `targetEx < 1`, solve for the positive-side equivalent and
return its negation:

```cpp
if (targetEx < 1.0) {
  double mirroredTarget = 2.0 - targetEx;   // 0.8 → 1.2, 0.5 → 1.5
  double theta = findThetaInBracket(geo, mirroredTarget, 0.0001, 0.6);
  return -theta;
}
```

This is simpler but only correct if the rig actually has mirror
symmetry. **Story A is safer** — let the math tell us, don't
impose symmetry.

### 4.3 `IrisStretcher::gotoExpansion` — lift the gate

```cpp
bool gotoExpansion(double targetEx) {
  if (targetEx >= _geo.maxEx || targetEx <= _geo.minEx) {
    _io.println("Error: targetEx out of range");
    return false;
  }
  if (fabs(targetEx - 1.0) < 1e-6) {
    rotateThetaRadians(0.0);   // back to center
    return true;
  }
  const double angle = findTheta(targetEx);
  if (isnan(angle)) { _io.println("Error: no θ found"); return false; }
  // ... existing print + rotateThetaRadians(angle) ...
}
```

### 4.4 `IrisMenuUI` — extend the Xgoto edit range

`UiState::EDIT_XGOTO` currently clamps to `[1.001, maxEx]`. Change to
`[minEx, maxEx]` and tweak the default starting value (currently
1.500) if you want labs to land somewhere else.

The fine/coarse step sizes (0.001 / 0.050) and the LCD layout don't
need to change — they handle three decimal digits fine.

### 4.5 What does *not* change

- `IrisStepperDriver::rotateThetaRadians` — already bidirectional.
- The step counter — already signed and absolute. Going to `Ex=1.3`
  then `Ex=0.8` then `Ex=1.3` lands back on the **same** step count
  as the first 1.3 move, because every call computes `targetSteps`
  from absolute θ.
- The serial parser — `Xgoto 0.8` already tokenizes fine.

---

## 5. Open questions for you before I touch code

1. **Notation.** Confirm read #1 in §3: values below 1.0 → CCW
   contraction. Or explain what "1.24 CCW" actually means.
2. **Kinematic story.** A, B, or C in §2? If you don't know, the safest
   move is for me to make `findTheta` *try* both `[0.0001, 0.6]` and
   `[−0.6, −0.0001]` brackets and use whichever one brackets the
   target — then we measure on hardware whether the iris actually
   contracts and adjust.
3. **`minEx` value.** What's the smallest `Ex` the iris can physically
   reach without binding or damaging the petals? 0.5? 0.7? Or as
   low as 0? I'll default to 0.5 if you don't specify.
4. **Behavior at exactly `Ex = 1.0`.** Move to step 0 (back to
   center), or refuse? I propose: move to 0.
5. **Calibration.** The `Xcalibrate` routine currently does
   `+0.7 rad`, zero, `−0.287 rad`, zero. Should it now sweep both
   directions to verify CCW works? (Probably yes, but easy to add
   later.)

---

## 6. Hardware test plan once implemented

1. **Smoke test.** `XsetZero` at rest, then `Xgoto 1.2` → confirm CW
   motion and Ex lands. `Xgoto 1.0` → returns to center. Position
   counter should land back on 0.
2. **CCW first move.** `Xgoto 0.8` → confirm motor turns CCW and
   physical iris contracts. Position counter should be negative.
3. **Round-trip.** From step 2, `Xgoto 1.2` → should pass through
   center and arrive at the same position as in step 1.
4. **Repeatability.** `Xgoto 1.3` → `Xgoto 0.7` → `Xgoto 1.3` —
   the three positions for `Ex=1.3` should be byte-identical in step
   count (within rounding).
5. **Mechanical limits.** Try `Xgoto minEx` and `Xgoto maxEx`;
   confirm no binding, no stalling, motor doesn't lose steps.
6. **Edit screen.** From LCD encoder UI, scroll Xgoto down past
   1.000 into the contraction range. Confirm display + commit work.

---

## 7. File-by-file change summary

| File | Change |
|---|---|
| `src/IrisGeometry.h` | add `float minEx = 0.5f;` |
| `src/IrisKinematics.h` | adjust `findTheta` signature (optional ranges optional or always picks based on targetEx) |
| `src/IrisKinematics.cpp` | split body into `findThetaInBracket` private helper; pick bracket based on `targetEx > 1` vs `< 1`; handle `Ex == 1` as θ=0 |
| `src/IrisStretcher.cpp` | replace `targetEx <= 1.001 || targetEx >= _geo.maxEx` with `targetEx <= _geo.minEx || targetEx >= _geo.maxEx`; add the `Ex == 1` short-circuit |
| `src/IrisMenuUI.cpp` | change `goMin` in `UiState::EDIT_XGOTO` from `1.001f` to `_stretcher.geometry().minEx` |
| `README.md` | document the new bidirectional behavior under Serial commands |

No changes to: `IrisStepperDriver.*`, `IrisSerialConsole.*`,
`IrisStrainNAU7802.*`, internal helpers, example sketches.

---

## 8. Estimated effort

Once §5 questions are answered: ~30 minutes to code + a hardware
verification session for the test plan in §6.
