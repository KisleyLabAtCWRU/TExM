// CustomGeometry — labs running a different mechanical iris.
//
// Demonstrates overriding IrisGeometry defaults for a rig with different
// link lengths, gear ratio, or microstepping. All math (computeEx,
// findTheta) automatically adapts.

#include <USB.h>
#include <KisleyIrisStretcher.h>

using namespace kisley::iris;

// Construct a custom geometry. Anything you don't override keeps its default.
IrisGeometry buildGeometry() {
  IrisGeometry g;
  g.r0            = 8.0f;    // longer crank
  g.rp            = 5.5f;
  g.Y0            = -6.4f;
  g.rPin          = 0.6f;
  g.g             = 20.0f;   // 20:1 gear ratio
  g.maxEx         = 5.0f;    // longer travel
  g.pulsesPerRev  = 3200;    // finer microstepping
  g.bladeRadiusCm = 12.5f;
  return g;
}

IrisStretcher     stretcher(/*STEP=*/12, /*DIR=*/13, buildGeometry());
IrisSerialConsole console(stretcher);

void setup() {
  delay(500);
  USB.begin();
  Serial.begin(115200);
  while (!Serial) {}

  stretcher.begin();
  console.begin();

  Serial.println();
  Serial.print("E0 = "); Serial.println(stretcher.geometry().E0(), 4);
  Serial.print("maxEx = "); Serial.println(stretcher.geometry().maxEx, 4);
}

void loop() {
  console.update();
}
