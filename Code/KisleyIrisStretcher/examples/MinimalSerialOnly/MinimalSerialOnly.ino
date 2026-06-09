// MinimalSerialOnly — headless iris stretcher.
//
// No LCD, no encoder, no buttons. Drive the stretcher entirely over
// USB serial using the X-prefixed commands. Useful for benchtop testing,
// scripted experiments, or rigs without a front panel.

#include <USB.h>
#include <KisleyIrisStretcher.h>

using namespace kisley::iris;

IrisStretcher     stretcher(/*STEP=*/12, /*DIR=*/13);
IrisSerialConsole console(stretcher);

void setup() {
  delay(500);
  USB.begin();
  Serial.begin(115200);
  while (!Serial) {}

  stretcher.begin();
  console.begin();
}

void loop() {
  console.update();
}
