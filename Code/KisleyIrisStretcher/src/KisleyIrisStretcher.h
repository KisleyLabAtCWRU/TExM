#pragma once
// Umbrella header — single include for sketches.
//
//   #include <KisleyIrisStretcher.h>
//   using namespace kisley::iris;
//
// Pulls in geometry, kinematics, motion, LCD/encoder UI, and the serial
// console. The optional NAU7802 strain adapter must be included separately:
//
//   #include <IrisStrainNAU7802.h>
//
// because not all installations want the Adafruit_NAU7802 dependency.

#include "IrisGeometry.h"
#include "IrisKinematics.h"
#include "IrisStepperDriver.h"
#include "IrisStretcher.h"
#include "IrisMenuUI.h"
#include "IrisSerialConsole.h"
#include "IrisStrainArray.h"
#include "IrisExperiment.h"

#define KISLEY_IRIS_STRETCHER_VERSION "2.0.0"
