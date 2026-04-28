# AI Fitness Mirror Demo

Firmware and model bundle for the Nuvoton M55 AI fitness mirror demo. The demo runs YOLOv8n pose detection on the camera stream, feeds pose keypoints into a rep counter and a workout-form error classifier, and draws the result on the display.

## Repository Layout

- `keil_firmware/` - Nuvoton M55 BSP plus the Keil uVision project for the demo.
- `keil_firmware/SampleCode/MachineLearning/AIFitnessMirror/` - application source code.
- `sd_card_root/` - copy these files directly to the root of the board SD card.
- `error-classes-model/` - training/evaluation scripts, notebooks, dataset, and model artifacts for the workout error classifier.

## Hardware

- Nuvoton M55M1 board with camera/display setup used by the AI fitness mirror demo
- SD card formatted as FAT32
- USB cable for power/debug/flashing
- Windows machine with Keil uVision

## Required SD Card Files

The firmware opens these files from the SD card root, using names hardcoded in `main.cpp`:

| SD card filename | Purpose |
| --- | --- |
| `YOLOv8n-pose.tflite` | Vela-compiled pose detector |
| `rep_counter_int8_vela.tflite` | Vela-compiled rep counter |
| `workout_error_classifier_int8_vela.tflite` | Vela-compiled form/error classifier |

The repo includes a ready-to-copy folder:

```text
sd_card_root/
├── YOLOv8n-pose.tflite
├── rep_counter_int8_vela.tflite
└── workout_error_classifier_int8_vela.tflite
```

Copy all three files from `sd_card_root/` onto the root of the SD card. Do not put them inside a subfolder.

## Build And Flash Firmware

1. Install Keil uVision 5 and the Nuvoton M55M1 device support pack.
2. Connect the board over USB/debug and insert the prepared SD card.
3. Open this project in Keil:

```text
keil_firmware/SampleCode/MachineLearning/AIFitnessMirror/KEIL/PoseLandmark.uvprojx
```

4. In Keil, select the `PoseLandmark` target if it is not already selected.
5. Build the project with `Project > Build Target`.
6. Flash the board with `Flash > Download`.
7. Reset the board.

If Keil reports that a model file cannot be prepared, re-check the SD card root filenames exactly. The firmware expects drive `0:\` and the filenames listed above.

## Running The Demo

1. Power the board with the SD card inserted.
2. Stand in view of the camera.
3. The display should show the camera feed with pose landmarks, rep/demo status, and the error-class model proof-of-concept output.
4. Current error-class display labels use compact text:
   - `ERR:<class>` for the predicted form state
   - `EC:<confidence>` for the model confidence

Example classes include `JJ ARM LOW`, `JJ LEG NAR`, `PUSH KNEE`, `SIT CORE`, and `SQUAT LOW`.

NOTE: Most recent firmware version may have slightly different labels. 

## Firmware Model Integration

The model file paths are defined in:

```text
keil_firmware/SampleCode/MachineLearning/AIFitnessMirror/main.cpp
```

The error-class model wrapper is:

```text
keil_firmware/SampleCode/MachineLearning/AIFitnessMirror/Model/include/ErrorClassModel.hpp
keil_firmware/SampleCode/MachineLearning/AIFitnessMirror/KEIL/ErrorClassModel.cpp
```

The error classifier consumes the same 51-value pose feature vector as the rep counter: 17 pose keypoints, each represented as normalized `x`, normalized `y`, and confidence.

