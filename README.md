# AI Fitness Mirror Demo

Firmware and model bundle for the Nuvoton M55 AI fitness mirror demo. The demo runs YOLOv8n pose detection on the camera stream, feeds pose keypoints into a rep counter and a workout-form error classifier, and draws the result on the display. **The current firmware counts jumping jacks, sit ups, and squats. Push ups and lunges are still in development.**

## Repository Layout

- `keil_firmware/` - Nuvoton M55 BSP plus the Keil uVision project for the demo.
- `keil_firmware/SampleCode/MachineLearning/AIFitnessMirror/` - application source code.
- `sd_card_root/` - copy these files directly to the root of the board SD card.
- `rep-counter-model/` - training notebooks, image datasets, and TFLite artifacts for the pose-phase rep counter.
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
├── rep_counter_int8_vela_OLD_nosquats.tflite
└── workout_error_classifier_int8_vela.tflite
```

Copy the three current model files from `sd_card_root/` onto the root of the SD card. Do not put them inside a subfolder. `rep_counter_int8_vela_OLD_nosquats.tflite` is kept only as an archive of the previous rep-counter model and should not be copied for the current demo.

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

The active rep counter currently displays `JUMPING JACK`, `SIT-UP`, or `SQUAT` once it sees the corresponding pose phase with enough confidence. Example error-class outputs include `JJ ARM LOW`, `JJ LEG NAR`, `PUSH KNEE`, `SIT CORE`, and `SQUAT LOW`.

NOTE: Most recent firmware version may have slightly different labels. 

## Training The Rep Counter

The rep-counter training assets live in:

```text
rep-counter-model/
```

The current model is an INT8 MLP that consumes 51 pose features: 17 YOLO pose keypoints, each represented as normalized `x`, normalized `y`, and confidence. It outputs these pose-phase classes in the order expected by `main.cpp`:

| Index | Label |
| ---: | --- |
| 0 | `jump_middle` |
| 1 | `lunges_middle` |
| 2 | `pushup_middle` |
| 3 | `pushup_start` |
| 4 | `situp_middle` |
| 5 | `situp_start` |
| 6 | `squat_middle` |
| 7 | `squat_start` |

See `rep-counter-model/README.md` for the full rep-counter workflow.

## Training The Error Classifier

The workout error classifier is trained from labeled videos in:

```text
error-classes-model/Error Classes Dataset/
```

Each subfolder name is treated as the label. The pipeline collapses left/right variants into one canonical class, so folders like `Push Ups - Left - Knees Touching Ground` and `Push Ups - Right - Knees Touching Ground` become the same model class.

### 1. Create A Python Environment

From the repo root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  tensorflow tf-keras ultralytics opencv-python scikit-learn matplotlib numpy \
  ethos-u-vela==4.0.0
```

If the machine is offline, install these packages ahead of time and place a local desktop YOLO pose model at a known path. The training pipeline uses `yolov8n-pose.pt` for desktop feature extraction. Do not use the board `YOLOv8n-pose.tflite` for desktop training; that file is Vela/Ethos-U compiled.

### 2. Train And Export INT8 TFLite

Run this from `error-classes-model/`:

```bash
cd error-classes-model

../.venv/bin/python workout_error_pipeline.py \
  --dataset-dir "Error Classes Dataset" \
  --pose-model yolov8n-pose.pt \
  --output-dir artifacts/workout_error_classifier
```

Useful flags:

- `--force-reextract` rebuilds pose features from videos instead of using the cache.
- `--sample-fps 3` controls how many frames are sampled per second.
- `--max-frames-per-clip 96` caps extraction per video.
- `--jitter-augmentations 1` controls light feature-space augmentation.

The script produces:

```text
error-classes-model/artifacts/workout_error_classifier/
├── workout_error_classifier.keras
├── workout_error_classifier_int8.tflite
├── workout_error_labels.json
├── training_summary.json
└── validation_confusion_matrix.png
```

### 3. Compile The Error Classifier With Vela

From `error-classes-model/`:

```bash
../.venv/bin/vela \
  artifacts/workout_error_classifier/workout_error_classifier_int8.tflite \
  --accelerator-config ethos-u55-256 \
  --output-dir artifacts/workout_error_classifier
```

This creates:

```text
error-classes-model/artifacts/workout_error_classifier/workout_error_classifier_int8_vela.tflite
```

### 4. Update The SD Card Payload

Copy the newly compiled model into the repo SD-card bundle:

```bash
cp artifacts/workout_error_classifier/workout_error_classifier_int8_vela.tflite \
  ../sd_card_root/workout_error_classifier_int8_vela.tflite
```

Then copy all three files from `sd_card_root/` to the SD card root before running the board demo.

### 5. Optional Desktop Test

Before deploying to the board, run the desktop demo against a local video:

```bash
cd error-classes-model

../.venv/bin/python demo_workout_error_classifier.py \
  --pose-model yolov8n-pose.pt \
  --classifier-model artifacts/workout_error_classifier/workout_error_classifier.keras \
  --labels artifacts/workout_error_classifier/workout_error_labels.json \
  --source "Error Classes Dataset/Push Ups - Left - Knees Touching Ground/AustinBadPushup.MOV"
```

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
