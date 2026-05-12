# Rep Counter Model

This folder contains the training notebook, source image dataset, and TensorFlow Lite artifacts for the AI Fitness Mirror rep-counter model.

The model is not a direct counter. It classifies the current pose phase from YOLOv8n pose keypoints, and the firmware turns phase transitions into reps. In the current firmware, jumping jacks, sit ups, and squats are wired into the rep-counting state machine.

## Current Status

The latest deployed model includes squat support. `sd_card_root/rep_counter_int8_vela.tflite` matches `rep-counter-model/rep_counter_int8_vela.tflite` and is the model loaded by firmware from `0:\rep_counter_int8_vela.tflite`.

`sd_card_root/rep_counter_int8_vela_OLD_nosquats.tflite` is an archived copy of the earlier model without squat outputs. Do not copy it onto the SD card for the current demo.

## Inputs And Outputs

Input tensor:

- Shape: `(1, 51)`
- Type: INT8 after quantization
- Features: 17 YOLO pose keypoints flattened as `x`, `y`, `confidence`

The training notebook extracts keypoints from images with `yolov8n-pose.pt` and normalizes `x` and `y` by image width and height. The current firmware builds a 51-value vector from the live YOLO model, normalizes coordinates relative to the detected pose box, and then quantizes it before inference. Keep those preprocessing paths aligned when retraining or changing firmware behavior.

Output tensor:

- Shape: `(1, 8)`
- Type: INT8 softmax probabilities
- Class order:

| Index | Label | Firmware use |
| ---: | --- | --- |
| 0 | `jump_middle` | Jumping jack active phase |
| 1 | `lunges_middle` | Model output only |
| 2 | `pushup_middle` | Model output only |
| 3 | `pushup_start` | Model output only |
| 4 | `situp_middle` | Sit-up active phase |
| 5 | `situp_start` | Sit-up reset/count phase |
| 6 | `squat_middle` | Squat active phase |
| 7 | `squat_start` | Squat reset/count phase |

The output order is part of the firmware contract. If the label list changes in `Rep_Count.ipynb`, the index mapping in `keil_firmware/SampleCode/MachineLearning/AIFitnessMirror/main.cpp` must be updated at the same time.

## Dataset Layout

Training images are organized by class under `images/`:

```text
images/
|-- jump_middle/
|-- lunges_middle/
|-- pushup_middle/
|-- pushup_start/
|-- situp_middle/
|-- situp_start/
|-- squat_middle/
`-- squat_start/
```

The current notebook also leaves a `lunges_right/` image folder in the tree, but it is not included in the training class list.

## Training Flow

1. Open `Rep_Count.ipynb`.
2. Load `yolov8n-pose.pt`.
3. Extract 17-keypoint pose vectors from each class folder.
4. Train the Keras MLP:

```text
Input(1, 51) -> Dense(128) -> ReLU6 -> Dense(64) -> ReLU6 -> Dense(8) -> Softmax
```

5. Export the fully INT8 model as `rep_counter_int8.tflite`.
6. Run `velacompiler-rep.ipynb` or the equivalent Vela command to produce `rep_counter_int8_vela.tflite`.

Equivalent Vela command:

```bash
vela rep_counter_int8.tflite \
  --accelerator-config ethos-u55-256 \
  --output-dir .
```

## Deployment

After retraining and compiling with Vela, copy the deployed artifact to the SD-card bundle:

```bash
cp rep-counter-model/rep_counter_int8_vela.tflite \
  sd_card_root/rep_counter_int8_vela.tflite
```

Then copy the three current SD-card models to the root of the board SD card:

- `YOLOv8n-pose.tflite`
- `rep_counter_int8_vela.tflite`
- `workout_error_classifier_int8_vela.tflite`

## Firmware Behavior

The current firmware uses a confidence threshold of `0.70` before updating the active exercise or rep-counting state.

Squats are counted by the transition from `squat_middle` to `squat_start`. Jumping jacks are counted by returning from `jump_middle` to `squat_start`, because `squat_start` is currently used as the shared standing/rest phase. Sit ups are counted by returning from `situp_middle` to `situp_start`.

Push-up and lunge labels exist in the model output, but the current firmware does not yet count push-up or lunge reps.
