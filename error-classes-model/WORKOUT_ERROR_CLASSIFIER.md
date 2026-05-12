# Workout Error Classifier

This workspace now has a desktop training pipeline and demo for a third model that classifies workout form states and errors from YOLO pose keypoints.

## What It Trains

The pipeline:

- reads labeled workout videos from `Error Classes Dataset`
- runs YOLO pose on sampled video frames
- converts each detected pose into the same 51-value feature format used by `Rep_Count.ipynb`
- collapses left/right folder variants into canonical classes such as `push_ups__good` and `push_ups__knees_touching_ground`
- trains a compact MLP classifier
- exports both a `.keras` model and a fully quantized `int8` `.tflite` model

## Important Note About The Pose Model

The board copy of `YOLOv8n-pose.tflite` is Ethos-U compiled and does not run in desktop Python.

For desktop training and demo, use an uncompiled pose model such as `yolov8n-pose.pt`:

```bash
../.venv/bin/python workout_error_pipeline.py --pose-model yolov8n-pose.pt
```

That matches `Rep_Count.ipynb`: Ultralytics can load or auto-download `yolov8n-pose.pt` if it is not already present locally.
If you already have a local copy somewhere else, pass the full path with `--pose-model`.

## Train

```bash
../.venv/bin/python workout_error_pipeline.py \
  --dataset-dir "Error Classes Dataset" \
  --pose-model yolov8n-pose.pt \
  --output-dir artifacts/workout_error_classifier
```

Useful flags:

- `--force-reextract` rebuilds the pose-feature cache from videos
- `--sample-fps 3` changes how densely frames are sampled from each clip
- `--max-frames-per-clip 96` caps per-video extraction
- `--jitter-augmentations 1` adds light feature-space augmentation

Artifacts land in `artifacts/workout_error_classifier/`:

- `pose_features.npz`
- `workout_error_classifier.keras`
- `workout_error_classifier_int8.tflite`
- `workout_error_labels.json`
- `training_summary.json`
- `validation_confusion_matrix.png`

## Compile For The Board With Vela

Install Vela in the repo virtual environment if needed:

```bash
../.venv/bin/python -m pip install ethos-u-vela==4.0.0
```

Compile the fully quantized TFLite model:

```bash
../.venv/bin/vela \
  artifacts/workout_error_classifier/workout_error_classifier_int8.tflite \
  --accelerator-config ethos-u55-256 \
  --output-dir artifacts/workout_error_classifier
```

The board-ready file is:

```text
artifacts/workout_error_classifier/workout_error_classifier_int8_vela.tflite
```

Copy it into the repository SD-card bundle:

```bash
cp artifacts/workout_error_classifier/workout_error_classifier_int8_vela.tflite \
  ../sd_card_root/workout_error_classifier_int8_vela.tflite
```

The firmware expects this exact filename at the SD card root:

```text
0:\workout_error_classifier_int8_vela.tflite
```

## Desktop Demo

Webcam:

```bash
../.venv/bin/python demo_workout_error_classifier.py \
  --pose-model yolov8n-pose.pt \
  --classifier-model artifacts/workout_error_classifier/workout_error_classifier.keras \
  --labels artifacts/workout_error_classifier/workout_error_labels.json \
  --source 0
```

Local video:

```bash
../.venv/bin/python demo_workout_error_classifier.py \
  --pose-model yolov8n-pose.pt \
  --classifier-model artifacts/workout_error_classifier/workout_error_classifier.keras \
  --labels artifacts/workout_error_classifier/workout_error_labels.json \
  --source "Error Classes Dataset/Push Ups - Left - Knees Touching Ground/AustinBadPushup.MOV"
```

## Notebook

`Workout_Error_Classifier.ipynb` mirrors the same flow in notebook form and imports the shared training code from `workout_error_pipeline.py`.
