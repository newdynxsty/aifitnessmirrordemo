# AI Fitness Mirror Firmware

Keil uVision firmware project for the Nuvoton M55 AI fitness mirror demo.

## Project

Open this file in Keil uVision:

```text
KEIL/PoseLandmark.uvprojx
```

The application loads three TensorFlow Lite models from the SD card:

- `YOLOv8n-pose.tflite`
- `rep_counter_int8_vela.tflite`
- `workout_error_classifier_int8_vela.tflite`

The repo root contains `sd_card_root/` with these files already collected.

## Build And Flash

1. Prepare a FAT32 SD card.
2. Copy the three `.tflite` files from repo-root `sd_card_root/` to the SD card root.
3. Insert the SD card into the board.
4. Open `KEIL/PoseLandmark.uvprojx` in Keil uVision.
5. Build with `Project > Build Target`.
6. Flash with `Flash > Download`.
7. Reset the board.

The demo expects the model files at `0:\` with the exact names above.

## Runtime

The display shows the camera feed, pose landmarks, rep/demo status, and compact error-classifier output:

- `ERR:<class>` - predicted workout/form class
- `EC:<confidence>` - error classifier confidence
