# SD Card Root

Copy the three `.tflite` files in this folder directly to the root of the demo SD card:

- `YOLOv8n-pose.tflite`
- `rep_counter_int8_vela.tflite`
- `workout_error_classifier_int8_vela.tflite`

The firmware looks for these exact filenames at `0:\`. Do not rename them and do not place them in a subfolder.

`rep_counter_int8_vela.tflite` is the current rep-counter model with squat support. `rep_counter_int8_vela_OLD_nosquats.tflite` is an archived copy of the previous model and is not loaded by the firmware.
