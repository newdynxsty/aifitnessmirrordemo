from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy as np
import tensorflow as tf
import tf_keras as keras


CORRECTION_MESSAGES = {
    "arms_not_high_enough": "Reach your arms higher overhead!",
    "legs_not_wide_enough": "Step your feet wider on each jack!",
    "not_low_enough": "Go lower and reach full depth!",
    "knees_touching_ground": "Get those knees off the ground!",
    "core_not_high_enough": "Sit up taller and lift your chest!",
    "good": "Great form. Keep it up!",
}


def correction_message_for_prediction(prediction: dict[str, Any] | None) -> str:
    if prediction is None:
        return "Move into frame so I can check your form."
    condition = prediction.get("condition", "")
    return CORRECTION_MESSAGES.get(condition, "Adjust your form and try again.")


def quantize_to_int8(value: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    quantized = np.round(value / scale + zero_point)
    return np.clip(quantized, -128, 127).astype(np.int8)


def dequantize_from_int8(value: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (value.astype(np.float32) - zero_point) * scale


def run_tflite_classifier(
    interpreter: tf.lite.Interpreter,
    input_detail: dict[str, Any],
    output_detail: dict[str, Any],
    X: np.ndarray,
) -> np.ndarray:
    probabilities = []
    in_scale, in_zero = input_detail["quantization"]
    out_scale, out_zero = output_detail["quantization"]

    for row in X:
        tensor = row.reshape(1, -1).astype(np.float32)
        if input_detail["dtype"] == np.int8:
            tensor = quantize_to_int8(tensor, in_scale, int(in_zero))
        interpreter.set_tensor(input_detail["index"], tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        if output_detail["dtype"] == np.int8:
            output = dequantize_from_int8(output, out_scale or (1.0 / 255.0), int(out_zero))
        probabilities.append(output[0])

    return np.asarray(probabilities, dtype=np.float32)


class ErrorClassifierRunner:
    def __init__(self, model_path: Path | str, labels_path: Path | str):
        self.model_path = Path(model_path)
        self.metadata = json.loads(Path(labels_path).read_text())
        self.class_infos = self.metadata["class_infos"]
        self.class_names = [info["canonical_label"] for info in self.class_infos]
        self.backend = ""
        self.model: Any = None
        self.interpreter: tf.lite.Interpreter | None = None
        self.input_detail: dict[str, Any] | None = None
        self.output_detail: dict[str, Any] | None = None

        if self.model_path.suffix == ".tflite":
            self.backend = "tflite"
            self.interpreter = tf.lite.Interpreter(model_path=str(self.model_path))
            self.interpreter.allocate_tensors()
            self.input_detail = self.interpreter.get_input_details()[0]
            self.output_detail = self.interpreter.get_output_details()[0]
        else:
            self.backend = "keras"
            self.model = keras.models.load_model(self.model_path)

    def predict_proba(self, feature: np.ndarray) -> np.ndarray:
        feature = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        if self.backend == "keras":
            return self.model.predict(feature, verbose=0)[0]

        assert self.interpreter is not None
        assert self.input_detail is not None
        assert self.output_detail is not None
        return run_tflite_classifier(self.interpreter, self.input_detail, self.output_detail, feature)[0]

    def describe_prediction(self, probabilities: np.ndarray) -> dict[str, Any]:
        index = int(np.argmax(probabilities))
        info = self.class_infos[index]
        return {
            "index": index,
            "confidence": float(probabilities[index]),
            "canonical_label": info["canonical_label"],
            "display_name": info["display_name"],
            "workout": info["workout"],
            "condition": info["condition"],
            "condition_display": info["condition_display"],
            "is_error": bool(info["is_error"]),
            "top3": [
                {
                    "canonical_label": self.class_infos[top_idx]["canonical_label"],
                    "display_name": self.class_infos[top_idx]["display_name"],
                    "confidence": float(probabilities[top_idx]),
                }
                for top_idx in np.argsort(probabilities)[::-1][:3]
            ],
        }
