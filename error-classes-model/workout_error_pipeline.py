from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tf_keras as keras
import tf_keras.src.backend as keras_backend
import tf_keras.src.utils.tf_utils as keras_tf_utils
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from ultralytics import YOLO

FEATURE_DIM = 51
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}
DEFAULT_DATASET_DIR = Path("Error Classes Dataset")
DEFAULT_OUTPUT_DIR = Path("artifacts/workout_error_classifier")
DEFAULT_CACHE_NAME = "pose_features.npz"
DEFAULT_DESKTOP_POSE_MODEL = "yolov8n-pose.pt"
BOARD_POSE_MODEL_HINT = Path(
    "a/en-us--M55M1_Series_BSP_CMSIS_V3.01.002/"
    "SampleCode/MachineLearning/PoseLandmark_YOLOv8n_workout_w_accel/Model/"
    "YOLOv8n-pose.tflite"
)

LEFT_RIGHT_KEYPOINT_PAIRS = [
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
]


class _Py312SafeRandom(random.Random):
    def randint(self, a, b):
        return super().randint(int(a), int(b))


def install_tfkeras_seed_workaround(seed: int) -> None:
    """Work around tf_keras using float bounds in randint() under Python 3.12."""
    keras.utils.set_random_seed(seed)
    keras_backend._SEED_GENERATOR.generator = _Py312SafeRandom(int(seed))
    keras_tf_utils.backend._SEED_GENERATOR.generator = keras_backend._SEED_GENERATOR.generator

    def _safe_get_random_seed() -> int:
        generator = getattr(keras_backend._SEED_GENERATOR, "generator", None)
        if generator is not None:
            return generator.randint(1, int(1e9))
        return random.randint(1, int(1e9))

    keras_tf_utils.get_random_seed = _safe_get_random_seed


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


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


@dataclass(frozen=True)
class ClassInfo:
    index: int
    canonical_label: str
    display_name: str
    workout: str
    workout_display: str
    condition: str
    condition_display: str
    is_error: bool


@dataclass(frozen=True)
class ClipInfo:
    clip_id: int
    video_path: str
    video_name: str
    raw_label: str
    canonical_label: str
    class_index: int
    workout: str
    condition: str
    is_error: bool


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def parse_folder_label(folder_name: str) -> dict[str, Any]:
    parts = [part.strip() for part in folder_name.split(" - ")]
    workout_display = parts[0]
    workout = slugify(workout_display)

    if len(parts) >= 3 and parts[1].lower() in {"left", "right"}:
        condition_display = " - ".join(parts[2:])
    elif len(parts) >= 2:
        condition_display = " - ".join(parts[1:])
    else:
        condition_display = "Good"

    condition = slugify(condition_display)
    canonical_label = f"{workout}__{condition}"
    display_name = f"{workout_display} - {condition_display}"

    return {
        "canonical_label": canonical_label,
        "display_name": display_name,
        "workout": workout,
        "workout_display": workout_display,
        "condition": condition,
        "condition_display": condition_display,
        "is_error": condition != "good",
    }


def collect_video_clips(dataset_dir: Path | str) -> tuple[list[ClipInfo], list[ClassInfo]]:
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    class_map: dict[str, dict[str, Any]] = {}
    raw_clips: list[dict[str, Any]] = []

    for folder in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        label_info = parse_folder_label(folder.name)
        class_map.setdefault(label_info["canonical_label"], label_info)
        video_files = sorted(
            p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        for video_path in video_files:
            raw_clips.append(
                {
                    "video_path": str(video_path),
                    "video_name": video_path.name,
                    "raw_label": folder.name,
                    **label_info,
                }
            )

    class_infos = [
        ClassInfo(index=index, **class_map[label])
        for index, label in enumerate(sorted(class_map.keys()))
    ]
    class_index_lookup = {info.canonical_label: info.index for info in class_infos}

    clips = [
        ClipInfo(
            clip_id=index,
            video_path=clip["video_path"],
            video_name=clip["video_name"],
            raw_label=clip["raw_label"],
            canonical_label=clip["canonical_label"],
            class_index=class_index_lookup[clip["canonical_label"]],
            workout=clip["workout"],
            condition=clip["condition"],
            is_error=clip["is_error"],
        )
        for index, clip in enumerate(raw_clips)
    ]

    return clips, class_infos


def resolve_desktop_pose_model(model_path: Path | str) -> str:
    model_path = Path(model_path)

    if model_path.exists():
        if model_path.suffix.lower() == ".tflite":
            raise ValueError(
                f"'{model_path}' is a TFLite model. The board copy at '{BOARD_POSE_MODEL_HINT}' "
                "is Ethos-U compiled and will not run in desktop Python. Use an uncompiled "
                "desktop model such as 'yolov8n-pose.pt' instead."
            )
        return str(model_path)

    if model_path.suffix.lower() == ".pt" and len(model_path.parts) == 1:
        return str(model_path)

    raise FileNotFoundError(
        "Could not find a desktop YOLO pose model at "
        f"{model_path}. Pass a local uncompiled model path, or use the Ultralytics model name "
        "'yolov8n-pose.pt' so it can be loaded or downloaded automatically."
    )


def choose_best_person(result: Any) -> int | None:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None or keypoints.data is None or len(keypoints.data) == 0:
        return None

    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xyxy is None or len(boxes.xyxy) == 0:
        return 0

    xyxy = boxes.xyxy.detach().cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    return int(np.argmax(areas))


def extract_pose_feature_from_result(result: Any, min_pose_confidence: float = 0.25) -> np.ndarray | None:
    person_index = choose_best_person(result)
    if person_index is None:
        return None

    keypoints = result.keypoints.data[person_index].detach().cpu().numpy()
    if keypoints.shape[0] != 17 or keypoints.shape[1] < 3:
        return None

    if float(np.mean(keypoints[:, 2])) < min_pose_confidence:
        return None

    boxes = getattr(result, "boxes", None)
    if boxes is not None and boxes.xyxy is not None and len(boxes.xyxy) > person_index:
        bbox = boxes.xyxy[person_index].detach().cpu().numpy()
        x1, y1, x2, y2 = [float(v) for v in bbox]
    else:
        visible = keypoints[:, 2] > 0.05
        if not np.any(visible):
            return None
        x1 = float(np.min(keypoints[visible, 0]))
        y1 = float(np.min(keypoints[visible, 1]))
        x2 = float(np.max(keypoints[visible, 0]))
        y2 = float(np.max(keypoints[visible, 1]))

    box_w = max(x2 - x1, 1.0)
    box_h = max(y2 - y1, 1.0)

    feature = np.zeros(FEATURE_DIM, dtype=np.float32)
    offset = 0
    for x, y, conf in keypoints:
        feature[offset] = np.clip((float(x) - x1) / box_w, 0.0, 1.0)
        feature[offset + 1] = np.clip((float(y) - y1) / box_h, 0.0, 1.0)
        feature[offset + 2] = np.clip(float(conf), 0.0, 1.0)
        offset += 3

    return feature


def mirror_pose_feature(feature: np.ndarray) -> np.ndarray:
    mirrored = feature.reshape(17, 3).copy()
    mirrored[:, 0] = 1.0 - mirrored[:, 0]
    for left_idx, right_idx in LEFT_RIGHT_KEYPOINT_PAIRS:
        mirrored[[left_idx, right_idx]] = mirrored[[right_idx, left_idx]]
    return mirrored.reshape(-1).astype(np.float32)


def jitter_pose_feature(feature: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    jittered = feature.reshape(17, 3).copy()
    jittered[:, :2] += rng.normal(0.0, sigma, size=(17, 2))
    jittered[:, :2] = np.clip(jittered[:, :2], 0.0, 1.0)
    return jittered.reshape(-1).astype(np.float32)


def iterate_sampled_frames(
    video_path: Path, sample_fps: float, max_frames_per_clip: int | None
) -> tuple[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1.0:
        fps = 30.0
    frame_step = max(int(round(fps / sample_fps)), 1)

    frame_index = 0
    yielded = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % frame_step == 0:
                yield frame_index, frame
                yielded += 1
                if max_frames_per_clip is not None and yielded >= max_frames_per_clip:
                    break
            frame_index += 1
    finally:
        cap.release()


def load_cached_dataset(cache_path: Path | str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache_path = Path(cache_path)
    metadata_path = cache_path.with_suffix(".json")
    with np.load(cache_path, allow_pickle=True) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int32)
        metadata = {
            "sample_clip_ids": data["sample_clip_ids"].astype(np.int32).tolist(),
            "sample_frame_indices": data["sample_frame_indices"].astype(np.int32).tolist(),
            "class_names": data["class_names"].tolist(),
        }
    metadata.update(json.loads(metadata_path.read_text()))
    return X, y, metadata


def build_pose_feature_dataset(
    dataset_dir: Path | str = DEFAULT_DATASET_DIR,
    pose_model_path: Path | str = DEFAULT_DESKTOP_POSE_MODEL,
    cache_path: Path | str = DEFAULT_OUTPUT_DIR / DEFAULT_CACHE_NAME,
    sample_fps: float = 3.0,
    max_frames_per_clip: int | None = 96,
    min_pose_confidence: float = 0.25,
    mirror_augmentation: bool = True,
    jitter_augmentations: int = 1,
    jitter_sigma: float = 0.01,
    reuse_cache: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if reuse_cache and cache_path.exists() and cache_path.with_suffix(".json").exists():
        print(f"Loading cached pose features from {cache_path}")
        return load_cached_dataset(cache_path)

    pose_model_source = resolve_desktop_pose_model(pose_model_path)
    clips, class_infos = collect_video_clips(dataset_dir)
    print(f"Found {len(clips)} labeled videos across {len(class_infos)} canonical classes.")
    print(f"Loading YOLO pose model from {pose_model_source}")
    pose_model = YOLO(pose_model_source)

    rng = np.random.default_rng(seed)
    X_samples: list[np.ndarray] = []
    y_samples: list[int] = []
    sample_clip_ids: list[int] = []
    sample_frame_indices: list[int] = []
    clip_summaries: list[dict[str, Any]] = []

    for clip in clips:
        video_path = Path(clip.video_path)
        print(f"Extracting pose features from {video_path.name} [{clip.canonical_label}]")
        clip_total = 0
        clip_kept = 0

        for frame_index, frame in iterate_sampled_frames(video_path, sample_fps, max_frames_per_clip):
            clip_total += 1
            result = pose_model(frame, verbose=False)[0]
            feature = extract_pose_feature_from_result(result, min_pose_confidence)
            if feature is None:
                continue

            X_samples.append(feature)
            y_samples.append(clip.class_index)
            sample_clip_ids.append(clip.clip_id)
            sample_frame_indices.append(frame_index)
            clip_kept += 1

            if mirror_augmentation:
                X_samples.append(mirror_pose_feature(feature))
                y_samples.append(clip.class_index)
                sample_clip_ids.append(clip.clip_id)
                sample_frame_indices.append(frame_index)

            for _ in range(jitter_augmentations):
                X_samples.append(jitter_pose_feature(feature, jitter_sigma, rng))
                y_samples.append(clip.class_index)
                sample_clip_ids.append(clip.clip_id)
                sample_frame_indices.append(frame_index)

        clip_summaries.append(
            {
                "clip_id": clip.clip_id,
                "video_name": clip.video_name,
                "canonical_label": clip.canonical_label,
                "sampled_frames": clip_total,
                "usable_frames": clip_kept,
            }
        )
        print(f"  kept {clip_kept} of {clip_total} sampled frames")

    if not X_samples:
        raise RuntimeError(
            "No pose features were extracted. Check that your pose model can detect people in these videos."
        )

    X = np.asarray(X_samples, dtype=np.float32)
    y = np.asarray(y_samples, dtype=np.int32)
    class_names = [info.canonical_label for info in class_infos]

    np.savez_compressed(
        cache_path,
        X=X,
        y=y,
        sample_clip_ids=np.asarray(sample_clip_ids, dtype=np.int32),
        sample_frame_indices=np.asarray(sample_frame_indices, dtype=np.int32),
        class_names=np.asarray(class_names, dtype=object),
    )

    metadata = {
        "dataset_dir": str(Path(dataset_dir).resolve()),
        "pose_model_path": pose_model_source,
        "feature_dim": FEATURE_DIM,
        "sample_fps": sample_fps,
        "max_frames_per_clip": max_frames_per_clip,
        "min_pose_confidence": min_pose_confidence,
        "mirror_augmentation": mirror_augmentation,
        "jitter_augmentations": jitter_augmentations,
        "jitter_sigma": jitter_sigma,
        "class_infos": [asdict(info) for info in class_infos],
        "clips": [asdict(clip) for clip in clips],
        "clip_summaries": clip_summaries,
        "sample_clip_ids": sample_clip_ids,
        "sample_frame_indices": sample_frame_indices,
        "class_names": class_names,
    }
    cache_path.with_suffix(".json").write_text(json.dumps(to_jsonable(metadata), indent=2))
    print(f"Saved pose feature cache to {cache_path}")

    return X, y, metadata


def split_clips_for_validation(
    clips: list[dict[str, Any]], val_fraction: float, seed: int
) -> tuple[set[int], set[int]]:
    clips_by_class: dict[int, list[int]] = defaultdict(list)
    for clip in clips:
        clips_by_class[int(clip["class_index"])].append(int(clip["clip_id"]))

    train_clip_ids: set[int] = set()
    val_clip_ids: set[int] = set()
    rng = random.Random(seed)

    for clip_ids in clips_by_class.values():
        clip_ids = clip_ids[:]
        rng.shuffle(clip_ids)
        if len(clip_ids) == 1:
            train_clip_ids.add(clip_ids[0])
            continue

        n_val = max(1, int(round(len(clip_ids) * val_fraction)))
        n_val = min(n_val, len(clip_ids) - 1)
        val_clip_ids.update(clip_ids[:n_val])
        train_clip_ids.update(clip_ids[n_val:])

    if not val_clip_ids:
        train_clip_ids = {int(clip["clip_id"]) for clip in clips}

    return train_clip_ids, val_clip_ids


def build_classifier(num_classes: int, input_dim: int = FEATURE_DIM) -> keras.Model:
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(128),
            keras.layers.ReLU(max_value=6.0),
            keras.layers.Dropout(0.20),
            keras.layers.Dense(64),
            keras.layers.ReLU(max_value=6.0),
            keras.layers.Dropout(0.10),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    return model


def make_representative_dataset(X_train: np.ndarray):
    def representative_dataset():
        for row in X_train:
            yield [row.reshape(1, -1).astype(np.float32)]

    return representative_dataset


def quantize_to_int8(value: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    quantized = np.round(value / scale + zero_point)
    return np.clip(quantized, -128, 127).astype(np.int8)


def dequantize_from_int8(value: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (value.astype(np.float32) - zero_point) * scale


def run_tflite_classifier(
    interpreter: tf.lite.Interpreter, input_detail: dict[str, Any], output_detail: dict[str, Any], X: np.ndarray
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


def plot_confusion_matrix_image(
    cm: np.ndarray, class_names: list[str], output_path: Path, title: str = "Validation Confusion Matrix"
) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.8), max(6, len(class_names) * 0.6)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if cm.size > 0:
        threshold = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > threshold else "black",
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def aggregate_clip_predictions(
    probabilities: np.ndarray, y_true: np.ndarray, clip_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    grouped_probs: dict[int, list[np.ndarray]] = defaultdict(list)
    grouped_true: dict[int, int] = {}

    for prob, label, clip_id in zip(probabilities, y_true, clip_ids):
        grouped_probs[int(clip_id)].append(prob)
        grouped_true[int(clip_id)] = int(label)

    clip_true = []
    clip_pred = []
    for clip_id in sorted(grouped_probs):
        mean_prob = np.mean(grouped_probs[clip_id], axis=0)
        clip_true.append(grouped_true[clip_id])
        clip_pred.append(int(np.argmax(mean_prob)))

    return np.asarray(clip_true, dtype=np.int32), np.asarray(clip_pred, dtype=np.int32)


def train_error_classifier(
    cache_path: Path | str = DEFAULT_OUTPUT_DIR / DEFAULT_CACHE_NAME,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    epochs: int = 150,
    batch_size: int = 64,
    val_fraction: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, metadata = load_cached_dataset(cache_path)
    class_infos = metadata["class_infos"]
    class_names = [info["canonical_label"] for info in class_infos]
    sample_clip_ids = np.asarray(metadata["sample_clip_ids"], dtype=np.int32)
    clips = metadata["clips"]

    train_clip_ids, val_clip_ids = split_clips_for_validation(clips, val_fraction, seed)
    train_mask = np.isin(sample_clip_ids, np.asarray(sorted(train_clip_ids), dtype=np.int32))
    val_mask = np.isin(sample_clip_ids, np.asarray(sorted(val_clip_ids), dtype=np.int32))

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    present_classes = np.unique(y_train)
    class_weights_raw = compute_class_weight("balanced", classes=present_classes, y=y_train)
    class_weight = {int(index): float(weight) for index, weight in zip(present_classes, class_weights_raw)}

    install_tfkeras_seed_workaround(seed)
    model = build_classifier(num_classes=len(class_names))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-5),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    keras_model_path = output_dir / "workout_error_classifier.keras"
    model.save(keras_model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_representative_dataset(X_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_bytes = converter.convert()
    tflite_path = output_dir / "workout_error_classifier_int8.tflite"
    tflite_path.write_bytes(tflite_bytes)

    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    val_probabilities = run_tflite_classifier(interpreter, input_detail, output_detail, X_val)
    val_predictions = np.argmax(val_probabilities, axis=1)
    report = classification_report(
        y_val,
        val_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_val, val_predictions, labels=list(range(len(class_names))))
    confusion_matrix_path = output_dir / "validation_confusion_matrix.png"
    plot_confusion_matrix_image(cm, class_names, confusion_matrix_path)

    clip_true, clip_pred = aggregate_clip_predictions(val_probabilities, y_val, sample_clip_ids[val_mask])
    clip_accuracy = float(np.mean(clip_true == clip_pred)) if len(clip_true) else 0.0

    history_path = output_dir / "training_history.json"
    history_path.write_text(json.dumps(to_jsonable(history.history), indent=2))

    labels_path = output_dir / "workout_error_labels.json"
    labels_payload = {
        "feature_dim": FEATURE_DIM,
        "class_infos": class_infos,
        "class_names": class_names,
        "keras_model_path": str(keras_model_path.resolve()),
        "tflite_model_path": str(tflite_path.resolve()),
        "source_cache_path": str(Path(cache_path).resolve()),
        "notes": [
            "Features are 17 YOLO pose keypoints in bbox-normalized x, y, confidence format.",
            "Canonical labels collapse left/right variants into one class.",
        ],
    }
    labels_path.write_text(json.dumps(to_jsonable(labels_payload), indent=2))

    summary = {
        "training_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "train_clip_ids": sorted(train_clip_ids),
        "validation_clip_ids": sorted(val_clip_ids),
        "class_distribution_train": dict(Counter(y_train.tolist())),
        "class_distribution_val": dict(Counter(y_val.tolist())),
        "frame_accuracy": float(np.mean(val_predictions == y_val)) if len(y_val) else 0.0,
        "clip_accuracy": clip_accuracy,
        "keras_model_path": str(keras_model_path.resolve()),
        "tflite_model_path": str(tflite_path.resolve()),
        "labels_path": str(labels_path.resolve()),
        "confusion_matrix_path": str(confusion_matrix_path.resolve()),
        "classification_report": report,
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2))

    csv_path = output_dir / "clip_summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["clip_id", "video_name", "canonical_label", "sampled_frames", "usable_frames"])
        for row in metadata["clip_summaries"]:
            writer.writerow(
                [
                    row["clip_id"],
                    row["video_name"],
                    row["canonical_label"],
                    row["sampled_frames"],
                    row["usable_frames"],
                ]
            )

    print(f"Saved Keras model to {keras_model_path}")
    print(f"Saved INT8 TFLite model to {tflite_path}")
    print(f"Saved label metadata to {labels_path}")
    print(f"Saved training summary to {summary_path}")
    return summary


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


def parse_training_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a workout error classifier from YOLO pose keypoints.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--pose-model", default=DEFAULT_DESKTOP_POSE_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--sample-fps", type=float, default=3.0)
    parser.add_argument("--max-frames-per-clip", type=int, default=96)
    parser.add_argument("--min-pose-confidence", type=float, default=0.25)
    parser.add_argument("--no-mirror-augmentation", action="store_true")
    parser.add_argument("--jitter-augmentations", type=int, default=1)
    parser.add_argument("--jitter-sigma", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-reextract", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_training_args()
    cache_path = args.cache_path or (args.output_dir / DEFAULT_CACHE_NAME)

    build_pose_feature_dataset(
        dataset_dir=args.dataset_dir,
        pose_model_path=args.pose_model,
        cache_path=cache_path,
        sample_fps=args.sample_fps,
        max_frames_per_clip=args.max_frames_per_clip,
        min_pose_confidence=args.min_pose_confidence,
        mirror_augmentation=not args.no_mirror_augmentation,
        jitter_augmentations=args.jitter_augmentations,
        jitter_sigma=args.jitter_sigma,
        reuse_cache=not args.force_reextract,
        seed=args.seed,
    )

    train_error_classifier(
        cache_path=cache_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
