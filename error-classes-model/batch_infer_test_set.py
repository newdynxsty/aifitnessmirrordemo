from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from workout_error_pipeline import (
    DEFAULT_DESKTOP_POSE_MODEL,
    DEFAULT_OUTPUT_DIR,
    ErrorClassifierRunner,
    correction_message_for_prediction,
    extract_pose_feature_from_result,
    resolve_desktop_pose_model,
    to_jsonable,
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}
DEFAULT_TEST_DIR = Path("test-set")
DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "test_set_inference"


def find_videos(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS])


def draw_large_text(
    frame,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    scale: float,
    thickness: int = 2,
) -> None:
    x, y = origin
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_overlay(frame: np.ndarray, prediction: dict | None) -> None:
    height, width = frame.shape[:2]
    top_height = max(150, height // 6)
    bottom_height = max(96, height // 10)

    if prediction is None:
        status_color = (0, 200, 255)
        label_text = "No reliable pose"
        confidence_text = "Confidence: --"
        message = correction_message_for_prediction(None)
        is_error = True
    else:
        is_error = bool(prediction["is_error"])
        status_color = (0, 0, 220) if is_error else (0, 180, 0)
        prefix = "ERROR" if is_error else "GOOD FORM"
        label_text = f"{prefix}: {prediction['display_name']}"
        confidence_text = f"Confidence: {prediction['confidence']:.2f}"
        message = correction_message_for_prediction(prediction)

    top = frame.copy()
    cv2.rectangle(top, (0, 0), (width, top_height), (18, 18, 18), -1)
    cv2.addWeighted(top, 0.62, frame, 0.38, 0, frame)

    text_scale = max(0.9, min(1.8, width / 900))
    sub_scale = max(0.7, text_scale * 0.72)
    draw_large_text(frame, label_text, (24, 56), status_color, text_scale, thickness=2)
    draw_large_text(frame, confidence_text, (24, 108), (245, 245, 245), sub_scale, thickness=1)

    bottom = frame.copy()
    y1 = height - bottom_height
    cv2.rectangle(bottom, (0, y1), (width, height), status_color, -1)
    cv2.addWeighted(bottom, 0.72, frame, 0.28, 0, frame)
    draw_large_text(frame, message, (24, y1 + bottom_height // 2 + 14), (255, 255, 255), text_scale, thickness=2)


def run_video(
    video_path: Path,
    output_path: Path,
    pose_model: YOLO,
    classifier: ErrorClassifierRunner,
    min_pose_confidence: float,
    smoothing: float,
    classify_every: int,
    max_seconds: float | None,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1.0:
        fps = 30.0
    max_frames = int(round(fps * max_seconds)) if max_seconds else None

    writer: cv2.VideoWriter | None = None
    frame_index = 0
    smoothed_probabilities: np.ndarray | None = None
    last_prediction: dict | None = None
    predictions: list[dict] = []

    try:
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break

            ok, frame = cap.read()
            if not ok:
                break

            result = None
            if frame_index % max(classify_every, 1) == 0:
                result = pose_model(frame, verbose=False)[0]
                feature = extract_pose_feature_from_result(result, min_pose_confidence)
                if feature is not None:
                    probabilities = classifier.predict_proba(feature)
                    if smoothed_probabilities is None:
                        smoothed_probabilities = probabilities
                    else:
                        alpha = float(np.clip(smoothing, 0.0, 0.99))
                        smoothed_probabilities = alpha * smoothed_probabilities + (1.0 - alpha) * probabilities
                    last_prediction = classifier.describe_prediction(smoothed_probabilities)
                    predictions.append(last_prediction)

            annotated = result.plot() if result is not None else frame.copy()
            draw_overlay(annotated, last_prediction)

            if writer is None:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                height, width = annotated.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
            writer.write(annotated)
            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if predictions:
        label_counts = Counter(prediction["display_name"] for prediction in predictions)
        top_label, top_count = label_counts.most_common(1)[0]
        top_predictions = [prediction for prediction in predictions if prediction["display_name"] == top_label]
        mean_confidence = float(np.mean([prediction["confidence"] for prediction in top_predictions]))
        final_prediction = top_predictions[-1]
    else:
        top_label = "No reliable pose"
        top_count = 0
        mean_confidence = 0.0
        final_prediction = None

    return {
        "input": str(video_path),
        "output": str(output_path),
        "frames_written": frame_index,
        "predicted_label": top_label,
        "prediction_count": top_count,
        "mean_confidence": mean_confidence,
        "correction_message": correction_message_for_prediction(final_prediction),
        "is_error": bool(final_prediction["is_error"]) if final_prediction else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference over every video in the test-set folder.")
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR)
    parser.add_argument("--pose-model", default=DEFAULT_DESKTOP_POSE_MODEL)
    parser.add_argument(
        "--classifier-model",
        type=Path,
        default=DEFAULT_OUTPUT_DIR.parent / "workout_error_classifier.keras",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_OUTPUT_DIR.parent / "workout_error_labels.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-pose-confidence", type=float, default=0.25)
    parser.add_argument("--smoothing", type=float, default=0.85)
    parser.add_argument("--classify-every", type=int, default=2)
    parser.add_argument("--max-seconds", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    videos = find_videos(args.test_dir)
    if not videos:
        raise RuntimeError(f"No videos found under {args.test_dir}")
    if not args.classifier_model.exists():
        raise FileNotFoundError(f"Could not find classifier model: {args.classifier_model}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Could not find label metadata: {args.labels}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pose_model = YOLO(resolve_desktop_pose_model(args.pose_model))
    classifier = ErrorClassifierRunner(args.classifier_model, args.labels)

    rows = []
    for video in videos:
        output_name = f"{video.stem}.annotated.mp4"
        output_path = args.output_dir / output_name
        print(f"Running inference on {video.name}...")
        row = run_video(
            video,
            output_path,
            pose_model,
            classifier,
            args.min_pose_confidence,
            args.smoothing,
            args.classify_every,
            args.max_seconds,
        )
        rows.append(row)
        print(f"  {row['predicted_label']} ({row['mean_confidence']:.2f}) -> {output_path}")

    summary_json = args.output_dir / "test_set_predictions.json"
    summary_csv = args.output_dir / "test_set_predictions.csv"
    summary_json.write_text(json.dumps(to_jsonable(rows), indent=2))

    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "input",
                "output",
                "frames_written",
                "predicted_label",
                "prediction_count",
                "mean_confidence",
                "correction_message",
                "is_error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved JSON summary to {summary_json}")
    print(f"Saved CSV summary to {summary_csv}")


if __name__ == "__main__":
    main()
