from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import cv2
import numpy as np
from ultralytics import YOLO

from workout_error_model import ErrorClassifierRunner, correction_message_for_prediction
from workout_error_pipeline import (
    DEFAULT_DESKTOP_POSE_MODEL,
    DEFAULT_OUTPUT_DIR,
    extract_pose_feature_from_result,
    resolve_desktop_pose_model,
)


def parse_source(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def should_enable_display() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return False
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def list_video_devices() -> list[str]:
    devices = []
    dev_dir = Path("/dev")
    if dev_dir.exists():
        for path in sorted(dev_dir.glob("video*")):
            devices.append(str(path))
    return devices


def build_camera_error(source: int | str) -> str:
    lines = [f"Could not open source: {source}"]
    devices = list_video_devices()
    if devices:
        lines.append(f"Detected video nodes: {', '.join(devices)}")

        inaccessible = [dev for dev in devices if not (os.access(dev, os.R_OK) and os.access(dev, os.W_OK))]
        if inaccessible:
            lines.append(
                "Current user does not have camera permission for: "
                + ", ".join(inaccessible)
            )
            lines.append("On WSL/Linux, add your user to the 'video' group and restart the shell:")
            lines.append("  sudo usermod -aG video $USER")

        lines.append("If one node is metadata-only, try another source such as /dev/video1.")
    else:
        lines.append("No /dev/video* nodes were detected.")

    lines.append("You can also run the demo against a local video file to verify the classifier path.")
    return "\n".join(lines)


def open_source_with_probe(source: int | str) -> tuple[cv2.VideoCapture, np.ndarray | None, str]:
    candidates: list[tuple[int | str, str]] = []

    if isinstance(source, int):
        candidates.append((source, f"camera index {source}"))
        linux_path = f"/dev/video{source}"
        if Path(linux_path).exists():
            candidates.insert(0, (linux_path, linux_path))
    else:
        candidates.append((source, str(source)))

    last_cap: cv2.VideoCapture | None = None
    for candidate, label in candidates:
        cap = cv2.VideoCapture(candidate, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(candidate)

        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        deadline = time.time() + 8.0
        first_frame = None
        while time.time() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                first_frame = frame
                return cap, first_frame, label
            time.sleep(0.1)

        last_cap = cap

    if last_cap is not None:
        last_cap.release()
    raise RuntimeError(build_camera_error(source))


def draw_text_block(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    x = 12
    y = 28
    for text, color in lines:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1, cv2.LINE_AA)
        y += 30


def draw_correction_banner(frame: np.ndarray, message: str, is_error: bool) -> None:
    height, width = frame.shape[:2]
    banner_height = max(96, height // 10)
    y1 = height - banner_height
    color = (0, 0, 210) if is_error else (0, 150, 0)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y1), (width, height), color, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    scale = max(0.9, min(1.8, width / 900))
    thickness = max(2, int(round(scale * 1.4)))
    x = 24
    y = y1 + banner_height // 2 + 14
    cv2.putText(frame, message, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, message, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the workout error classifier on webcam or a local video.")
    parser.add_argument("--pose-model", default=DEFAULT_DESKTOP_POSE_MODEL)
    parser.add_argument(
        "--classifier-model",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "workout_error_classifier.keras",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "workout_error_labels.json",
    )
    parser.add_argument("--source", default="0", help="Webcam index like 0 or a local video path.")
    parser.add_argument("--min-pose-confidence", type=float, default=0.25)
    parser.add_argument("--min-display-confidence", type=float, default=0.45)
    parser.add_argument("--smoothing", type=float, default=0.85)
    parser.add_argument("--classify-every", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None, help="Optional annotated output video path.")
    parser.add_argument("--display", action="store_true", help="Force on-screen display.")
    parser.add_argument("--no-display", action="store_true", help="Disable on-screen display.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.display and args.no_display:
        raise ValueError("Use only one of --display or --no-display.")

    pose_model_path = resolve_desktop_pose_model(args.pose_model)
    if not args.classifier_model.exists():
        raise FileNotFoundError(f"Could not find classifier model: {args.classifier_model}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Could not find label metadata: {args.labels}")

    print(f"Loading pose model from {pose_model_path}")
    pose_model = YOLO(pose_model_path)
    classifier = ErrorClassifierRunner(args.classifier_model, args.labels)

    parsed_source = parse_source(args.source)
    source_is_file = isinstance(parsed_source, str) and Path(parsed_source).exists()
    show_display = should_enable_display()
    if args.display:
        show_display = True
    if args.no_display:
        show_display = False

    output_path = args.output
    if output_path is None and not show_display and source_is_file:
        source_path = Path(parsed_source)
        output_path = DEFAULT_OUTPUT_DIR / f"{source_path.stem}.annotated.mp4"

    cap, first_frame, opened_label = open_source_with_probe(parsed_source)
    print(f"Opened source: {opened_label}")
    if not show_display:
        print("Running in headless mode.")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Annotated output will be written to {output_path}")

    smoothed_probabilities: np.ndarray | None = None
    frame_index = 0
    writer: cv2.VideoWriter | None = None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1.0:
        fps = 30.0

    try:
        while True:
            if first_frame is not None:
                frame = first_frame
                ok = True
                first_frame = None
            else:
                ok, frame = cap.read()
                if not ok:
                    break

            prediction_summary = None
            result = None

            if frame_index % max(args.classify_every, 1) == 0:
                result = pose_model(frame, verbose=False)[0]
                feature = extract_pose_feature_from_result(result, args.min_pose_confidence)
                if feature is not None:
                    probabilities = classifier.predict_proba(feature)
                    if smoothed_probabilities is None:
                        smoothed_probabilities = probabilities
                    else:
                        alpha = float(np.clip(args.smoothing, 0.0, 0.99))
                        smoothed_probabilities = alpha * smoothed_probabilities + (1.0 - alpha) * probabilities
                    prediction_summary = classifier.describe_prediction(smoothed_probabilities)
                else:
                    smoothed_probabilities = None

            annotated = result.plot() if result is not None else frame.copy()
            lines: list[tuple[str, tuple[int, int, int]]] = [("Press Q to quit", (255, 255, 255))]

            if prediction_summary is None:
                lines.append(("No reliable pose detected", (0, 200, 255)))
            else:
                is_confident = prediction_summary["confidence"] >= args.min_display_confidence
                status_color = (0, 220, 0) if not prediction_summary["is_error"] else (0, 0, 255)
                if not is_confident:
                    status_color = (0, 200, 255)

                prefix = "ERROR" if prediction_summary["is_error"] else "GOOD FORM"
                lines.append(
                    (
                        f"{prefix}: {prediction_summary['display_name']} ({prediction_summary['confidence']:.2f})",
                        status_color,
                    )
                )
                correction = correction_message_for_prediction(prediction_summary)
                lines.append((correction, status_color))
                for item in prediction_summary["top3"]:
                    lines.append((f"alt: {item['display_name']} ({item['confidence']:.2f})", (255, 255, 255)))

            draw_text_block(annotated, lines)
            if prediction_summary is not None:
                draw_correction_banner(
                    annotated,
                    correction_message_for_prediction(prediction_summary),
                    bool(prediction_summary["is_error"]),
                )
            if writer is None and output_path is not None:
                height, width = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            if writer is not None:
                writer.write(annotated)

            if show_display:
                cv2.imshow("Workout Error Classifier Demo", annotated)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            elif prediction_summary is not None and frame_index % max(args.classify_every * 15, 15) == 0:
                print(
                    f"frame={frame_index} label={prediction_summary['display_name']} "
                    f"confidence={prediction_summary['confidence']:.2f}"
                )

            frame_index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show_display:
            cv2.destroyAllWindows()

    if output_path is not None:
        print(f"Saved annotated demo to {output_path}")


if __name__ == "__main__":
    main()
