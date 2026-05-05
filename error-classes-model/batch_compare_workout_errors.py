from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from workout_error_model import ErrorClassifierRunner
from workout_error_pipeline import (
    DEFAULT_DESKTOP_POSE_MODEL,
    DEFAULT_OUTPUT_DIR,
    extract_pose_feature_from_result,
    resolve_desktop_pose_model,
)


DATASET_DIR = Path("Error Classes Dataset")
DEFAULT_OUTPUT_SUBDIR = DEFAULT_OUTPUT_DIR / "comparisons"
TARGET_PANEL_WIDTH = 720
HEADER_HEIGHT = 210
BOTTOM_BAR_HEIGHT = 90
CLASSIFY_EVERY = 2


@dataclass(frozen=True)
class ComparisonJob:
    workout: str
    workout_display: str
    side_tag: str
    side_display: str
    error_folder: Path
    good_folder: Path
    error_display: str
    good_display: str
    output_slug: str


def slugify(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def parse_folder(folder_name: str) -> dict[str, str]:
    parts = [part.strip() for part in folder_name.split(" - ")]
    workout_display = parts[0]
    workout = slugify(workout_display)

    side_tag = "default"
    side_display = ""
    if len(parts) >= 3 and parts[1].lower() in {"left", "right"}:
        side_tag = parts[1].lower()
        side_display = parts[1].title()
        condition_display = " - ".join(parts[2:])
    elif len(parts) >= 2:
        condition_display = " - ".join(parts[1:])
    else:
        condition_display = "Good"

    condition = slugify(condition_display)
    return {
        "workout": workout,
        "workout_display": workout_display,
        "side_tag": side_tag,
        "side_display": side_display,
        "condition": condition,
        "condition_display": condition_display,
    }


def collect_jobs(dataset_dir: Path) -> list[ComparisonJob]:
    folders = [p for p in sorted(dataset_dir.iterdir()) if p.is_dir()]
    parsed = {folder: parse_folder(folder.name) for folder in folders}
    jobs: list[ComparisonJob] = []

    for folder, info in parsed.items():
        if info["condition"] == "good":
            continue

        candidates = [
            other
            for other, other_info in parsed.items()
            if other_info["workout"] == info["workout"]
            and other_info["condition"] == "good"
            and other_info["side_tag"] == info["side_tag"]
        ]
        if not candidates:
            candidates = [
                other
                for other, other_info in parsed.items()
                if other_info["workout"] == info["workout"] and other_info["condition"] == "good"
            ]
        if not candidates:
            continue

        good_folder = candidates[0]
        side_suffix = f"_{info['side_tag']}" if info["side_tag"] != "default" else ""
        output_slug = f"{info['workout']}{side_suffix}__{info['condition']}"
        jobs.append(
            ComparisonJob(
                workout=info["workout"],
                workout_display=info["workout_display"],
                side_tag=info["side_tag"],
                side_display=info["side_display"],
                error_folder=folder,
                good_folder=good_folder,
                error_display=info["condition_display"],
                good_display="Good",
                output_slug=output_slug,
            )
        )

    return jobs


def pick_matching_clip(folder: Path, preferred_token: str | None = None) -> Path:
    files = sorted([p for p in folder.iterdir() if p.is_file()])
    if preferred_token:
        preferred_token = preferred_token.lower()
        for file in files:
            if preferred_token in file.stem.lower():
                return file
    return files[0]


def build_clip_pair(job: ComparisonJob) -> tuple[Path, Path]:
    error_files = sorted([p for p in job.error_folder.iterdir() if p.is_file()])
    error_clip = error_files[0]

    preferred_token = None
    for token in ["austin", "justin", "jiesheng"]:
        if token in error_clip.stem.lower():
            preferred_token = token
            break

    good_clip = pick_matching_clip(job.good_folder, preferred_token)
    return good_clip, error_clip


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def fit_panel(frame: np.ndarray, width: int) -> np.ndarray:
    height, current_width = frame.shape[:2]
    scale = width / current_width
    target_height = int(round(height * scale))
    return cv2.resize(frame, (width, target_height), interpolation=cv2.INTER_AREA)


def draw_big_label(
    image: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int], float]],
    x: int,
    y: int,
    line_gap: int = 14,
) -> None:
    current_y = y
    for text, color, scale in lines:
        thickness = max(1, int(round(scale * 1.25)))
        cv2.putText(image, text, (x, current_y), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(image, text, (x, current_y), cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)
        text_height = int(38 * scale)
        current_y += text_height + line_gap


def color_for_prediction(is_error: bool, confidence: float) -> tuple[int, int, int]:
    if confidence < 0.45:
        return (0, 200, 255)
    return (0, 0, 255) if is_error else (0, 220, 0)


def render_header(width: int, title: str, subtitle: str) -> np.ndarray:
    header = np.full((HEADER_HEIGHT, width, 3), 250, dtype=np.uint8)
    cv2.rectangle(header, (0, 0), (width - 1, HEADER_HEIGHT - 1), (40, 40, 40), 3)
    draw_big_label(
        header,
        [
            (title, (10, 10, 10), 1.3),
            (subtitle, (60, 60, 60), 0.9),
        ],
        x=24,
        y=60,
        line_gap=18,
    )
    return header


def annotate_panel(
    frame: np.ndarray,
    expected_label: str,
    prediction: dict | None,
    side_title: str,
    border_color: tuple[int, int, int],
) -> np.ndarray:
    panel = frame.copy()
    height, width = panel.shape[:2]
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), border_color, 8)

    lines: list[tuple[str, tuple[int, int, int], float]] = [
        (side_title, border_color, 1.0),
        (f"Expected: {expected_label}", (255, 255, 255), 0.95),
    ]
    if prediction is None:
        lines.append(("Predicted: No pose", (0, 200, 255), 0.9))
    else:
        pred_color = color_for_prediction(prediction["is_error"], prediction["confidence"])
        lines.append((f"Predicted: {prediction['display_name']}", pred_color, 0.95))
        lines.append((f"Confidence: {prediction['confidence']:.2f}", (255, 255, 255), 0.85))
        if prediction["top3"]:
            lines.append((f"Alt: {prediction['top3'][1]['display_name']}", (220, 220, 220), 0.70) if len(prediction["top3"]) > 1 else ("", (0, 0, 0), 0.1))

    overlay = panel.copy()
    cv2.rectangle(overlay, (18, 18), (width - 18, 230), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.48, panel, 0.52, 0, panel)
    draw_big_label(panel, [line for line in lines if line[0]], 36, 62, line_gap=12)
    return panel


def compose_frame(left: np.ndarray, right: np.ndarray, header: np.ndarray, footer_text: str) -> np.ndarray:
    max_panel_height = max(left.shape[0], right.shape[0])
    if left.shape[0] != max_panel_height:
        top = (max_panel_height - left.shape[0]) // 2
        bottom = max_panel_height - left.shape[0] - top
        left = cv2.copyMakeBorder(left, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18))
    if right.shape[0] != max_panel_height:
        top = (max_panel_height - right.shape[0]) // 2
        bottom = max_panel_height - right.shape[0] - top
        right = cv2.copyMakeBorder(right, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18))

    divider = np.full((max_panel_height, 18, 3), 18, dtype=np.uint8)
    body = np.hstack([left, divider, right])
    footer = np.full((BOTTOM_BAR_HEIGHT, body.shape[1], 3), 22, dtype=np.uint8)
    draw_big_label(footer, [(footer_text, (245, 245, 245), 0.75)], 24, 52, line_gap=8)
    return np.vstack([header, body, footer])


def run_prediction(
    frame: np.ndarray,
    pose_model: YOLO,
    classifier: ErrorClassifierRunner,
    min_pose_confidence: float,
    previous_probs: np.ndarray | None,
    smoothing: float,
) -> tuple[np.ndarray | None, dict | None, Any]:
    result = pose_model(frame, verbose=False)[0]
    feature = extract_pose_feature_from_result(result, min_pose_confidence)
    if feature is None:
        return None, None, result

    probabilities = classifier.predict_proba(feature)
    if previous_probs is None:
        smoothed = probabilities
    else:
        alpha = float(np.clip(smoothing, 0.0, 0.99))
        smoothed = alpha * previous_probs + (1.0 - alpha) * probabilities
    return smoothed, classifier.describe_prediction(smoothed), result


def process_job(
    job: ComparisonJob,
    pose_model: YOLO,
    classifier: ErrorClassifierRunner,
    output_dir: Path,
    min_pose_confidence: float,
    smoothing: float,
    classify_every: int,
    max_seconds: float | None,
) -> Path:
    good_clip, error_clip = build_clip_pair(job)
    cap_good = open_video(good_clip)
    cap_error = open_video(error_clip)

    fps_candidates = [cap_good.get(cv2.CAP_PROP_FPS), cap_error.get(cv2.CAP_PROP_FPS)]
    fps_candidates = [fps for fps in fps_candidates if fps and fps > 1.0]
    fps = min(fps_candidates) if fps_candidates else 30.0
    fps = min(fps, 30.0)

    output_path = output_dir / f"{job.output_slug}.mp4"
    header = render_header(
        TARGET_PANEL_WIDTH * 2 + 18,
        f"{job.workout_display}: Good vs {job.error_display}",
        f"Left: Good reference    Right: Mistake example"
        + (f"    View: {job.side_display}" if job.side_display else ""),
    )

    frame_index = 0
    good_probs: np.ndarray | None = None
    error_probs: np.ndarray | None = None
    good_pred: dict | None = None
    error_pred: dict | None = None
    writer: cv2.VideoWriter | None = None
    max_frames = int(round(fps * max_seconds)) if max_seconds else None

    try:
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break

            ok_good, frame_good = cap_good.read()
            ok_error, frame_error = cap_error.read()
            if not ok_good or not ok_error:
                break

            good_result = None
            error_result = None

            if frame_index % max(classify_every, 1) == 0:
                good_probs, good_pred, good_result = run_prediction(
                    frame_good, pose_model, classifier, min_pose_confidence, good_probs, smoothing
                )
                error_probs, error_pred, error_result = run_prediction(
                    frame_error, pose_model, classifier, min_pose_confidence, error_probs, smoothing
                )

            left_vis = good_result.plot() if good_result is not None else frame_good.copy()
            right_vis = error_result.plot() if error_result is not None else frame_error.copy()
            left_vis = fit_panel(left_vis, TARGET_PANEL_WIDTH)
            right_vis = fit_panel(right_vis, TARGET_PANEL_WIDTH)

            left_panel = annotate_panel(left_vis, job.good_display, good_pred, "GOOD FORM", (40, 170, 40))
            right_panel = annotate_panel(right_vis, job.error_display, error_pred, "BAD FORM", (25, 25, 220))
            footer_text = f"{good_clip.name}  vs  {error_clip.name}"
            composed = compose_frame(left_panel, right_panel, header, footer_text)

            if writer is None:
                height, width = composed.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
            writer.write(composed)
            frame_index += 1
    finally:
        cap_good.release()
        cap_error.release()
        if writer is not None:
            writer.release()

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate side-by-side comparison videos for workout errors.")
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--min-pose-confidence", type=float, default=0.25)
    parser.add_argument("--smoothing", type=float, default=0.85)
    parser.add_argument("--classify-every", type=int, default=CLASSIFY_EVERY)
    parser.add_argument("--max-seconds", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pose_model_path = resolve_desktop_pose_model(args.pose_model)
    if not args.classifier_model.exists():
        raise FileNotFoundError(f"Could not find classifier model: {args.classifier_model}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Could not find label metadata: {args.labels}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = collect_jobs(args.dataset_dir)
    if not jobs:
        raise RuntimeError(f"No good-vs-bad comparison jobs found in {args.dataset_dir}")

    pose_model = YOLO(pose_model_path)
    classifier = ErrorClassifierRunner(args.classifier_model, args.labels)

    manifest: list[dict[str, str]] = []
    for job in jobs:
        print(f"Rendering {job.output_slug}...")
        output_path = process_job(
            job,
            pose_model,
            classifier,
            args.output_dir,
            args.min_pose_confidence,
            args.smoothing,
            args.classify_every,
            args.max_seconds,
        )
        manifest.append(
            {
                "output": str(output_path),
                "workout": job.workout_display,
                "error": job.error_display,
                "side": job.side_display or "Default",
            }
        )
        print(f"  saved {output_path}")

    manifest_path = args.output_dir / "comparison_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
