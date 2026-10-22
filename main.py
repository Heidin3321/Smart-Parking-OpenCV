"""Smart Parking Week 8 integration entry point — Ichsan.

Modes:
  input  : smoke-test webcam/video, no classifier or ROI (safe fallback)
  detect : automatic occupancy predictions with synthetic KNN baseline

This code is a starter; Fachriza's detector may replace the temporary KNN.
"""

import argparse
import csv
from pathlib import Path

import cv2

from preprocessing import preprocess_frame
from detector import ParkingDetector
from visualization import draw_detection, draw_input_test

ROOT = Path(__file__).resolve().parent
SYNTHETIC_DATA = ROOT / "SmartParking_TestDataset"
DEFAULT_SOURCE = SYNTHETIC_DATA / "dataset" / "parking_sample.mp4"
DEFAULT_CONFIG = SYNTHETIC_DATA / "config" / "parking_slots.json"
DEFAULT_TRAIN = SYNTHETIC_DATA / "dataset" / "patches" / "train"


def resolve_file(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_slots(path: Path) -> dict:
    """Validate Rifqi's JSON against the shared ROI contract."""
    import json
    if not path.is_file():
        raise FileNotFoundError(f"ROI JSON missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    width = data.get("frame_width")
    height = data.get("frame_height")
    if not isinstance(width, int) or not isinstance(height, int) or width < 1 or height < 1:
        raise ValueError("ROI JSON must contain positive integer frame_width and frame_height")
    slots = data.get("slots")
    if not isinstance(slots, list) or not slots:
        raise ValueError("ROI JSON must contain nonempty slots list")
    ids = set()
    for slot in slots:
        if not isinstance(slot, dict) or not all(k in slot for k in ("id", "x", "y", "w", "h")):
            raise ValueError("Every ROI requires id, x, y, w, h")
        sid = str(slot["id"])
        if not sid or sid in ids:
            raise ValueError(f"Empty or repeated parking slot id: {sid!r}")
        ids.add(sid)
        x, y, w, h = (slot[k] for k in ("x", "y", "w", "h"))
        if any(type(v) is not int for v in (x, y, w, h)):
            raise ValueError(f"Slot {sid}: ROI coordinates must be integers")
        if x < 0 or y < 0 or w < 1 or h < 1 or x+w > width or y+h > height:
            raise ValueError(f"Slot {sid}: ROI outside frame {width}x{height}")
    return data


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smart Parking Week 8 integrated prototype")
    parser.add_argument("--mode", choices=("input", "detect"), default="detect",
                        help="input: video/webcam check; detect: KNN occupancy (default)")
    parser.add_argument("--training-domain", choices=("synthetic", "real", "unknown"), default="synthetic",
                        help="Honest input/training provenance for video annotation")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE),
                        help="Video path or integer camera index such as 0")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="ROI JSON matching video")
    parser.add_argument("--train-dir", default=str(DEFAULT_TRAIN),
                        help="Two patch subfolders: empty/ and occupied/")
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "integration"))
    parser.add_argument("--no-window", action="store_true", help="Skip OpenCV GUI")
    parser.add_argument("--save-video", action="store_true", help="Save annotated MP4")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N frames; 0 means until video finishes")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    if args.max_frames < 0:
        raise ValueError("--max-frames must be 0 or a positive integer")
    source = int(args.source) if args.source.isdecimal() else str(resolve_file(args.source))
    if isinstance(source, str) and not Path(source).is_file():
        raise FileNotFoundError(f"Video missing: {source} (or use --mode input --source 0)")

    config = None
    detector = None
    if args.mode == "detect":
        config = read_slots(resolve_file(args.config))
        detector = ParkingDetector(resolve_file(args.train_dir))
        if args.training_domain == "synthetic":
            print("WARNING: SYNTHETIC training; not validated on real CCTV.")
        else:
            print(f"TRAINING DOMAIN: {args.training_domain}. Predictions on video require separate ground-truth evaluation.")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video/camera source: {source}")

    output = resolve_file(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not (1 <= fps <= 120):
        fps = 8.0
    wait_ms = max(1, int(round(1000 / fps)))
    csv_path = output / "predictions.csv"
    writer = None
    frame_id = 0
    first_frame_saved = False
    print(f"Mode={args.mode}; source={args.source}. Press S=screenshot, Q=quit.")
    try:
        if args.mode == "detect":
            csv_file = csv_path.open("w", encoding="utf-8", newline="")
            rows = csv.writer(csv_file)
            rows.writerow(("frame", "slot_id", "predicted_status"))
        else:
            csv_file = None
            rows = None
        try:
            while args.max_frames == 0 or frame_id < args.max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if args.mode == "detect":
                    actual = (frame.shape[1], frame.shape[0])
                    required = (config["frame_width"], config["frame_height"])
                    if actual != required:
                        raise ValueError(
                            f"ROI resolution {required} != input resolution {actual}. "
                            "Use corresponding coordinates / fixed camera."
                        )
                    # Preprocessing contract preserves resolution and BGR channels.
                    prepared = preprocess_frame(frame)
                    if prepared.shape != frame.shape:
                        raise ValueError("preprocess_frame changed frame dimensions; breaks ROI mapping")
                    predictions = detector.predict(prepared, config["slots"])
                    for item in predictions:
                        rows.writerow((frame_id, item["id"], item["status"]))
                    label = ("SYNTHETIC DEMO ONLY" if args.training_domain == "synthetic"
                             else f"TRAIN={args.training_domain.upper()} | VIDEO NOT YET VALIDATED")
                    rendered = draw_detection(frame, predictions, frame_id, dataset_label=label)
                else:
                    rendered = draw_input_test(frame)

                if not first_frame_saved:
                    destination = output / "first_frame.jpg"
                    if not cv2.imwrite(str(destination), rendered):
                        raise RuntimeError(f"Cannot write screenshot: {destination}")
                    first_frame_saved = True
                if args.save_video:
                    if writer is None:
                        destination = output / "annotated_video.mp4"
                        h, w = rendered.shape[:2]
                        writer = cv2.VideoWriter(
                            str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Cannot create annotated video: {destination}")
                    writer.write(rendered)
                if not args.no_window:
                    cv2.imshow("Smart Parking — Week 8 Integration", rendered)
                    key = cv2.waitKey(wait_ms) & 0xFF
                    if key == ord("s"):
                        screenshot = output / f"screenshot_frame_{frame_id:04d}.jpg"
                        if cv2.imwrite(str(screenshot), rendered):
                            print(f"Screenshot saved: {screenshot}")
                    if key == ord("q"):
                        frame_id += 1
                        break
                frame_id += 1
        finally:
            if csv_file is not None:
                csv_file.close()
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_window:
            cv2.destroyAllWindows()
    if frame_id == 0:
        raise RuntimeError("Video opened but contains no readable frames")
    print(f"Processed {frame_id} frames. Results: {output}")
    if args.mode == "detect":
        print(f"Predictions CSV: {csv_path}")
    if args.save_video:
        print(f"Annotated video: {output / 'annotated_video.mp4'}")
    return {"processed_frames": frame_id, "output_dir": output}


if __name__ == "__main__":
    try:
        run(build_args())
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
