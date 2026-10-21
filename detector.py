"""Temporary synthetic OpenCV KNN detector. Fachriza owns production replacement.

DO NOT report performance here as real parking-lot accuracy. This detector
uses synthetic images supplied in the companion sample ZIP only.
"""

from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = ("empty", "occupied")


def extract_features(patch: np.ndarray) -> np.ndarray:
    """Feature vector matching the prior quick_baseline and standalone demo."""
    if patch is None or patch.size == 0:
        raise ValueError("Empty ROI patch; verify parking slot coordinates")
    small = cv2.resize(patch, (16, 20), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    return np.concatenate(
        (small.astype(np.float32).ravel() / 255.0,
         hsv.astype(np.float32).ravel() / 255.0)
    ).astype(np.float32)


class ParkingDetector:
    """Contract: ParkingDetector(train_dir).predict(frame, slots) -> results.

    `frame` is HxWx3 BGR uint8 at original resolution.
    `slots` is a list of dicts with id, x, y, w, h.
    Results are dicts with id, status ('empty'|'occupied'), x, y, w, h.
    """

    def __init__(self, train_dir: str | Path, k: int = 3):
        self.train_dir = Path(train_dir)
        self.k = k
        samples: list[np.ndarray] = []
        labels: list[int] = []
        for label, name in enumerate(CLASS_NAMES):
            source = self.train_dir / name
            paths = sorted(source.glob("*.jpg")) + sorted(source.glob("*.png"))
            if not paths:
                raise FileNotFoundError(f"No synthetic training images found: {source}")
            for path in paths:
                img = cv2.imread(str(path))
                if img is None:
                    raise RuntimeError(f"Cannot read training image: {path}")
                samples.append(extract_features(img))
                labels.append(label)
        if not hasattr(cv2, "ml"):
            raise RuntimeError("OpenCV cv2.ml is unavailable. Use opencv-python==4.12.0.88")
        self.model = cv2.ml.KNearest_create()
        self.model.setDefaultK(k)
        data = np.asarray(samples, dtype=np.float32)
        target = np.asarray(labels, dtype=np.float32)
        if not self.model.train(data, cv2.ml.ROW_SAMPLE, target):
            raise RuntimeError("Failed to train OpenCV KNN")
        print(f"Loaded {len(labels)} SYNTHETIC patches; trained KNN (k={k}).")

    def predict(self, frame: np.ndarray, slots: list[dict]) -> list[dict]:
        height, width = frame.shape[:2]
        results = []
        for slot in slots:
            x, y, w, h = (int(slot[key]) for key in ("x", "y", "w", "h"))
            if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
                raise ValueError(f"ROI outside frame {width}x{height}: {slot}")
            patch = frame[y:y+h, x:x+w]
            sample = extract_features(patch).reshape(1, -1)
            _, classification, _, _ = self.model.findNearest(sample, k=self.k)
            class_id = int(classification[0, 0])
            results.append({
                "id": str(slot["id"]),
                "status": CLASS_NAMES[class_id],
                "x": x, "y": y, "w": w, "h": h,
            })
        return results
