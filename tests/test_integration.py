"""Dependency-free unit and smoke tests. Run: python -m unittest discover -s tests -v"""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import read_slots  # noqa: E402
from preprocessing import preprocess_frame  # noqa: E402
from visualization import draw_detection  # noqa: E402


class IntegrationContractTests(unittest.TestCase):
    def test_roi_json_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            file = Path(temp) / "slots.json"
            payload = {"frame_width": 320, "frame_height": 240,
                       "slots": [{"id": "A1", "x": 20, "y": 90, "w": 50, "h": 70}]}
            file.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(read_slots(file)["slots"][0]["id"], "A1")
            payload["slots"].append(dict(payload["slots"][0]))
            file.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_slots(file)

    def test_preprocessing_does_not_resize_frame(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.assertEqual(preprocess_frame(frame).shape, frame.shape)

    def test_visualization_preserves_dimensions(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        detected = [{"id": "A1", "status": "empty", "x": 20, "y": 90,
                     "w": 50, "h": 70}]
        painted = draw_detection(frame, detected, 0)
        self.assertEqual(painted.shape, frame.shape)

    def test_synthetic_pipeline_for_three_frames_if_available(self):
        sample = ROOT / "SmartParking_TestDataset"
        if not (sample / "dataset/parking_sample.mp4").is_file():
            self.skipTest("Companion synthetic dataset not yet extracted")
        with tempfile.TemporaryDirectory() as temp:
            args = [sys.executable, str(ROOT / "main.py"), "--no-window", "--max-frames", "3",
                    "--output-dir", temp]
            run = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, timeout=60)
            self.assertEqual(run.returncode, 0, msg=run.stdout+run.stderr)
            self.assertTrue((Path(temp) / "first_frame.jpg").is_file())
            with (Path(temp) / "predictions.csv").open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 3 * 12)
            self.assertTrue(all(item["predicted_status"] in ("empty", "occupied")
                                for item in rows))


if __name__ == "__main__":
    unittest.main()
