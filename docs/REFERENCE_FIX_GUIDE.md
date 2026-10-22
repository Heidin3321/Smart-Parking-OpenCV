# Parking V3: Per-slot reference classifier (scene-specific)

## Why the adapted KNN can still predict EMPTY for visible cars

The earlier `v3_adapt.py prepare` uses `choose_balanced` to downsample the bigger class when it constructs `deploy_train` (in a 14-slot scene the occupied class may have many more examples). Thus, the deployed KNN can discard visually relevant examples, and the 16x20 global color+HSV feature is not robust enough for all vehicles.

## New strategy

This independent script uses **all valid human-labelled frame crops** per specific ROI, and compares frames against labelled images of *that same slot*. It does not delete your existing KNN or update `main.py`. A substantially different crop is shown as **UNCERTAIN (yellow)** rather than automatically being marked as EMPTY.

Training on slot P007 and demonstrating slot P007 with the same vehicle is **scene memorization / camera-specific calibration**, not evidence of the model generalizing to another parking lot. The current video and labels are **not independent** of one another. For a graded accuracy claim, use a separate video / separately evaluated frames, and preserve held-out `v3_adapt.py evaluate` for comparison.

## Installation

Extract ZIP into `C:\Smart-Parking-OpenCV` while on your `feature/parking-v3` branch. Do not overwrite source modules.

## Confirm labels first

```
python scripts/run_v3_reference.py --audit-only
```

You need `evaluation/parking_v3_manual_labels.csv` from the earlier manual labeling step, a video at `dataset/parking_v3_stabilized.mp4`, and `config/parking_v3_slots_REVIEWED.json`.

Check `results/parking_v3_reference/reference_audit.json` for label count **per slot**, and open `results/parking_v3_reference/reference_label_audit.jpg` to **VISUALLY** verify the labels. If there are incorrect manual labels, correct them with visual inspection of the original frame. **Do not use model predictions as ground truth.**

## Try on the same video

```
python scripts/run_v3_reference.py --max-frames 150 --save-video
```

```
python scripts/run_v3_reference.py --save-video
```

```
ii .\results\parking_v3_reference\annotated_video.mp4
```

Yellow means **UNCERTAIN**, not free. Red means occupied, green means empty.

## Limits

- Valid mainly for a fixed/stabilized camera and the same ROI positions.
- New parking location/angle needs new manually labelled references.
- Vehicle motion across parking bays can be flagged uncertain during the transition.
- A pixel-based matching model may misinterpret large changes in sunshine, occlusions, or camera motion.
- DO NOT quote the output video as independent accuracy; keep evaluation separate and record limitations in Week 8 report.

## Troubleshooting

- Missing `parking_v3_manual_labels.csv`: run `python scripts/v3_adapt.py label` and complete labels.
- JSON contains 0 ROI: restore the reviewed 14 ROI; don't run `--reset` and save 0 entries.
- ROI changed since labeling: re-label using revised config; the script refuses stale coordinates intentionally.
- Too many yellow slots: inspect labels, frame stabilization, and reference-distance diagnostics in `predictions.csv`; capture additional correctly labeled frames.

## Git

The runner and guide are safe to commit. Keep `dataset/`, `results/`, and your personal manual labels out of a public repository unless you have appropriate permission and reviewed their contents.
