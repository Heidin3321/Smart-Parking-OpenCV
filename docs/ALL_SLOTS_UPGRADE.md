# Smart Parking — Expanded Parking-Slot Coverage (ROI Draft)

## Scope and limitations

This update expands the original 18 top-row ROIs to **250 manually estimated candidate rectangles** across the large parking lot shown in Pexels video 9010418. **250 is not a verified physical capacity**: some rectangles may need re-positioning, some potential spaces may not exist, and diagonal rows at the lower left have NOT all been annotated. It is a starting draft, not a validated set of parking spaces. The video must be stabilized to 1280x720 and matched to this configuration.

The boxes specify *where to classify*; they do NOT imply that a parking spot has been automatically discovered. KNN estimates `empty` / `occupied`, but its accuracy on this Pexels footage is NOT known. The earlier 96.56% result was measured on holdout patches from CNRPark, NOT on Pexels video frames.

## Existing inputs (already on your machine)

- `dataset/parking_real_720p_stabilized.mp4` — previous stabilized 720p MP4.
- `dataset/real_cnrpark/train/empty` and `dataset/real_cnrpark/train/occupied` — your already prepared REAL CNRPark training samples.
- `main.py`, `detector.py`, `preprocessing.py` — existing project integration; not overwritten by this ZIP.

## New files

- `config/parking_9010418_all_slots_DRAFT.json` — 250 candidate ROIs, 1280x720, including rotation for some side spaces.
- `scripts/preview_all_slots.py` — draw yellow ROI geometry at selected seconds, NO classifier.
- `scripts/edit_all_slots.py` — interactive correction of rectangles and bulk row additions.
- `scripts/run_all_slots.py` — BATCH KNN inference with thin overlays and CSV/video export; preserves original project entrypoint.
- `tests/test_all_slots.py` — validates config and compares batch vs existing scalar detector.
- `previews/` — screenshots of geometry at 1s, 10s, and 20s.

## 1. Inspect and correct ROI geometry

```powershell
python scripts/preview_all_slots.py
Start-Process .\results\all_slots\roi_review\roi_10s.jpg
```

The orange/yellow boxes are CANDIDATE geometry only. Check 1s, 10s and 20s. Pay special attention to left-side angled spaces, horizontal side bays and the two lower rows. Remove invalid boxes or add missing spaces using the editor:

```powershell
python scripts/edit_all_slots.py --time 10
```

- **Left drag mouse**: add single rectangular ROI.
- **Right click**: delete nearest rectangle (within 50 px).
- **R**: toggle rectangle / auto-row mode.
- **Row mode**: left-click first and last slot center to create N evenly spaced rectangles (default 10, `--row-count 10`).
- **I**: toggle ID labels; **U**: undo last edit; **S**: save to the REVIEWED config; **Q**: close.

To set row shape/count: `python scripts/edit_all_slots.py --row-count 8 --slot-width 20 --slot-height 48`.

**Caution:** Save often. All corrections are local unless saved. The script writes `config/parking_9010418_all_slots_REVIEWED.json`, leaving the original draft untouched. For diagonal ROIs, manually adjust rectangle geometry and, if necessary, set `rotation_deg` in the JSON before testing classification.

After editing, use the reviewed file for inference by supplying `--config config/parking_9010418_all_slots_REVIEWED.json`.

## 2. Run a fast trial (not real-video accuracy testing)

```powershell
python scripts/run_all_slots.py --max-frames 20 --step 2 --save-video
Start-Process .\results\all_slots\annotated_video.mp4
```

A longer run:

```powershell
python scripts/run_all_slots.py --step 2 --save-video --config config/parking_9010418_all_slots_REVIEWED.json
```

- `--step 2` processes every other source frame and exports half-frame-rate video with the original duration approximately retained; use `--step 1` for all frames.
- `--show-ids` turns on tiny ROI labels, but becomes cluttered on dense scenes.
- `--train-dir dataset/real_cnrpark/train` is the default; use `--training-domain synthetic` if intentionally supplying synthetic training patches.
- Files written to `results/all_slots/`: `predictions.csv`, `first_frame.jpg`, optional `annotated_video.mp4`.

**IMPORTANT:** The classifier runs across all configured ROIs and produces a count. It does NOT mean the counts are correct. At Pexels scale, cars are small and may be misclassified; mixed camera angle and shadows compound errors.

## 3. Evaluate, do not invent accuracy

Ask Lenno to label a sample of slots on 3 or more frames manually, without looking at KNN predictions. Compare the labels with `results/all_slots/predictions.csv`. Note false empty predictions, false occupied predictions, tracking drift, missed diagonal spaces, and moving cars occluding stationary slots. Make sure the evaluation script uses the SAME source frame IDs (0-based, after `--step` selection).

For Week 8 report, state exactly:

- **250 candidate ROIs** have been drafted, not 250 verified/automatically detected parking spaces.
- KNN trained with real CNRPark sample crops.
- No claim of real-video accuracy until ground truth is created.
- The dataset CNRPark holdout accuracy 96.56% is a separate result.

## Version control

On your feature branch, commit **only** the new code/config/docs/tests. Do not upload video, CNRPark images, synthetic dataset, or generated outputs without checking the source/license and privacy. When ready, review the branch and merge to `main` after tests pass.
