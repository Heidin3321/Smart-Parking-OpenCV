# SMART PARKING — NEW LOCATION VIDEO TEST (Week 8)

**What this is:** An independent **new parking video input** workflow, NOT the old single-image tester or Parking V3 hardcoded ROI. Works for MP4/AVI/MOV/MKV supported by OpenCV, with preferably *stationary* camera footage.

## Install

From PowerShell, while at `C:\Smart-Parking-OpenCV` (venv activated):

```powershell
Expand-Archive -LiteralPath "C:\Users\lenovo\Downloads\SmartParking_New_Video_Test_Kit.zip" -DestinationPath "C:\Smart-Parking-OpenCV" -Force
Test-Path .\scripts\test_new_parking_video.py
```

## Demonstration: choose DIFFERENT parking VIDEO

```powershell
python scripts/test_new_parking_video.py wizard --pick
```

Or supply the file path directly:

```powershell
python scripts/test_new_parking_video.py wizard --video "C:\Users\lenovo\Downloads\parking_lokasi_baru.mp4"
```

1. Pick **a video (MP4)**, not an image. Prefer 10–60 seconds from a camera that **does not move or pan**. The same fixed ROIs must remain aligned throughout the clip. Have some occupied and empty slots for calibration.
2. In the ROI window, **drag rectangular boxes over parking spaces** using the left mouse button. Press `S` to save; `U` undo; `C` clear; `Q` cancel. ROIs are stored per VIDEO ID and automatically resumed. NO `--reset-roi` unless deliberately redrawing from scratch.
3. In manual label window, classify the visible slot: `O` occupied, `E` empty, `U` unknown, `B` back, `Q` save/quit. This step does **not** show model predictions. Labels save after *every* keypress. Re-run the **same wizard** to resume.
4. Default samples: **2 early calibration frames** + **2 later blind test frames** x number of ROIs. For 20 slots, that is **80 manual labels**. You may increase sample count via `--calib-count 3 --test-count 3` **before** drawing the ROI; then always use same arguments on resuming.
5. Training: `new_scene_knn` is retrained on calibration crops **only**. `new_scene_reference` uses early-frame same-slot labelled gallery **only**. `cnrpark_pretrained` is an optional baseline trained on existing `dataset/real_cnrpark/train` (not this video).
6. Test: compare each model against human labels from **later, temporally held-out frames**. Neither KNN training nor reference gallery sees those labels. Outputs are saved automatically.

## Results

All outputs are under `results/new_parking_video/<video_name_id>/`:

- `report.html` — open in Chrome/Edge; test metrics, confusion matrix, visual previews
- `report.json` — metadata, protocol, metrics, limitations
- `annotated_new_scene_knn.mp4` — demo video from KNN trained on the new video's EARLY labelled frames
- `annotated_new_scene_reference.mp4` — demo video using EARLY-frame slot reference images
- `annotated_cnrpark_pretrained.mp4` — baseline model if CNRPark dataset exists
- `test_predictions_*.csv` — blind later-frame human ground truth versus model predictions
- `video_predictions_*.csv` — prediction per slot and rendered video frame
- `preview_*.jpg` — screenshot from the output video

Open most recent HTML report in PowerShell:

```powershell
$report = Get-ChildItem .\results\new_parking_video -Recurse -Filter report.html | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($report) { ii $report.FullName } else { Write-Host "Report not made yet; finish ROI and labels first." }
```

No Windows `.json.part -> .json` rename is used when saving labels, avoiding the previous WinError 5 on some Windows installations.

## Resume or re-render

If labeling was paused (Q): simply repeat `wizard --video "..."` and use the **same** `--calib-count`/`--test-count` settings.

If everything is already labeled and you only want to regenerate output (no GUI):

```powershell
python scripts/test_new_parking_video.py run --video "C:\Users\lenovo\Downloads\parking_lokasi_baru.mp4"
```

For a short demo only (10 output seconds): add `--max-seconds 10`. For saving full clip, omit it. Default output is sampled at ~10 FPS but preserves playback duration.

To **explicitly** redraw saved ROI: `wizard --video "..." --reset-roi`; previous labels are backed up as `.bak` and all ROI labels must be redone.

## Research integrity / what this proves

- Early video frames (first ~34%) are calibration/training; later ones (last ~43%) are blind human-labelled testing.
- This avoids *identical-frame* train/test overlap, but both sets still originate from **the same recording**. Therefore, scores are **within-video temporal holdout**, not independent-site or real-world deployment accuracy.
- Testing on a second independent session from the same camera—or a different parking lot, without calibrating on that test recording—is stronger. Camera scene can have different lighting and weather.
- ROIs are fixed rectangles, so **moving drone/panning footage will break alignment** unless footage is stabilized or ROI-tracked beforehand. Always inspect start/middle/end ROI alignment.
- For a stronger presentation, show (i) the old V3 demo, (ii) the NEW video selection and manually drawn ROI, (iii) per-video calibration, (iv) unseen later-frame evaluation **including errors**.
- A video where **all** calibration slots are occupied or **all** are empty cannot train a binary KNN. Choose another clip or additional early calibration frames containing both classes. The app will warn and still make reference predictions.
- A model trained on new footage cannot be treated as a test of generalization to that same footage; for generalization tests, use *previously trained* CNRPark baseline without retraining on that footage.

## Tests

```powershell
python -m unittest discover -s tests -v
```
