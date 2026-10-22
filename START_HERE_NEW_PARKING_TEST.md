# SMART PARKING — TEST ON A NEW PARKING IMAGE (ETS)

This upgrade adds a **blind unseen-photo test** to your existing OpenCV project. It is **not** a replacement for the V3 reference matcher, and **does not train on the test image**. Use a REAL picture of a DIFFERENT parking lot, preferably one with clearly visible empty and occupied spots.

## Install

1. Keep a backup/commit of your existing `feature/parking-v3` branch.
2. Extract this ZIP into `C:\Smart-Parking-OpenCV` (same folder as `main.py` and `detector.py`).
3. Activate the existing `.venv` and open PowerShell from the project root.
4. Put your independent parking lot JPG/PNG in Downloads. Name it `parkir_baru.jpg` in the examples below.

## One-command interactive demonstration

```powershell
python scripts/test_new_parking.py wizard --pick --model both
```

This opens a **Windows file picker**. If it is unavailable (no tkinter), use the original direct path command:

```powershell
python scripts/test_new_parking.py wizard --image "C:\Users\lenovo\Downloads\parkir_baru.jpg" --model both
```

1. **Manual ROI**: draw rectangles with the left mouse. `S` saves, `U` undoes, `C` clears, `Q` cancels. ROI IDs T001, T002, ... are assigned in drawing order. The source image is never resized for detection: only its on-screen *preview* is scaled; saved coordinates refer to the original resolution.
2. **Blind manual ground truth**: identify each slot **without seeing the AI prediction**. `O` = occupied, `E` = empty, `U` = unknown/occluded, `B` = previous, `Q` = save & quit. Incomplete labeling cannot enter testing. You can restart the command to resume.
3. **Testing**: program loads existing KNN training crops from CNRPark and V3 calibration and classifies the *new* image; it never adds the test photo to any training dataset.
4. **Outputs**: annotated images (one per model), independent human-truth image, per-slot predictions CSVs, `report.json` (includes confusion matrix) and a clear `report.html` to show the lecturer.

## Staged commands (if you want to control the demo)

```powershell
python scripts/test_new_parking.py roi --image "C:\Users\lenovo\Downloads\parkir_baru.jpg"
python scripts/test_new_parking.py label --image "C:\Users\lenovo\Downloads\parkir_baru.jpg"
python scripts/test_new_parking.py test --image "C:\Users\lenovo\Downloads\parkir_baru.jpg" --model both
python scripts/test_new_parking.py report
```

If only CNRPark training exists, use `--model cnrpark`. If you need to restart ROI drawing use `roi --replace-roi --image ...`; the previous truth labels must then be renewed. To correct a human label use `label --start-at T003 --image ...`.

## Training prerequisites

`--model cnrpark` requires `dataset/real_cnrpark/train/empty/` and `.../occupied/` images.

`--model v3_adapted` requires `dataset/v3_calibrated/deploy_train/empty/` and `.../occupied/` images created with `python scripts/v3_adapt.py prepare` earlier. The V3 model was calibrated on the previous V3 video, not on this new photo.

`--model both` (default) compares them on **exactly the same new photo/ROI/ground-truth labels**.

## File structure

```
config/new_parking/<image_name_hash>_rois.json          # Original-resolution ROI coordinates
evaluation/new_parking/<image_name_hash>_truth.json     # Human labels entered before prediction
results/new_parking/<image_name_hash>/ground_truth.jpg
results/new_parking/<image_name_hash>/annotated_cnrpark.jpg
results/new_parking/<image_name_hash>/annotated_v3_adapted.jpg
results/new_parking/<image_name_hash>/predictions_cnrpark.csv
results/new_parking/<image_name_hash>/predictions_v3_adapted.csv
results/new_parking/<image_name_hash>/report.json
results/new_parking/<image_name_hash>/report.html
results/new_parking/ALL_NEW_IMAGE_TESTS.csv
```

`results/` and `dataset/` are ignored by Git in your repository, so copy the screenshots and report to your separate ETS evidence folder before cleaning anything.

## Honest claims during presentation

- **What you can demonstrate**: a different parking photo, manual ROI configuration, blind human labeling, inference by models pretrained on CNRPark/V3, measurable accuracy and per-slot mistakes.
- **What you cannot claim**: that one photo proves CCTV readiness or generalizes to all lighting/weather/angles; that a V3 reference classifier works unchanged in a different parking lot; or that pictures containing only cars establish performance on empty spaces.
- A small test should contain **both EMPTY and OCCUPIED** slots. Make repeated tests with **more than one independently captured location** where feasible. Unknown/occluded true labels are excluded from accuracy and reported separately.
- For *online, dynamic* state transitions (car enters/leaves), still use the existing V3 video demo.
- This is a **software prototype test without new hardware**, not a claim of real CCTV deployment.

## Tests

```powershell
python -m unittest discover -s tests -v
```

In the Python OpenCV ROI window, click the image and press keys in that window (not in PowerShell). Windows requires a desktop session; the GUI cannot run on a headless server.
