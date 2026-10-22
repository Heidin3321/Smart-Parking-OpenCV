# Pexels 9010418 — Real Aerial Parking Footage for Smart Parking OpenCV

## Input footage and its scope
- Source: footage provided by the project team, from https://www.pexels.com/video/aerial-footage-of-a-parking-lot-9010418/
- Original: `9010418-uhd_3840_2160_30fps.mp4`, 3840×2160, 30 FPS, ≈21.5 seconds.
- Packaged derivative: `dataset/parking_real_720p_stabilized.mp4`, **1280×720, 10 FPS, 215 frames, 21.5 seconds**.
- The packaged video is actual aerial footage, **not the older synthetic cartoon parking data**.
- The drone moves slightly. A feature-based ORB partial-affine transform aligned frames to a reference around 10 seconds before manual ROI coordinates were defined. This is *stabilization*, not an occupancy detector.
- Source video may show changes in individual slot status, but do not assume every frame includes changes. Check video and manually label frames for evaluation.

## ROI configuration
`config/parking_9010418_roi.json` provides **18 rectangles**, IDs `T01`–`T18`, along the **top horizontal parking row** of the stabilized video, in 1280×720 coordinates.

The boxes were selected using the stabilized reference frame around 10s and spot-checked visually at 1s and 20s. **They are only suitable for this stabilized output**, not the original uncorrected 4K file nor a differently cropped/rotated clip.

`previews/roi_01s.jpg`, `roi_10s.jpg`, `roi_20s.jpg` have YELLOW ROI outlines. They are *geometry previews*, NOT vehicle-status predictions or accuracy evidence.

## Install into existing Windows project
From PowerShell in `C:\Smart-Parking-OpenCV` with `.venv` active:

```powershell
Expand-Archive -LiteralPath "C:\Users\lenovo\Downloads\SmartParking_Pexels9010418_VSCode_Update.zip" -DestinationPath "." -Force
python scripts/verify_real_video.py --time 10
```

The update **does not include or overwrite `main.py`, `detector.py`, `preprocessing.py`, or `visualization.py`**. This protects contributions from teammates.

The output video is already included; **copying the original file from Downloads is optional**:

```powershell
Copy-Item -LiteralPath "C:\Users\lenovo\Downloads\9010418-uhd_3840_2160_30fps.mp4" -Destination ".\dataset\parking_original_9010418.mp4"
```

To regenerate the stabilized video from the original (optional, potentially CPU-intensive):

```powershell
python scripts/prepare_aerial_video.py --source dataset/parking_original_9010418.mp4 --output dataset/parking_real_720p_stabilized.mp4
```

## Run existing Smart Parking program using CNRPark-trained KNN
This requires your **previously prepared** `dataset/real_cnrpark/train/empty` and `dataset/real_cnrpark/train/occupied` folders. You already made these via `python scripts/cnrpark_real.py prepare`.

```powershell
python main.py --mode detect --training-domain real --source dataset/parking_real_720p_stabilized.mp4 --config config/parking_9010418_roi.json --train-dir dataset/real_cnrpark/train --output-dir results/real_video_9010418 --no-window --save-video
```

You should then find:
- `results/real_video_9010418/first_frame.jpg`
- `results/real_video_9010418/annotated_video.mp4`
- `results/real_video_9010418/predictions.csv`

Display annotated video:

```powershell
ii .\results\real_video_9010418\annotated_video.mp4
```

**Important:** Real video input + real training photos does NOT guarantee correct predictions. The previous 96.56% was **CNRPark crop test accuracy**, not accuracy on this Pexels drone video. Small vehicles, different scale, and viewpoint cause domain shift. Inspect predictions against vehicles and do manual video-specific evaluation.

## Manual testing for Lenno
1. Open `evaluation/real_video_ground_truth_template.csv` (54 rows for 3 timestamps × 18 slots).
2. Inspect the stabilized video around frame **10 (1.0s), 100 (10.0s), and 200 (20.0s)**. Fill `true_status` manually with `empty` or `occupied` for *every slot inspected*. If uncertain, leave blank, and explain.
3. After running `main.py`, run:

```powershell
python scripts/evaluate_real_video_predictions.py
```

The output measures agreement with *manually reviewed* ROI states for just those samples, not whole-parking or unseen-camera accuracy. Do not present unverified 100% synthetic scores as real-world results.

## Version control
`dataset/*.mp4` is ignored in your existing `.gitignore` — keep it ignored. Only commit `config/parking_9010418_roi.json`, scripts, guides, and (if desired) ground-truth **template**. Share the MP4 separately with teammates while following the source license. Do not `git add -f` the movie unless everyone agrees licensing and repo access are appropriate.

```powershell
git status
git add config/parking_9010418_roi.json scripts/prepare_aerial_video.py scripts/verify_real_video.py scripts/evaluate_real_video_predictions.py docs/PARKING_9010418_GUIDE.md evaluation/real_video_ground_truth_template.csv
git commit -m "Add stabilized real parking video ROI and evaluation tools"
git push origin main
```

## Next steps
- Rifqi: verify ROI boxes in actual output frame at 1/10/20 seconds and document source/processing.
- Fachriza: review KNN predictions on these 18 slots and improve real-video performance where needed.
- Ichsan: integrate results into report with clear provenance.
- Lenno: fill ground truth and run evaluation; capture actual annotated outputs.
