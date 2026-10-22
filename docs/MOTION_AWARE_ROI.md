# Smart Parking — Camera Motion / Dynamic ROI Fix

This addon takes the **original video** (not an old `annotated_video.mp4`) and **reuses the ROI file already drawn for that video**. It estimates the global camera translation, rotation and slight scale change relative to the initial ROI calibration frame, then transforms all ROI boxes for the actual camera position at each frame.

## Windows PowerShell quick start

1. Extract this ZIP into `C:\Smart-Parking-OpenCV` using `Expand-Archive ... -Force`.
2. Keep your **original source video** in its original location and don't delete `config/new_parking_video/*_rois.json` and `evaluation/new_parking_video/*_labels.json`.
3. Run:

   ```powershell
   python scripts/run_motion_aware_parking.py --pick
   ```

   Pick the exact **same source MP4** you used before when drawing ROIs, not a screenshot or an annotated output video. Existing ROIs are reused. If you have completed calibration labels, scene-reference classifier is loaded automatically; otherwise a pre-trained available KNN is used.

   Or:

   ```powershell
   python scripts/run_motion_aware_parking.py --video "C:\Users\lenovo\Downloads\your_original_parking.mp4"
   ```

4. The video opens automatically and is saved under:

   `results/new_parking_video/<video_folder>/annotated_video_motion_roi.mp4`

   Sidecar files: `preview_motion_roi.jpg`, `motion_roi_predictions.csv`, and `camera_motion_tracking.csv`.

## Reading the result

- The **ROI follows the moving scene** instead of staying fixed at pixel coordinates. Colors show `EMPTY`, `OCCUPIED`, `UNCERTAIN`.
- Banner reports `MOTION ROI ORB_OK` (feature tracking), `PHASE_OK` (fallback) or `LOST` (insufficient alignment evidence).
- A `LOST` frame is colored UNCERTAIN by design. Lowering the similarity threshold cannot correct misaligned ROIs.
- Check a frame at the beginning, middle and end. ROI tracking may fail when the camera rotates far, zooms greatly, motion blur is strong, or there are few stable visual features.
- Calibration and demo on one video still do **not** imply independent testing accuracy. Use manually labeled later frames and report uncertainty/coverage.
- **No retraining or re-labeling is required** to run this motion-aware overlay on the video whose ROIs and labels you already created.

## Practical guidance

This patch was tested against synthetic videos with known camera translations. The exact footage on your laptop was not supplied, so tracking quality on that video must be visually verified before claiming it is fixed. For large viewpoint/parallax changes, a fixed CCTV-style video is still preferable.
