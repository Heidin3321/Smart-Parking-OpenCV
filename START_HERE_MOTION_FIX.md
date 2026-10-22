# MOTION ROI FIX — NO NEED TO DRAW ROIs AGAIN

In VS Code PowerShell (C:\Smart-Parking-OpenCV):

```powershell
Expand-Archive -LiteralPath "C:\Users\lenovo\Downloads\SmartParking_Camera_Motion_ROI_FIX.zip" -DestinationPath "C:\Smart-Parking-OpenCV" -Force
python scripts/run_motion_aware_parking.py --pick
```

Select the **same ORIGINAL new-location video file** whose ROIs you already saved. Do NOT pick the old annotated output video. It keeps existing ROIs and calibration labels.

Output: `results/new_parking_video/<video_ID>/annotated_video_motion_roi.mp4`. Video opens automatically on Windows. You can inspect movement estimate in `camera_motion_tracking.csv`.

Read `docs/MOTION_AWARE_ROI.md` for details and reliability limitations.
