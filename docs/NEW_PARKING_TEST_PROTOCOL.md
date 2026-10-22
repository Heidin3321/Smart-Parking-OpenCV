# ETS validation protocol: Smart Parking on an unseen photo

## Objective
Evaluate whether **existing** two-class OpenCV KNN classifiers transfer to a different parking-lot photograph, without retraining on that photo. The existing Reference Matcher is video-/slot-specific, so it is not eligible for a zero-shot test on a new camera.

## Procedure
1. Obtain at least one real parking photograph from a **different** lot than the CNRPark and V3 calibration footage (check license and provenance).
2. Manually annotate parking-bay rectangles. Record the number of ROI, source location, and image source (if public). These manual ROIs are a configuration step, not training labels.
3. **Before inference**, a team member inspects each ROI and labels EMPTY/OCCUPIED/UNKNOWN. UNKNOWN is for ambiguous or occluded bays.
4. Run both **pre-existing** KNN models on the frozen image/ROI/labels. These models do not ingest the test photo into their training folders.
5. Export per-slot CSV and image annotations; use report.html in the live demo. Record TP/FP/FN/TN and accuracy for each model on the SAME photo.
6. Show at least one failure case if it occurs. The reason may include camera angle, lighting, vehicle scale, or unsuitable boundaries.
7. Repeat on a SECOND genuinely different parking photo if feasible; keep each test run separately and summarize the results with the `report` command.

## Metrics
- Accuracy = correct prediction among valid manually labeled EMPTY and OCCUPIED ROIs / valid labeled ROIs.
- For each class separately, precision, recall, F1 (undefined reported as null/N/A when no support/predictions).
- Confusion matrix columns = predicted EMPTY/OCCUPIED; rows = actual EMPTY/OCCUPIED.
- Ground-truth UNKNOWN excluded from accuracy and displayed as counts. Predicted UNCERTAIN, if supported by a future detector, is a failure for strict accuracy and also shown in prediction coverage.

## Scientific limitations
A few manually selected ROIs on one image are not an independent sample of full-day CCTV footage. All slots in one photo share lighting and perspective, and manually adjusted ROIs can improve results. No independent accuracy is inferred for the original V3 per-slot Reference Matcher. Hardware deployment is not demonstrated.

## 3–5 minute demonstration script
- 0:00–0:35: show raw parking photo from a new location and training/test split motivation
- 0:35–1:25: draw 4–8 ROIs on the new photo live
- 1:25–2:10: label EMPTY/OCCUPIED manually before showing any classifier results
- 2:10–3:10: run the two pretrained KNN models and compare their predictions with truth
- 3:10–3:50: open HTML report, discuss failure examples and accuracy limitations
- 3:50–4:40: open original V3 video to demonstrate changing occupancy over time
