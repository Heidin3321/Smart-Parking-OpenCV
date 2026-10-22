"""Camera-motion compensation for Smart Parking / NEW LOCATION VIDEO.

Use the original NEW VIDEO file, reusing its existing saved ROIs and calibration
labels (when available). Tracks scene motion relative to the very frame on
which the ROI editor first displayed the video. Each frame's ROIs move WITH
vehicles and parking lines. Generates a real annotated MP4 (not just CSV).

Examples:
    python scripts/run_motion_aware_parking.py --pick
    python scripts/run_motion_aware_parking.py --video "C:\\Users\\lenovo\\Downloads\\parking.mp4"
    python scripts/run_motion_aware_parking.py --video "parking.mp4" --detector scene_reference

Important: global scene alignment is only valid for mostly rigid camera motion.
Untrustworthy motion estimates are NOT silently treated as zero motion.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

import test_new_parking_video as base


def gray_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # CLAHE makes dark vehicle edges and painted lines easier to match.
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


class CameraMotion:
    """Finds reference->current similarity transform using ORB/RANSAC.

    Tracks each frame against a SINGLE reference (no accumulating drift).
    Falls back to a gradient phase-correlation translation only when good.
    """
    def __init__(self, reference, max_shift_ratio=0.25):
        self.h, self.w = reference.shape[:2]
        self.max_shift_ratio = max_shift_ratio
        self.orb = cv2.ORB_create(nfeatures=3200, scaleFactor=1.18, nlevels=8,
                                  edgeThreshold=12, fastThreshold=10)
        self.ref_gray = gray_features(reference)
        self.ref_pts, self.ref_des = self.orb.detectAndCompute(self.ref_gray, None)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.prev = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float64)
        self.ref_gradient = self.gradient(self.ref_gray)
        self.last_good = 0

    @staticmethod
    def gradient(gray):
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        m = cv2.magnitude(gx, gy)
        return m - float(m.mean())

    def is_reasonable(self, mat):
        a, b = mat[0, 0], mat[1, 0]
        scale = math.hypot(float(a), float(b))
        rot = abs(math.degrees(math.atan2(float(b), float(a))))
        return (0.91 <= scale <= 1.09 and rot < 9.0 and
                abs(float(mat[0, 2])) < self.w * self.max_shift_ratio and
                abs(float(mat[1, 2])) < self.h * self.max_shift_ratio)

    def estimate(self, current):
        gray = gray_features(current)
        keypoints, desc = self.orb.detectAndCompute(gray, None)
        if self.ref_des is not None and desc is not None and len(keypoints) >= 15:
            matches = self.matcher.knnMatch(self.ref_des, desc, k=2)
            selected = [m for pair in matches if len(pair) == 2
                        for m, n in [pair] if m.distance < 0.77 * n.distance]
            if len(selected) >= 12:
                src = np.float32([self.ref_pts[m.queryIdx].pt for m in selected])
                dst = np.float32([keypoints[m.trainIdx].pt for m in selected])
                mat, inliers = cv2.estimateAffinePartial2D(
                    src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.5,
                    maxIters=2500, confidence=0.995)
                support = int(inliers.sum()) if inliers is not None else 0
                ratio = support / len(selected)
                if (mat is not None and support >= 10 and ratio >= 0.38
                        and self.is_reasonable(mat)):
                    self.prev = mat.astype(np.float64)
                    return self.prev.copy(), 'ORB_OK', support, ratio

        # If ORB fails, phase correlate STATIC frame structure (image gradients).
        try:
            shift, response = cv2.phaseCorrelate(self.ref_gradient, self.gradient(gray))
            dx, dy = shift
            mat = np.array([[1., 0., dx], [0., 1., dy]], dtype=np.float64)
            if (response >= 0.17 and math.isfinite(dx) and math.isfinite(dy)
                    and self.is_reasonable(mat)):
                self.prev = mat
                return mat.copy(), 'PHASE_OK', 0, float(response)
        except cv2.error:
            pass
        # Do not confidently classify with stale ROIs when tracking fails.
        return self.prev.copy(), 'LOST', 0, 0.


def transform_slot(slot, mat, width, height):
    x, y, w, h = (int(slot[k]) for k in ('x', 'y', 'w', 'h'))
    pts = np.float32([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
    moved = pts @ mat[:, :2].T + mat[:, 2]
    minxy = moved.min(axis=0)
    maxxy = moved.max(axis=0)
    x0, y0 = [int(math.floor(c)) for c in minxy]
    x1, y1 = [int(math.ceil(c)) for c in maxxy]
    area = max(0, x1-x0)*max(0, y1-y0)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(width, x1), min(height, y1)
    clipped = max(0, x1-x0)*max(0, y1-y0)
    valid = (x1-x0>=8 and y1-y0>=8 and area>0 and clipped/area>=0.82)
    return ({'id':slot['id'], 'x':x0, 'y':y0, 'w':max(1,x1-x0), 'h':max(1,y1-y0)}, valid)


def predict_with_motion(detector, frame, slots, mat, alignment_valid):
    h, w = frame.shape[:2]
    moving = [transform_slot(s,mat,w,h) for s in slots]
    valid_slots = [s for s, valid in moving if valid and alignment_valid]
    predictions = {p['id']:p for p in detector.predict(frame,valid_slots)} if valid_slots else {}
    output = []
    for slot, valid in moving:
        outcome = predictions.get(slot['id'], dict(slot, status='uncertain'))
        if not valid or not alignment_valid:
            outcome = dict(slot, status='uncertain')
        output.append(outcome)
    return output


def run(video, detector_mode, max_seconds, output_fps, no_open, detector_override=None):
    video = base.abspath(video)
    paths = base.paths_for(video)
    cap, meta = base.open_video(video)
    try:
        plan = base.label_plan(meta['count'], 2, 2)
        if not paths['roi'].is_file():
            raise ValueError('ROI asli belum ditemukan untuk video ini. Jalankan dulu: '
                             'python scripts/test_new_parking_video.py video --video "PATH_VIDEO" '
                             '(gambar ROI dan tekan S).')
        cfg = json.loads(paths['roi'].read_text(encoding='utf-8'))
        # Respect original image dimensions and video identity; custom counts
        # are preserved in ROI file rather than hardcoding 2/2 when validating.
        cfg_plan = cfg.get('sampling_plan', plan)
        base.check_cfg(cfg, video, paths, meta, cfg_plan)
        size = (cfg['frame_width'],cfg['frame_height'])
        anchor_index = cfg_plan['calibration'][0]
        reference = base.frame_at(cap, anchor_index, size)
        tracker = CameraMotion(reference)
        name, detector = (detector_override if detector_override is not None
                          else base.select_video_detector(cap,cfg,paths,method=detector_mode))
        outdir = paths['out'];outdir.mkdir(parents=True,exist_ok=True)
        output = outdir/'annotated_video_motion_roi.mp4'
        fps=float(meta['fps'])
        step=max(1,int(round(fps/output_fps)))
        actual_fps=fps/step
        total=meta['count'] if max_seconds<=0 else min(meta['count'], int(math.ceil(max_seconds*fps)))
        writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),actual_fps,
                               (size[0],size[1]+90))
        if not writer.isOpened():
            raise RuntimeError(f'Cannot write MP4: {output}. Close Windows Media Player first.')
        stats={'ORB_OK':0,'PHASE_OK':0,'LOST':0}
        motion_rows=[]; pred_rows=[]; count=0
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        try:
            for idx in range(total):
                ok,frame=cap.read()
                if not ok:break
                if idx%step:continue
                if frame.shape[1::-1] != size:
                    frame=cv2.resize(frame,size,interpolation=cv2.INTER_AREA)
                if idx==anchor_index:
                    matrix=np.array([[1.,0.,0.],[0.,1.,0.]],dtype=np.float64)
                    tracking='ORB_OK';support=0;quality=1.
                else:
                    matrix,tracking,support,quality=tracker.estimate(frame)
                stats[tracking]+=1
                preds=predict_with_motion(detector,frame,cfg['slots'],matrix,tracking!='LOST')
                label=f'MOTION ROI {tracking} | dx={matrix[0,2]:+.0f} dy={matrix[1,2]:+.0f}px'
                annotated=base.annotate(frame,preds,idx,name,label)
                if tracking=='LOST':
                    cv2.putText(annotated,'TRACKING LOST - CLASSIFICATION PAUSED',
                                (10, min(annotated.shape[0]-10,110)),
                                cv2.FONT_HERSHEY_SIMPLEX,.55,(0,210,255),2,cv2.LINE_AA)
                writer.write(annotated)
                if count==0:cv2.imwrite(str(outdir/'preview_motion_roi.jpg'),annotated)
                motion_rows.append((idx,round(idx/fps,3),tracking,
                                    round(float(matrix[0,2]),2),round(float(matrix[1,2]),2),
                                    support,round(float(quality),4)))
                for p in preds:
                    pred_rows.append((idx,round(idx/fps,3),p['id'],p['status'],
                                      p['x'],p['y'],p['w'],p['h'],tracking))
                count+=1
                if count%100==0:print(f'Frames annotated: {count} / approximately {math.ceil(total/step)}')
        finally:
            writer.release()
        if count==0:raise RuntimeError('No output frames written.')
        base.verify_mp4(output)
        with (outdir/'camera_motion_tracking.csv').open('w',newline='',encoding='utf-8') as f:
            wr=csv.writer(f);wr.writerow(('frame','seconds','tracking','dx','dy','orb_inliers','confidence'));wr.writerows(motion_rows)
        with (outdir/'motion_roi_predictions.csv').open('w',newline='',encoding='utf-8') as f:
            wr=csv.writer(f);wr.writerow(('frame','seconds','slot_id','predicted_status','x','y','w','h','tracking'));wr.writerows(pred_rows)
        total_ok=stats['ORB_OK']+stats['PHASE_OK']
        print('\n'+'='*62)
        print(f'MOTION-AWARE ANNOTATED VIDEO: {output}')
        print(f'Processed: {count} frames, {len(cfg["slots"])} selected ROI, model={name}')
        print('Tracking reliability: '+' | '.join(f'{key}={val}' for key,val in stats.items()))
        print(f'Alignment coverage: {total_ok}/{count} = {total_ok/count:.1%}')
        print('NOTE: Alignment coverage is NOT occupancy classification accuracy.')
        print('If many LOST frames, use a steadier clip or consider camera-specific tracking.')
        print('If boxes drift, do NOT treat predictions as validated.')
        print('='*62)
        base.open_video_result(output,skip=no_open)
        return output,stats
    finally:
        cap.release()


def main():
    ap=argparse.ArgumentParser(description='Fix moving-camera ROIs on NEW parking videos; exports annotated MP4.')
    ap.add_argument('--pick',action='store_true',help='Choose original video using Windows file dialog')
    ap.add_argument('--video',help='Original source video (the one with saved ROIs)')
    ap.add_argument('--detector',default='auto',choices=('auto','scene_reference','cnrpark','v3','scene_knn'))
    ap.add_argument('--max-seconds',type=float,default=0)
    ap.add_argument('--output-fps',type=float,default=10)
    ap.add_argument('--no-open',action='store_true')
    args=ap.parse_args()
    if bool(args.pick)==bool(args.video):ap.error('Use exactly one of --pick or --video')
    if args.max_seconds<0 or args.output_fps<=0:ap.error('--max-seconds >=0 and --output-fps >0')
    video=base.choose_video() if args.pick else args.video
    try:run(video,args.detector,args.max_seconds,args.output_fps,args.no_open)
    except (ValueError,RuntimeError,OSError,cv2.error) as exc:raise SystemExit(f'ERROR: {exc}') from exc

if __name__=='__main__':main()
