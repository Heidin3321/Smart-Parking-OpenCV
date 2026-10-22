"""Expanded Smart Parking occupancy run for rectangular ROI configuration.

Uses the SAME feature extraction and KNN implementation as detector.py,
but sends all slot feature vectors to cv2.ml.KNearest.findNearest in one batch.
No detection accuracy is claimed until hand-labelled video evaluation.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from detector import ParkingDetector, CLASS_NAMES, extract_features  # noqa
from main import read_slots  # existing validated ROI contract
from preprocessing import preprocess_frame


def rotate_roi(patch: np.ndarray, degrees: float) -> np.ndarray:
    """Turn sideways/diagonal slot crops toward upright for the feature vector."""
    if degrees == 90:
        return cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
    if degrees == -90:
        return cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if degrees == 180:
        return cv2.rotate(patch, cv2.ROTATE_180)
    if not degrees:
        return patch
    h, w = patch.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(patch, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def predict_batch(model: ParkingDetector, frame: np.ndarray, slots: list[dict]) -> list[dict]:
    """Batch inference avoids thousands of repeated findNearest() Python calls."""
    features = []
    height, width = frame.shape[:2]
    for s in slots:
        x, y, w, h = [s[k] for k in ('x', 'y', 'w', 'h')]
        if x < 0 or y < 0 or x + w > width or y + h > height:
            raise ValueError(f'Out of bounds: {s["id"]}')
        roi = rotate_roi(frame[y:y+h, x:x+w], float(s.get('rotation_deg', 0)))
        features.append(extract_features(roi))
    vectors = np.stack(features).astype(np.float32)
    _, labels, _, _ = model.model.findNearest(vectors, k=model.k)
    return [dict(s, status=CLASS_NAMES[int(pred)]) for s, pred in zip(slots, labels[:, 0])]


def draw(frame: np.ndarray, results: list[dict], fid: int,
         verified=False, show_ids=False, training_domain="unknown") -> np.ndarray:
    out = frame.copy()
    for result in results:
        x, y, w, h = [result[k] for k in ('x', 'y', 'w', 'h')]
        color = (40, 215, 65) if result['status'] == 'empty' else (40, 35, 230)
        cv2.rectangle(out, (x, y), (x+w, y+h), color, 1)
        if show_ids:
            cv2.putText(out, result['id'], (x, max(y+9, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, .24, (255,255,255), 1, cv2.LINE_AA)
    empty = sum(s['status'] == 'empty' for s in results)
    occupied = len(results) - empty
    cv2.rectangle(out, (0,0),(out.shape[1],77),(27,36,47),-1)
    cv2.putText(out, f'SMART PARKING | ROI: {len(results)} | EMPTY {empty} | OCCUPIED {occupied}',
                (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .67, (255,255,255), 2, cv2.LINE_AA)
    foot = ('MANUALLY VERIFIED ROI' if verified else 'DRAFT ROI (REVIEW REQUIRED)')
    cv2.putText(out, f'Frame {fid} | TRAIN: {training_domain.upper()} KNN | {foot} | Predictions NOT validated',
                (12, 58), cv2.FONT_HERSHEY_SIMPLEX, .43, (66,215,249), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser(description='All-slot KNN trial for stabilized Pexels footage')
    ap.add_argument('--source', default='dataset/parking_real_720p_stabilized.mp4')
    ap.add_argument('--config', default='config/parking_9010418_all_slots_DRAFT.json')
    ap.add_argument('--train-dir', default='dataset/real_cnrpark/train')
    ap.add_argument('--training-domain',choices=('real','synthetic','unknown'),default='real')
    ap.add_argument('--output-dir', default='results/all_slots')
    ap.add_argument('--max-frames', type=int, default=0)
    ap.add_argument('--step', type=int, default=2, help='Process every Nth frame; defaults to 2 for speed')
    ap.add_argument('--show-ids', action='store_true')
    ap.add_argument('--save-video', action='store_true')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--verified-roi', action='store_true', help='Set ONLY after manually reviewing every ROI')
    args=ap.parse_args()
    if args.step < 1 or args.max_frames < 0: ap.error('step>=1, max-frames>=0 required')
    source=PROJECT/args.source if not Path(args.source).is_absolute() else Path(args.source)
    config=PROJECT/args.config if not Path(args.config).is_absolute() else Path(args.config)
    train=PROJECT/args.train_dir if not Path(args.train_dir).is_absolute() else Path(args.train_dir)
    out=PROJECT/args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    if not source.is_file(): ap.error(f'Video missing: {source}')
    slots_info=read_slots(config)
    slots=slots_info['slots']
    print(f'ROI loaded: {len(slots)} CANDIDATE slots; DRAFT does NOT mean exact parking capacity')
    detector=ParkingDetector(train, k=3)
    cap=cv2.VideoCapture(str(source))
    if not cap.isOpened(): raise RuntimeError(f'Cannot open {source}')
    fps=cap.get(cv2.CAP_PROP_FPS) or 10
    out.mkdir(parents=True,exist_ok=True)
    writer=None
    row_count=0; frame_idx=0; video_frames=0
    rows_path=out/'predictions.csv'
    try:
        with rows_path.open('w',newline='',encoding='utf-8') as fp:
            csvout=csv.writer(fp);csvout.writerow(['frame','slot_id','predicted_status'])
            while True:
                ok,frame=cap.read()
                if not ok or (args.max_frames and frame_idx>=args.max_frames): break
                if frame_idx % args.step:
                    frame_idx+=1
                    continue
                if (frame.shape[1],frame.shape[0]) != (slots_info['frame_width'],slots_info['frame_height']):
                    raise ValueError('Video resolution does not match ROI config')
                pre=preprocess_frame(frame)
                preds=predict_batch(detector,pre,slots)
                for s in preds:
                    csvout.writerow([frame_idx,s['id'],s['status']]); row_count+=1
                rendered=draw(frame,preds,frame_idx,args.verified_roi,args.show_ids,args.training_domain)
                if not video_frames:
                    cv2.imwrite(str(out/'first_frame.jpg'),rendered)
                if args.save_video:
                    if writer is None:
                        fourcc=cv2.VideoWriter_fourcc(*'mp4v')
                        writer=cv2.VideoWriter(str(out/'annotated_video.mp4'),fourcc,fps/args.step,(frame.shape[1],frame.shape[0]))
                        if not writer.isOpened(): raise RuntimeError('Video writer failed')
                    writer.write(rendered)
                if args.show:
                    cv2.imshow('Smart Parking All Slots — DRAFT',rendered)
                    if cv2.waitKey(max(1,int(1000*args.step/fps)))&255==ord('q'): break
                video_frames+=1
                frame_idx+=1
    finally:
        cap.release()
        if writer:writer.release()
        if args.show:cv2.destroyAllWindows()
    print(f'Processed {video_frames} output frames and {row_count} raw predictions.')
    print(f'Results: {out}')
    print('WARNING: Candidate ROI positions and predictions have NOT been fully validated.')

if __name__=='__main__':main()
