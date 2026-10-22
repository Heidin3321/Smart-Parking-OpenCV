"""Real video parking-v3 demo using existing team KNN and ROI contract.

Predictions are preliminary. Use manually annotated ground-truth to measure
accuracy; the CSV is NOT itself ground truth.
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from main import read_slots
from detector import ParkingDetector, extract_features, CLASS_NAMES
from preprocessing import preprocess_frame


def resolve(s: str) -> Path:
    p=Path(s)
    return p if p.is_absolute() else ROOT/p


def batch_predict(detector: ParkingDetector, frame: np.ndarray, slots: list[dict]) -> list[dict]:
    # Feature computation follows the SAME preprocessing as model training.
    vectors=[extract_features(frame[s['y']:s['y']+s['h'], s['x']:s['x']+s['w']]) for s in slots]
    matrix=np.stack(vectors).astype(np.float32)
    _, labels, _, _=detector.model.findNearest(matrix,k=detector.k)
    return [dict(slot,status=CLASS_NAMES[int(pred)]) for slot,pred in zip(slots, labels[:,0])]


def annotate(frame: np.ndarray, predictions: list[dict], frame_index: int, show_ids=True, scale=2):
    """Header placed ABOVE frame so top row of cars never gets occluded."""
    height,width=frame.shape[:2]
    enlarged=cv2.resize(frame,(width*scale,height*scale),interpolation=cv2.INTER_LINEAR)
    panel_height=105 if scale==2 else 55
    canvas=np.zeros((enlarged.shape[0]+panel_height,enlarged.shape[1],3),np.uint8)
    canvas[:]=(28,36,45)
    canvas[panel_height:]=enlarged
    for s in predictions:
        x,y,w,h=[int(s[k]) for k in ('x','y','w','h')]
        color=(70,200,75) if s['status']=='empty' else (75,85,235)
        x0,y0,x1,y1=x*scale,y*scale+panel_height,(x+w)*scale,(y+h)*scale+panel_height
        cv2.rectangle(canvas,(x0,y0),(x1,y1),color,2)
        if show_ids:
            cv2.rectangle(canvas,(x0,y0),(x0+44,y0+21),color,-1)
            cv2.putText(canvas,s['id'],(x0+3,y0+16),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
    empty=sum(s['status']=='empty' for s in predictions)
    occupied=len(predictions)-empty
    cv2.putText(canvas,'SMART PARKING V3 | V3-ADAPTED KNN',(14,28),cv2.FONT_HERSHEY_SIMPLEX,.72,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(canvas,f'EMPTY: {empty}    OCCUPIED: {occupied}    TOTAL ROI: {len(predictions)}',
                (14,59),cv2.FONT_HERSHEY_SIMPLEX,.57,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(canvas,f'Frame {frame_index} | Demo model; independent holdout metrics separate',
                (14,86),cv2.FONT_HERSHEY_SIMPLEX,.45,(75,223,255),1,cv2.LINE_AA)
    return canvas


def run(argv=None):
    parser=argparse.ArgumentParser(description='Parking V3 occupancy KNN preview (real video, unvalidated status)')
    parser.add_argument('--source',default='dataset/parking_v3_stabilized.mp4')
    parser.add_argument('--config',default='config/parking_v3_slots_REVIEWED.json')
    parser.add_argument('--train-dir',default='dataset/v3_calibrated/deploy_train')
    parser.add_argument('--output-dir',default='results/parking_v3_adapted')
    parser.add_argument('--max-frames',type=int,default=0,help='Max source frames, 0=all')
    parser.add_argument('--step',type=int,default=1,help='Process every Nth source frame')
    parser.add_argument('--save-video',action='store_true')
    parser.add_argument('--show',action='store_true')
    parser.add_argument('--hide-ids',action='store_true')
    a=parser.parse_args(argv)
    if a.step<1 or a.max_frames<0:parser.error('--step must be >=1 and --max-frames >=0')
    src,conf,train,out=[resolve(s) for s in [a.source,a.config,a.train_dir,a.output_dir]]
    if not src.is_file():parser.error(f'Input video is missing: {src}')
    info=read_slots(conf)
    slots=info['slots']
    cap=cv2.VideoCapture(str(src))
    if not cap.isOpened():raise RuntimeError(f'OpenCV cannot read {src}')
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    if not 1<=fps<=120:fps=10.
    writer=None; processed=0; frame_index=0; rows_written=0
    out.mkdir(parents=True,exist_ok=True)
    try:
        detector=ParkingDetector(train,k=3)
        with (out/'predictions.csv').open('w',newline='',encoding='utf-8') as output:
            rows=csv.writer(output)
            rows.writerow(['frame','slot_id','predicted_status'])
            while True:
                if a.max_frames and frame_index>=a.max_frames:break
                ok,frame=cap.read()
                if not ok:break
                if frame_index%a.step:
                    frame_index+=1
                    continue
                if (frame.shape[1],frame.shape[0])!=(info['frame_width'],info['frame_height']):
                    raise ValueError(f'ROI config {info["frame_width"]}x{info["frame_height"]} does not match video {frame.shape[1]}x{frame.shape[0]}')
                proc=preprocess_frame(frame)
                if proc.shape!=frame.shape:raise ValueError('preprocessing changed frame shape')
                predictions=batch_predict(detector,proc,slots)
                for p in predictions:
                    rows.writerow((frame_index,p['id'],p['status']))
                    rows_written+=1
                rendered=annotate(frame,predictions,frame_index,show_ids=not a.hide_ids)
                if processed==0:cv2.imwrite(str(out/'first_frame.jpg'),rendered)
                if a.save_video:
                    if writer is None:
                        destination=out/'annotated_video.mp4'
                        hh,ww=rendered.shape[:2]
                        writer=cv2.VideoWriter(str(destination),cv2.VideoWriter_fourcc(*'mp4v'),fps/a.step,(ww,hh))
                        if not writer.isOpened():raise RuntimeError('Cannot write output mp4')
                    writer.write(rendered)
                if a.show:
                    cv2.imshow('Smart Parking V3 - preliminary predictions',rendered)
                    if cv2.waitKey(max(1,int(1000*a.step/fps)))&255==ord('q'):
                        processed+=1;break
                processed+=1;frame_index+=1
    finally:
        cap.release()
        if writer is not None:writer.release()
        if a.show:cv2.destroyAllWindows()
    if processed==0:raise RuntimeError('No frames processed')
    print(f'Processed {processed} output frames, {len(slots)} ROI per frame, {rows_written} preliminary predictions')
    print(f'Output: {out}')
    print('IMPORTANT: All-slot demo training includes labelled V3 crops, including evaluation-holdout slots; do NOT use this video for accuracy metrics. Run v3_adapt.py evaluate for proper held-out evaluation.')
    return {'frames':processed,'roi':len(slots),'predictions':rows_written}

if __name__=='__main__':run()
