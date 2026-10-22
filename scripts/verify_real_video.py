"""Verify sample stabilized video ROI geometry, not occupancy accuracy."""
from pathlib import Path
import argparse
import cv2
import json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='dataset/parking_real_720p_stabilized.mp4')
    p.add_argument('--config',default='config/parking_9010418_roi.json')
    p.add_argument('--output',default='results/real_video_9010418/roi_preview.jpg')
    p.add_argument('--time',type=float,default=10.0)
    args=p.parse_args()
    config=json.loads(Path(args.config).read_text(encoding='utf8'))
    cap=cv2.VideoCapture(args.source)
    if not cap.isOpened(): raise SystemExit(f'ERROR: Missing/unreadable video {args.source}')
    cap.set(cv2.CAP_PROP_POS_MSEC,args.time*1000)
    success,frame=cap.read()
    cap.release()
    if not success:raise SystemExit(f'ERROR: No video frame at {args.time} seconds')
    h,w=frame.shape[:2]
    assert (w,h)==(config['frame_width'],config['frame_height']), 'ROI resolution mismatch'
    for slot in config['slots']:
        x,y,dx,dy=[slot[k] for k in ('x','y','w','h')]
        assert 0 <= x < x+dx <=w and 0<=y<y+dy<=h, f'Invalid ROI: {slot}'
        cv2.rectangle(frame,(x,y),(x+dx,y+dy),(0,220,255),2)
        cv2.putText(frame,slot['id'],(x-3,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.38,(255,255,255),1,cv2.LINE_AA)
    # Preview boxes are yellow: no guessed detections.
    cv2.rectangle(frame,(0,0),(w,60),(18,29,38),-1)
    cv2.putText(frame,'ROI ALIGNMENT PREVIEW - NO PREDICTIONS',(14,29),cv2.FONT_HERSHEY_SIMPLEX,0.68,(255,255,255),2)
    cv2.putText(frame,f'ROI={len(config["slots"])} | t={args.time:.1f}s | VIDEO STABILIZED',(14,51),cv2.FONT_HERSHEY_SIMPLEX,0.43,(0,220,255),1)
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    assert cv2.imwrite(str(out),frame)
    print(f'Success: {len(config["slots"])} ROI fits {w}x{h} at {args.time:.1f}s; preview saved: {out}')

if __name__=='__main__':main()
