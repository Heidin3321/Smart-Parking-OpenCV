"""Draw ROI rectangles only (not occupancy predictions) for 0, 10, 20 seconds."""
import argparse
import sys
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from main import read_slots

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='dataset/parking_v3_stabilized.mp4')
    p.add_argument('--config',default='config/parking_v3_slots_DRAFT.json')
    p.add_argument('--output-dir',default='results/parking_v3/roi_review')
    p.add_argument('--times',default='0,10,20',help='Comma separated timestamps in seconds')
    a=p.parse_args()
    root=lambda x:(Path(x) if Path(x).is_absolute() else ROOT/x)
    src=root(a.source);info=read_slots(root(a.config));out=root(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(src))
    if not cap.isOpened():raise FileNotFoundError(src)
    try:
        for v in a.times.split(','):
            sec=float(v.strip());cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000);ok,frame=cap.read()
            if not ok:raise ValueError(f'No frame at {sec}s')
            if (frame.shape[1],frame.shape[0])!=(info['frame_width'],info['frame_height']):raise ValueError('ROI resolution mismatch')
            canvas=cv2.resize(frame,(frame.shape[1]*2,frame.shape[0]*2))
            for slot in info['slots']:
                x,y,w,h=(slot[k] for k in ('x','y','w','h'))
                cv2.rectangle(canvas,(x*2,y*2),((x+w)*2,(y+h)*2),(0,210,255),2)
                cv2.putText(canvas,slot['id'],(x*2+3,y*2+17),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1,cv2.LINE_AA)
            dest=out/f'roi_{sec:g}s.jpg';cv2.imwrite(str(dest),canvas);print('Preview:',dest)
    finally:cap.release()

if __name__=='__main__':main()
