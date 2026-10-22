"""Preview the candidate ROI geometry at 1, 10, 20 s without classifying slots."""
from pathlib import Path
import argparse
import sys
import cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from main import read_slots

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',default='dataset/parking_real_720p_stabilized.mp4')
    parser.add_argument('--config',default='config/parking_9010418_all_slots_DRAFT.json')
    parser.add_argument('--output-dir',default='results/all_slots/roi_review')
    parser.add_argument('--times',nargs='+',type=float,default=[1,10,20])
    args=parser.parse_args()
    config=read_slots(ROOT/args.config)
    cap=cv2.VideoCapture(str(ROOT/args.source))
    if not cap.isOpened(): raise RuntimeError('Cannot read video')
    out=ROOT/args.output_dir;out.mkdir(parents=True,exist_ok=True)
    for sec in args.times:
        cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        ok,frame=cap.read()
        if not ok:
            print('Frame not found',sec);continue
        for s in config['slots']:
            x,y,w,h=(s[k] for k in ('x','y','w','h'))
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,210,255),1)
        cv2.rectangle(frame,(0,0),(frame.shape[1],62),(31,37,46),-1)
        cv2.putText(frame,f'ROI DRAFT: {len(config["slots"])} candidates | t={sec:g}s | NOT VALIDATED',
                   (12,38),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA)
        target=out/f'roi_{sec:g}s.jpg';cv2.imwrite(str(target),frame)
        print(target)
    cap.release()

if __name__=='__main__':main()
