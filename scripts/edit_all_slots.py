"""Mouse ROI editor for many parking slots; does not edit existing code or video.

Mouse: left drag creates rectangle; right click removes closest ROI.
Keyboard: S save, U undo, R switch rectangle/row mode, I toggle IDs, Q quit.
Row mode: set --row-count N (number of slots in each row). Two left clicks
          at the centers of first and last slots build N rectangles.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default='config/parking_9010418_all_slots_DRAFT.json')
    ap.add_argument('--save-as',default='config/parking_9010418_all_slots_REVIEWED.json')
    ap.add_argument('--video',default='dataset/parking_real_720p_stabilized.mp4')
    ap.add_argument('--time',type=float,default=10)
    ap.add_argument('--row-count',type=int,default=10)
    ap.add_argument('--slot-width',type=int,default=18)
    ap.add_argument('--slot-height',type=int,default=48)
    args=ap.parse_args()
    if args.row_count < 2:ap.error('row-count must be >=2')
    cfg=json.loads((ROOT/args.config).read_text(encoding='utf8'))
    slots=cfg['slots']
    cap=cv2.VideoCapture(str(ROOT/args.video));cap.set(cv2.CAP_PROP_POS_MSEC,args.time*1000)
    ok,frame=cap.read();cap.release()
    if not ok:raise RuntimeError('Could not read selected frame')
    if (cfg['frame_width'],cfg['frame_height'])!=(frame.shape[1],frame.shape[0]):
        raise ValueError('ROI config does not match video resolution')
    history=[]
    drag=None
    mode='rectangle'
    row_first=None
    show_ids=False
    next_index=1

    def snapshot():history.append(copy.deepcopy(slots))
    def new_id():
        nonlocal next_index
        used={s['id'] for s in slots}
        while f'NEW{next_index:03d}' in used:next_index+=1
        val=f'NEW{next_index:03d}';next_index+=1
        return val
    def clamp(slot):
        slot['x']=min(frame.shape[1]-1,max(0,int(slot['x'])))
        slot['y']=min(frame.shape[0]-1,max(0,int(slot['y'])))
        slot['w']=min(frame.shape[1]-slot['x'],max(1,int(slot['w'])))
        slot['h']=min(frame.shape[0]-slot['y'],max(1,int(slot['h'])))
        return slot

    def onmouse(event,x,y,flags,param):
        nonlocal drag,row_first
        if event == cv2.EVENT_RBUTTONDOWN and slots:
            def dist(s):
                cx=s['x']+s['w']/2;cy=s['y']+s['h']/2
                return (cx-x)**2+(cy-y)**2
            ind=min(range(len(slots)),key=lambda i:dist(slots[i]))
            if dist(slots[ind]) <= (50**2):
                snapshot();print('Removed', slots[ind]['id']);slots.pop(ind)
        if mode == 'rectangle':
            if event == cv2.EVENT_LBUTTONDOWN:drag=(x,y)
            if event == cv2.EVENT_LBUTTONUP and drag:
                x0,y0=drag;drag=None
                if abs(x-x0)>=5 and abs(y-y0)>=5:
                    snapshot();new=clamp(dict(id=new_id(),x=min(x,x0),y=min(y,y0),w=abs(x-x0),h=abs(y-y0)))
                    slots.append(new)
                    print('Added',new)
        elif mode == 'row' and event == cv2.EVENT_LBUTTONDOWN:
            if row_first is None:
                row_first=(x,y);print('Row first center',row_first)
            else:
                snapshot()
                x0,y0=row_first;row_first=None
                for i in range(args.row_count):
                    t=i/(args.row_count-1)
                    cx=int(round(x0+(x-x0)*t));cy=int(round(y0+(y-y0)*t))
                    new=clamp(dict(id=new_id(),x=cx-args.slot_width//2,
                                   y=cy-args.slot_height//2,w=args.slot_width,h=args.slot_height))
                    slots.append(new)
                print('Added',args.row_count,'row slots')

    cv2.namedWindow('SmartParking ROI Editor',cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('SmartParking ROI Editor',onmouse)
    print('Left-drag: ADD ROI; Right click: REMOVE nearest (within 50 px); U undo; S save; R row mode; I labels; Q quit')
    print('Row mode: click center of FIRST and LAST slot; this inserts',args.row_count,'ROI rectangles')
    while True:
        display=frame.copy()
        for s in slots:
            x,y,w,h=(s[k] for k in ('x','y','w','h'))
            cv2.rectangle(display,(x,y),(x+w,y+h),(0,220,255),1)
            if show_ids:
                cv2.putText(display,str(s['id']), (x,max(8,y+8)), cv2.FONT_HERSHEY_SIMPLEX,.23,(255,255,255),1,cv2.LINE_AA)
        cv2.rectangle(display,(0,0),(frame.shape[1],50),(22,30,40),-1)
        cv2.putText(display,f'ROI {len(slots)} | Mode: {mode.upper()} | S=SAVE U=UNDO R=MODE I=IDs Q=QUIT',
                    (12,32),cv2.FONT_HERSHEY_SIMPLEX,.61,(255,255,255),2,cv2.LINE_AA)
        cv2.imshow('SmartParking ROI Editor',display)
        k=cv2.waitKey(50)&255
        if k==ord('q') or k==27:break
        elif k==ord('r'):
            mode='row' if mode=='rectangle' else 'rectangle'
            row_first=None;print('Mode',mode)
        elif k==ord('i'):show_ids=not show_ids
        elif k==ord('u') and history:
            slots[:]=history.pop();print('Undo ->',len(slots))
        elif k==ord('s'):
            dest=ROOT/args.save_as;dest.parent.mkdir(exist_ok=True,parents=True)
            cfg['slots']=slots
            cfg['notes']='REVIEWED FILE: manually review at timestamps 1, 10, 20; saved with ROI editor. Check accuracy separately.'
            dest.write_text(json.dumps(cfg,indent=2),encoding='utf8')
            print('Saved',dest,'slots',len(slots))
    cv2.destroyAllWindows()
    print('Exited. Changes are only retained if you pressed S.')

if __name__=='__main__':main()
