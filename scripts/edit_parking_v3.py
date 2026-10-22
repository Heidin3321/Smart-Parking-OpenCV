"""Visual ROI editor for small 380x452 video using a 2x display.
Left drag adds box; right click removes nearest; M toggles MOVE mode where
left drag moves nearest box; U undo, S save, I toggle names, Q quit.
Saved config is separate from original draft.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='dataset/parking_v3_stabilized.mp4')
    p.add_argument('--config',default='config/parking_v3_slots_DRAFT.json')
    p.add_argument('--save-as',default='config/parking_v3_slots_REVIEWED.json')
    p.add_argument('--time',type=float,default=10.)
    a=p.parse_args();path=lambda v:(Path(v) if Path(v).is_absolute() else ROOT/v)
    data=json.loads(path(a.config).read_text(encoding='utf8')); slots=data['slots']
    cap=cv2.VideoCapture(str(path(a.source)));cap.set(cv2.CAP_PROP_POS_MSEC,a.time*1000)
    ok,frame=cap.read();cap.release()
    if not ok:raise RuntimeError('Unable to read video frame')
    h,w=frame.shape[:2]
    if (w,h)!=(data['frame_width'],data['frame_height']):raise ValueError('Wrong ROI resolution')
    scale=2;history=[];drag=None;move_index=None;show_ids=True;mode='add';nextid=1
    def undo_checkpoint():history.append(copy.deepcopy(slots))
    def new_id():
        nonlocal nextid
        used={s['id'] for s in slots}
        while f'NEW{nextid:02}' in used:nextid+=1
        name=f'NEW{nextid:02}';nextid+=1;return name
    def nearest(x,y):
        if not slots:return None
        i=min(range(len(slots)),key=lambda j:(slots[j]['x']+slots[j]['w']/2-x)**2+(slots[j]['y']+slots[j]['h']/2-y)**2)
        sx=slots[i]['x']+slots[i]['w']/2;sy=slots[i]['y']+slots[i]['h']/2
        return i if (sx-x)**2+(sy-y)**2<65**2 else None
    def mouse(event,X,Y,flags,param):
        nonlocal drag,move_index
        x=min(max(int(X/scale),0),w-1);y=min(max(int(Y/scale),0),h-1)
        if event==cv2.EVENT_RBUTTONDOWN:
            i=nearest(x,y)
            if i is not None:undo_checkpoint();print('Removed',slots.pop(i)['id'])
        elif event==cv2.EVENT_LBUTTONDOWN:
            if mode=='move':
                move_index=nearest(x,y)
                if move_index is not None: undo_checkpoint();drag=(x,y)
            else: drag=(x,y)
        elif event==cv2.EVENT_LBUTTONUP and drag is not None:
            x0,y0=drag;drag=None
            if mode=='move' and move_index is not None:
                s=slots[move_index];s['x']=max(0,min(w-s['w'],s['x']+x-x0));s['y']=max(0,min(h-s['h'],s['y']+y-y0));print('Moved',s['id']);move_index=None
            elif mode=='add' and abs(x-x0)>=6 and abs(y-y0)>=6:
                undo_checkpoint();slots.append({'id':new_id(),'x':min(x,x0),'y':min(y,y0),'w':abs(x-x0),'h':abs(y-y0)})
    window='Parking V3 ROI editor (display 2x)'
    cv2.namedWindow(window,cv2.WINDOW_NORMAL);cv2.resizeWindow(window,w*2,h*2+70)
    cv2.setMouseCallback(window,mouse)
    print('Left drag add/move | Right click remove | M move/add | U undo | S SAVE | I labels | Q quit')
    while True:
        screen=cv2.resize(frame,(w*scale,h*scale))
        for s in slots:
            x,y,bw,bh=(s[k] for k in ('x','y','w','h'))
            cv2.rectangle(screen,(x*scale,y*scale),((x+bw)*scale,(y+bh)*scale),(0,210,255),2)
            if show_ids:cv2.putText(screen,s['id'],(x*scale+3,y*scale+16),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
        label=f'ROI {len(slots)} | MODE {mode.upper()} | M=MODE  S=SAVE  U=UNDO  Q=QUIT'
        cv2.putText(screen,label,(7,25),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        cv2.imshow(window,screen)
        key=cv2.waitKey(50)&255
        if key in (ord('q'),27):break
        if key==ord('m'):mode='move' if mode=='add' else 'add';print('Mode:',mode)
        if key==ord('i'):show_ids=not show_ids
        if key==ord('u') and history:slots[:]=history.pop();print('Undo')
        if key==ord('s'):
            outfile=path(a.save_as);outfile.parent.mkdir(exist_ok=True,parents=True)
            data['slots']=slots;data['roi_status']='MANUALLY EDITED - verify all slots at multiple timestamps'
            outfile.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8')
            print('Saved',outfile)
    cv2.destroyAllWindows()
    print('Exited. Press S while window open to write REVIEWED JSON.')
if __name__=='__main__':main()
