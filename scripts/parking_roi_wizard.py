"""User-selectable Manual / Auto-Suggest / Reuse ROI launcher for NEW VIDEO.

Follows up with the existing camera-motion-aware classifier and saves an MP4.
Auto ROI is only a proposal based on parking-line geometry and must be reviewed.

 python scripts/parking_roi_wizard.py --pick
 python scripts/parking_roi_wizard.py --video "C:/Users/lenovo/Downloads/lot.mp4" --roi-mode auto
 python scripts/parking_roi_wizard.py --pick --roi-mode manual
 python scripts/parking_roi_wizard.py --pick --roi-mode existing
"""
from __future__ import annotations

import argparse
import json
import shutil
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import test_new_parking_video as base
import run_motion_aware_parking as motion
from auto_parking_rois import suggest_rois


def numbered(slots):
    return [{'id':f'N{i+1:03d}',**{k:int(slot[k]) for k in ('x','y','w','h')}}
            for i,slot in enumerate(slots)]


def choose_mode(has_saved):
    print('\n=== SMART PARKING — PILIH CARA MEMBUAT ROI ===')
    print('1. MANUAL  - gambar ROI sendiri, tekan ENTER/S untuk simpan')
    print('2. AUTO    - ROI dibuat otomatis (TANPA DRAG), tekan ENTER untuk setuju')
    if has_saved:
        print('3. EXISTING- ROI tersimpan, langsung lanjut LABELING')
    print('0. Batal')
    while True:
        value=input('KETIK ANGKA 1 atau 2'+(' atau 3' if has_saved else '')+' lalu ENTER: ').strip()
        choice={'1':'manual','2':'auto','3':'existing' if has_saved else None,'0':'cancel'}.get(value)
        if choice:
            return choice
        print('Pilihan tidak valid, coba lagi.')


def roi_review_window(frame,initial,mode,max_rois):
    """Draw new ROI: left drag; right-click remove; U undo; C clear; S save."""
    h,w=frame.shape[:2]
    title='PARKING ROI REVIEW - MANUAL/AUTO - S=SAVE'
    header=91
    slots=numbered(initial)
    state={'start':None,'cur':None,'history':[]}
    cv2.namedWindow(title,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title,max(640,w),h+header)

    def push():
        state['history'].append([dict(s) for s in slots])
        if len(state['history'])>50:state['history'].pop(0)

    def mouse(evt,x,y,flags,param):
        yy=y-header
        if evt==cv2.EVENT_LBUTTONDOWN and 0<=x<w and 0<=yy<h:
            state['start']=(x,yy);state['cur']=(x,yy)
        elif evt==cv2.EVENT_MOUSEMOVE and state['start'] is not None:
            state['cur']=(x,yy)
        elif evt==cv2.EVENT_LBUTTONUP and state['start'] is not None:
            sx,sy=state['start']
            x0,x1=sorted([max(0,min(w,sx)),max(0,min(w,x))])
            y0,y1=sorted([max(0,min(h,sy)),max(0,min(h,yy))])
            if x1-x0>=9 and y1-y0>=9:
                push();slots.append({'id':'TEMP','x':x0,'y':y0,'w':x1-x0,'h':y1-y0})
            state['start']=state['cur']=None
        elif evt==cv2.EVENT_RBUTTONDOWN and 0<=x<w and 0<=yy<h:
            targets=[i for i,s in enumerate(slots) if s['x']<=x<=s['x']+s['w'] and s['y']<=yy<=s['y']+s['h']]
            if targets:
                push(); slots.pop(targets[-1])

    cv2.setMouseCallback(title,mouse)
    try:
        while True:
            show=np.full((h+header,w,3),(25,30,40),np.uint8)
            show[header:]=frame
            base.put(show,f'{mode.upper()} ROI | {len(slots)} proposed/drawn  |  MAX SUGGEST {max_rois}',8,25,.52)
            base.put(show,'LEFT-DRAG add | RIGHT-CLICK delete | U undo | C clear',8,49,.43,(0,220,250))
            base.put(show,'ENTER/S accept ROI | LEFT-DRAG optional correction | Q cancel',8,74,.43,(0,220,250))
            for i,s in enumerate(slots):
                x,y,bw,bh=(s[k] for k in ('x','y','w','h'))
                cv2.rectangle(show,(x,y+header),(x+bw,y+bh+header),(0,210,245),2)
                base.put(show,f'N{i+1:03d}',x+2,y+header+15,.4,(0,210,245))
            if state['start'] and state['cur']:
                a,b=state['start'],state['cur']
                cv2.rectangle(show,(a[0],a[1]+header),(b[0],b[1]+header),(245,245,245),2)
            cv2.imshow(title,show)
            key=cv2.waitKey(30)&0xff
            if key in (ord('s'), 13, 10, 32, ord('a')):
                if slots:return numbered(slots)
                print('Draw at least one ROI or choose AUTO. Empty ROI list cannot be saved.')
            elif key==ord('u') and state['history']:
                slots[:]=state['history'].pop()
            elif key==ord('c'):
                push();slots.clear()
            elif key in (ord('q'),27):
                raise RuntimeError('Canceled by user. Existing saved ROI has not changed.')
    finally:
        cv2.destroyWindow(title)


def _backup_labels(paths):
    old=paths['labels']
    if not old.exists():return
    backup=old.with_suffix('.json.bak')
    i=1
    while backup.exists():
        backup=old.with_suffix(f'.json.bak{i}');i+=1
    shutil.copy2(old,backup)
    old.unlink()
    print('Old labels backed up because ROI coordinates/IDs changed:',backup)


def save_roi(paths,meta,plan,slots,mode,diagnostic=None):
    w,h=base.display_dimensions(meta)
    cfg={'frame_width':w,'frame_height':h,
         'source_video':f'{meta["width"]}x{meta["height"]}',
         'fingerprint':paths['fingerprint'],'sampling_plan':plan,
         'annotation':f'{mode}: all ROI proposals require user acceptance before use',
         'roi_creation_mode':mode,'slots':numbered(slots)}
    if diagnostic is not None:cfg['auto_roi_diagnostic']=diagnostic
    base.assert_valid_roi(cfg)
    if paths['roi'].exists():
        old=json.loads(paths['roi'].read_text(encoding='utf-8'))
        # Preserve labels if ROI AND frame sampling plan remain byte-identical.
        changed=(base.roi_signature(cfg)!=base.roi_signature(old))
    else:
        changed=True
    if changed:
        _backup_labels(paths)
    base.save_json(paths['roi'],cfg)
    print(f'Saved {len(slots)} {mode.upper()} ROI:',paths['roi'])
    return cfg


def configure(video,roi_mode='choose',max_rois=15,columns=None,auto_accept=False):
    video=base.abspath(video)
    paths=base.paths_for(video)
    cap,meta=base.open_video(video)
    try:
        plan=base.label_plan(meta['count'],2,2)
        has_existing=paths['roi'].is_file()
        mode=choose_mode(has_existing) if roi_mode=='choose' else roi_mode
        if mode=='cancel':raise RuntimeError('Canceled by user.')
        if mode=='existing':
            if not has_existing:raise ValueError('No saved ROIs for this video. Use manual or auto first.')
            cfg=json.loads(paths['roi'].read_text(encoding='utf-8'))
            base.check_cfg(cfg,video,paths,meta,cfg.get('sampling_plan',plan))
            print(f'Reusing {len(cfg["slots"])} saved ROIs:',paths['roi'])
            return cfg
        size=base.display_dimensions(meta)
        image=base.frame_at(cap,plan['calibration'][0],size)
        proposed=[]; diagnostic=None
        if mode=='auto':
            proposed,diagnostic=suggest_rois(image,max_rois=max_rois,columns=columns)
            print('AUTO ROI suggestions:',len(proposed),'| Diagnostics:',json.dumps(diagnostic))
            if not proposed:
                print('WARNING: Painted parking lines not detected confidently. Draw ROIs manually in the review window.')
            else:
                print('WARNING: ROI positions are only algorithmic suggestions; verify every slot!')
        if mode == 'auto' and not proposed:
            raise ValueError('AUTO ROI belum bisa menemukan marka parkir. Coba mode 1 MANUAL atau video lain.')
        if mode == 'auto':
            print(f'\nAUTO MEMBUAT {len(proposed)} ROI. TIDAK PERLU DRAG!')
            print('Tekan ENTER pada jendela preview untuk setuju; atau gunakan --auto-accept untuk langsung lanjut.')
            out=paths['out'];out.mkdir(parents=True,exist_ok=True)
            preview=image.copy()
            for s in proposed:
                x,y,sw,sh=[s[k] for k in ('x','y','w','h')]
                cv2.rectangle(preview,(x,y),(x+sw,y+sh),(0,220,255),2)
                base.put(preview,s['id'],x+2,y+14,.45,(0,220,255))
            cv2.imwrite(str(out/'auto_roi_preview.jpg'),preview)
            print('AUTO ROI preview saved:',out/'auto_roi_preview.jpg')
        slots=(proposed if (mode=='auto' and auto_accept)
               else roi_review_window(image,proposed,mode,max_rois))
        return save_roi(paths,meta,plan,slots,mode,diagnostic)
    finally:
        cap.release()


def main(argv=None):
    ap=argparse.ArgumentParser(description='Manual/Auto/Existing ROI -> MANDATORY human labels -> train/test -> motion-aware annotated VIDEO.')
    group=ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--pick',action='store_true',help='Pick original video using Windows file dialog.')
    group.add_argument('--video',help='Original parking video path (not previously annotated video).')
    ap.add_argument('--roi-mode',default='choose',choices=('choose','manual','auto','existing'),
                    help='If omitted, choose 1 MANUAL / 2 AUTO / 3 EXISTING in terminal.')
    ap.add_argument('--auto-accept',action='store_true',
                    help='When --roi-mode auto, generate automatic ROI and continue WITHOUT drawing or GUI review. Inspect auto_roi_preview.jpg later.')
    ap.add_argument('--max-rois',type=int,default=15,help='Maximum AUTO suggestions, default 15.')
    ap.add_argument('--columns',type=int,choices=[1,2,3,4],help='Hint: number of parking columns in top-down video.')
    ap.add_argument('--detector',default='auto',choices=('auto','scene_reference','cnrpark','scene_knn'),
                    help='Classifier used for annotated video after calibrating.')
    ap.add_argument('--max-seconds',type=float,default=0)
    ap.add_argument('--output-fps',type=float,default=10)
    ap.add_argument('--no-open',action='store_true')
    args=ap.parse_args(argv)
    if not (1<=args.max_rois<=100):ap.error('--max-rois must be from 1 to 100')
    if args.max_seconds<0 or args.output_fps<=0:ap.error('Invalid time/FPS')
    if args.auto_accept and args.roi_mode!='auto':
        ap.error('--auto-accept requires --roi-mode auto')
    try:
        video=base.choose_video() if args.pick else base.abspath(args.video)
        print('\nSOURCE VIDEO:',video)
        cfg=configure(video,args.roi_mode,args.max_rois,args.columns,args.auto_accept)
        paths=base.paths_for(video)
        import video_label_flow
        print(f'\nROI SELECTION COMPLETE: {len(cfg["slots"])} slots. NEXT: HUMAN LABELING is REQUIRED!')
        video_label_flow.execute(video,cfg,paths,None,args.detector,args.max_seconds,args.output_fps,args.no_open)
    except (ValueError,RuntimeError,OSError,cv2.error) as exc:
        raise SystemExit(f'ERROR: {exc}') from exc

if __name__=='__main__':main()
