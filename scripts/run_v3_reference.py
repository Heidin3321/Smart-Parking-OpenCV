"""Scene-specific OpenCV parking ROI reference matcher.

Uses ALL reviewed, manually labelled video crops per slot as reference images.
Does not generalize to new camera viewpoints without labelling new footage.
DO NOT use this in-scene demo to claim independent accuracy; use separate held-out evaluation.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict, deque
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
VALID={'empty','occupied'}
COLORS={'occupied':(68,74,235),'empty':(69,192,71),'uncertain':(0,214,255)}


def resolve(p):
    p=Path(p)
    return p if p.is_absolute() else ROOT/p


def roi_feature(patch):
    if patch is None or patch.size==0: raise ValueError('empty patch')
    # Mild blur and fixed-size spatial texture/color preserve the identity of cars,
    # unlike old, aggressively downsampled generic KNN.
    p=cv2.resize(patch,(48,24),interpolation=cv2.INTER_AREA)
    p=cv2.GaussianBlur(p,(3,3),0)
    return p.astype(np.float32)


def pixel_difference(a,b):
    return float(np.mean(np.abs(a-b)))


def parse_config(path):
    if not path.is_file(): raise FileNotFoundError(f'ROI JSON not found: {path}')
    d=json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(d.get('slots'),list) or not d['slots']:
        raise ValueError('ROI JSON must have a nonempty slots list')
    ids=set()
    for s in d['slots']:
        k=s.get('id');
        if not k or k in ids: raise ValueError(f'Duplicate/missing slot ID {k}')
        ids.add(k)
        x,y,w,h=(int(s[z]) for z in ('x','y','w','h'))
        if min(x,y)<0 or min(w,h)<=0 or x+w>int(d['frame_width']) or y+h>int(d['frame_height']):
            raise ValueError(f'ROI outside configured frame: {s}')
    return d


def load_labels(path, info):
    if not path.is_file():
        raise FileNotFoundError(f'Manual labels missing: {path}\nRun: python scripts/v3_adapt.py label')
    lookup={s['id']:s for s in info['slots']}
    grouped=defaultdict(list)
    with path.open(newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    duplicates=set()
    for r in rows:
        status=r.get('true_status','').strip().lower()
        if status not in VALID: continue  # unknown labels are never training examples
        sid=r.get('slot_id','')
        if sid not in lookup: raise ValueError(f'Label contains unknown slot {sid}')
        index=int(r['frame'])
        k=(index,sid)
        if k in duplicates: raise ValueError(f'Duplicate reference label {k}')
        duplicates.add(k)
        for name in ('x','y','w','h'):
            if int(r[name])!=int(lookup[sid][name]):
                raise ValueError(f'ROI for {sid} changed after labelling; labels are stale. Review before using.')
        grouped[index].append((sid,status,lookup[sid]))
    if not grouped: raise ValueError('No manually labelled empty/occupied images in CSV')
    return grouped, Counter(r['true_status'].strip().lower() for r in rows)


def get_reference_data(cap, info, labels_by_frame):
    gallery=defaultdict(list)
    max_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for fi, entries in sorted(labels_by_frame.items()):
        if fi<0 or (max_frame and fi>=max_frame):
            raise ValueError(f'Labelled frame {fi} not in video (total {max_frame})')
        cap.set(cv2.CAP_PROP_POS_FRAMES,fi)
        ok,fr=cap.read()
        if not ok: raise RuntimeError(f'Could not read manually labelled frame {fi}')
        if fr.shape[:2]!=(info['frame_height'],info['frame_width']):
            raise ValueError('Labelled video and ROI frame dimension mismatch')
        for sid,status,s in entries:
            x,y,w,h=(int(s[k]) for k in ('x','y','w','h'))
            gallery[sid].append((fi,status,roi_feature(fr[y:y+h,x:x+w])))
    for s in info['slots']:
        if s['id'] not in gallery:
            print(f'WARNING: no manual reference for {s["id"]}; will be marked UNCERTAIN.')
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    return gallery


def reference_threshold(refs):
    # Based on within-class reference changes, with a conservative upper cap.
    distances=[]
    for i,(_,cl,feat) in enumerate(refs):
        for _,cl2,feat2 in refs[i+1:]:
            if cl==cl2: distances.append(pixel_difference(feat,feat2))
    if not distances:return 12.0
    return max(12.0,min(24.0,float(np.percentile(distances,90))*2.5))


class SlotReferenceDetector:
    def __init__(self,gallery):
        self.gallery=gallery
        self.limits={s:reference_threshold(r) for s,r in gallery.items()}

    def classify(self,slot_id,patch):
        refs=self.gallery.get(slot_id,[])
        if not refs:return ('uncertain',None,float('nan'),'missing_manual_references')
        feature=roi_feature(patch)
        ordered=sorted((pixel_difference(feature,p),f,cl) for f,cl,p in refs)
        d,near_frame,pred=ordered[0]
        threshold=self.limits[slot_id]
        if d>threshold:
            return ('uncertain',near_frame,d,f'no_similar_reference_d>{threshold:.1f}')
        return (pred,near_frame,d,'slot_specific_reference')


def draw(frame,predictions,index,show_ids=True):
    h,w=frame.shape[:2]
    scale=2 if w<720 else 1
    bigger=cv2.resize(frame,(w*scale,h*scale),interpolation=cv2.INTER_LINEAR)
    ph=104 if scale==2 else 80
    out=np.full((bigger.shape[0]+ph,bigger.shape[1],3),(27,31,40),dtype=np.uint8)
    out[ph:]=bigger
    counts=Counter(p['status'] for p in predictions)
    tscale=.68 if scale==2 else .5
    cv2.putText(out,'SMART PARKING V3 | PER-SLOT REFERENCE MATCHING',(12,30),cv2.FONT_HERSHEY_SIMPLEX,tscale,(255,255,255),2,cv2.LINE_AA)
    line=f'EMPTY: {counts["empty"]}   OCCUPIED: {counts["occupied"]}   UNCERTAIN: {counts["uncertain"]}   ROI: {len(predictions)}'
    cv2.putText(out,line,(12,59),cv2.FONT_HERSHEY_SIMPLEX,.56 if scale==2 else .45,(255,255,255),1,cv2.LINE_AA)
    cv2.putText(out,f'Frame {index} | CALIBRATED ON THIS VIDEO - NOT INDEPENDENT ACCURACY',(12,86 if scale==2 else 73),cv2.FONT_HERSHEY_SIMPLEX,.43 if scale==2 else .31,(0,235,255),1,cv2.LINE_AA)
    for p in predictions:
        x,y,w0,h0=map(int,[p[k] for k in ('x','y','w','h')])
        x,y,w0,h0=x*scale,y*scale+ph,w0*scale,h0*scale
        col=COLORS[p['status']]
        cv2.rectangle(out,(x,y),(x+w0,y+h0),col,2)
        if show_ids:
            cv2.rectangle(out,(x,y),(x+52,y+22),col,-1)
            cv2.putText(out,p['id'],(x+4,y+17),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
    return out


def audit(out,info,gallery):
    out.mkdir(parents=True,exist_ok=True)
    count=Counter(cl for rs in gallery.values() for _,cl,_ in rs)
    per_slot={s['id']:{'count':len(gallery.get(s['id'],[])),
                       'classes':dict(Counter(cl for _,cl,_ in gallery.get(s['id'],[]))),
                       'threshold':round(float(reference_threshold(gallery[s['id']])),2) if gallery.get(s['id']) else None}
              for s in info['slots']}
    report={'manual_reference_count':sum(len(v) for v in gallery.values()),'labels':dict(count),'per_slot':per_slot,
            'note':'Manual labels must be verified; scene-specific gallery is unsuitable as independent holdout evaluation.'}
    (out/'reference_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    # A readable contact sheet to spot mistaken HUMAN labels, not to infer new labels.
    frame_ids=sorted({fi for rs in gallery.values() for fi,_,_ in rs})
    col_w,row_h,left,top=148,94,80,42
    sheet=np.full((top+len(info['slots'])*row_h, left+len(frame_ids)*col_w,3), (30,34,41),dtype=np.uint8)
    for j,fi in enumerate(frame_ids):
        cv2.putText(sheet,f'F{fi}',(left+j*col_w+11,27),cv2.FONT_HERSHEY_SIMPLEX,.55,(240,240,240),1,cv2.LINE_AA)
    for i,slot in enumerate(info['slots']):
        sid=slot['id'];y=top+i*row_h
        cv2.putText(sheet,sid,(5,y+44),cv2.FONT_HERSHEY_SIMPLEX,.55,(230,230,230),1,cv2.LINE_AA)
        refs={fi:(cl,img) for fi,cl,img in gallery.get(sid,[])}
        for j,fi in enumerate(frame_ids):
            if fi not in refs:continue
            cl,feature=refs[fi]
            x=left+j*col_w
            rgb=np.clip(feature,0,255).astype(np.uint8)
            patch=cv2.resize(rgb,(138,68),interpolation=cv2.INTER_LINEAR)
            sheet[y+2:y+70,x+4:x+142]=patch
            color=COLORS[cl]
            cv2.rectangle(sheet,(x+3,y+1),(x+143,y+71),color,2)
            cv2.putText(sheet,cl.upper(),(x+7,y+88),cv2.FONT_HERSHEY_SIMPLEX,.41,color,1,cv2.LINE_AA)
    cv2.imwrite(str(out/'reference_label_audit.jpg'),sheet)
    print('Reference audit:',out/'reference_audit.json')
    print('Visual label audit:',out/'reference_label_audit.jpg')
    print('Human labels used:',report['manual_reference_count'],dict(count))
    missing=[sid for sid,d in per_slot.items() if d['count']<3]
    if missing:print('WARNING: Fewer than 3 reference images for:',', '.join(missing))
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description='Calibrated Parking V3 demo using manual same-slot references')
    parser.add_argument('--source',default='dataset/parking_v3_stabilized.mp4')
    parser.add_argument('--config',default='config/parking_v3_slots_REVIEWED.json')
    parser.add_argument('--labels',default='evaluation/parking_v3_manual_labels.csv')
    parser.add_argument('--output-dir',default='results/parking_v3_reference')
    parser.add_argument('--max-frames',type=int,default=0)
    parser.add_argument('--step',type=int,default=1)
    parser.add_argument('--save-video',action='store_true')
    parser.add_argument('--show',action='store_true')
    parser.add_argument('--hide-ids',action='store_true')
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args(argv)
    if args.step<1 or args.max_frames<0:parser.error('--step must be positive; --max-frames nonnegative')
    source,config,labels,out=map(resolve,[args.source,args.config,args.labels,args.output_dir])
    info=parse_config(config)
    if not source.is_file():parser.error(f'Video not found: {source}')
    cap=cv2.VideoCapture(str(source))
    if not cap.isOpened():raise RuntimeError(f'Video could not be opened: {source}')
    writer=None
    try:
        sample_labels,counts=load_labels(labels,info)
        gallery=get_reference_data(cap,info,sample_labels)
        audit(out,info,gallery)
        if args.audit_only:return
        model=SlotReferenceDetector(gallery)
        fps=float(cap.get(cv2.CAP_PROP_FPS))
        if fps<=0 or fps>120:fps=10.0
        out.mkdir(parents=True,exist_ok=True)
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        processed=0; fi=0
        with (out/'predictions.csv').open('w',newline='',encoding='utf8') as f:
            csvwr=csv.writer(f);csvwr.writerow(['frame','slot_id','predicted_status','nearest_manual_frame','pixel_difference','method'])
            while True:
                if args.max_frames and fi>=args.max_frames:break
                ok,frame=cap.read()
                if not ok:break
                if fi%args.step:
                    fi+=1;continue
                if frame.shape[:2]!=(info['frame_height'],info['frame_width']):
                    raise ValueError('Video dimensions changed or ROI configuration mismatched video')
                preds=[]
                for s in info['slots']:
                    x,y,w,h=[int(s[z]) for z in ('x','y','w','h')]
                    status,near,d,method=model.classify(s['id'],frame[y:y+h,x:x+w])
                    pred={'id':s['id'],'x':x,'y':y,'w':w,'h':h,'status':status}
                    preds.append(pred)
                    csvwr.writerow([fi,s['id'],status,near,'' if not np.isfinite(d) else round(d,3),method])
                rendered=draw(frame,preds,fi,not args.hide_ids)
                if processed==0:cv2.imwrite(str(out/'first_frame.jpg'),rendered)
                if args.save_video:
                    if writer is None:
                        H,W=rendered.shape[:2]
                        writer=cv2.VideoWriter(str(out/'annotated_video.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps/args.step,(W,H))
                        if not writer.isOpened():raise RuntimeError('Cannot create output video')
                    writer.write(rendered)
                if args.show:
                    cv2.imshow('V3 - Scene Calibrated Parking',rendered)
                    if cv2.waitKey(max(1,round(1000*args.step/fps)))&255==ord('q'):
                        processed+=1;break
                processed+=1;fi+=1
        print(f'Processed {processed} frames x {len(info["slots"])} ROI = {processed*len(info["slots"])} predictions')
        print(f'Results: {out}')
        print('IMPORTANT: This model matches manually labelled images from the SAME VIDEO. This is a scene-specific demo, NOT independent test accuracy.')
    finally:
        cap.release()
        if writer:writer.release()
        if args.show:cv2.destroyAllWindows()

if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError,cv2.error) as exc:
        print('ERROR:',exc,file=sys.stderr)
        sys.exit(1)
