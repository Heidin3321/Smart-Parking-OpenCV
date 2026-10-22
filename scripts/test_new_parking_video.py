"""Smart Parking / NEW video location demo and honest within-video temporal test.

Wizard: choose a *NEW, mostly fixed-camera* video; draw ROI once; blind-label
calibration frames at the beginning and test frames near the end; compare existing
CNRPark KNN, newly trained scene KNN and per-slot reference matching; export video,
CSV, screenshots and HTML report. No test frames enter training.

Requires: existing project detector.py (OpenCV KNN), cv2, numpy, tkinter optional.
Fast VIDEO (NO labels needed): python scripts/test_new_parking_video.py video --pick
Calibrated + evaluated VIDEO: python scripts/test_new_parking_video.py wizard --pick
Resume: same command and same video, or --video "path/to/new_video.mp4"
Render again without dialogs: python scripts/test_new_parking_video.py run --video "..."
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detector import ParkingDetector

CLASSES = ('empty', 'occupied')
COLORS = {'empty': (75, 200, 75), 'occupied': (75, 78, 239), 'uncertain': (0, 215, 255)}
SOURCE_TRAIN = ROOT / 'dataset' / 'real_cnrpark' / 'train'


def abspath(p):
    p = Path(p).expanduser()
    return p if p.is_absolute() else ROOT / p


def save_json(path, obj):
    """Direct write avoids .part -> .json rename denied by Windows AV/indexers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def video_identity(path):
    path = abspath(path)
    if not path.is_file():
        raise FileNotFoundError(f'Video not found: {path}')
    # Hash sampled file bytes, not the whole potentially huge video.
    length = path.stat().st_size
    sha = hashlib.sha256()
    sha.update(str(length).encode())
    with path.open('rb') as f:
        sha.update(f.read(262144))
        if length > 262144:
            f.seek(max(0, length - 262144))
            sha.update(f.read(262144))
    slug = re.sub(r'[^a-zA-Z0-9_-]+', '_', path.stem).strip('_')[:24] or 'parking'
    tag = sha.hexdigest()[:12]
    return f'{slug}_{tag}', tag


def paths_for(video):
    name, fingerprint = video_identity(video)
    return {
        'name': name, 'fingerprint': fingerprint,
        'roi': ROOT / 'config' / 'new_parking_video' / f'{name}_rois.json',
        'labels': ROOT / 'evaluation' / 'new_parking_video' / f'{name}_labels.json',
        'train': ROOT / 'dataset' / 'new_parking_video' / name / 'train',
        'out': ROOT / 'results' / 'new_parking_video' / name,
    }


def choose_video():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as exc:
        raise RuntimeError('Tkinter missing: use --video "C:/path/parking_new.mp4"') from exc
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    try:
        p = filedialog.askopenfilename(parent=root, title='Pilih VIDEO parkiran BARU (bukan V3)',
            filetypes=[('Video parkiran', '*.mp4 *.avi *.mov *.mkv'), ('All files', '*.*')])
    finally:
        root.destroy()
    if not p:
        raise RuntimeError('Tidak ada video dipilih. Jalankan lagi dan pilih MP4 baru.')
    return abspath(p)


def open_video(path):
    cap = cv2.VideoCapture(str(abspath(path)))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {path}. Try converting to MP4 H.264.')
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if count < 12:
        cap.release()
        raise ValueError('Video needs at least 12 readable frames (fixed-camera parking video recommended).')
    if not (fps > 0 and math.isfinite(fps)) or fps > 240:
        fps = 10.0
    if min(width, height) <= 0:
        cap.release()
        raise ValueError('Invalid video dimensions.')
    return cap, {'count': count, 'fps': fps, 'width': width, 'height': height}


def display_dimensions(meta, max_width=960, max_height=720):
    factor = min(1., max_width / meta['width'], max_height / meta['height'])
    return max(1, int(round(meta['width']*factor))), max(1, int(round(meta['height']*factor)))


def frame_at(cap, index, size):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f'Cannot read frame {index}; video may be corrupted/codec unsupported')
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if frame.shape[1::-1] != (size[0], size[1]) else frame


def sample_indices(count, n, start, end):
    if n < 1:
        raise ValueError('sample count must be >= 1')
    # Deterministic temporal split: first 35% for training; final 45% for test.
    pts = np.linspace(start, end, n+2)[1:-1]
    ret = [int(round((count-1)*float(t))) for t in pts]
    if len(set(ret)) != len(ret):
        raise ValueError('Video too short for chosen number of distinct sampling frames.')
    return ret


def label_plan(count, calibration_count, test_count):
    calibration = sample_indices(count, calibration_count, .02, .34)
    testing = sample_indices(count, test_count, .57, .98)
    if set(calibration) & set(testing) or max(calibration) >= min(testing):
        raise ValueError('Temporal split overlaps; reduce --calib-count / --test-count')
    return {'calibration': calibration, 'test': testing}


def check_cfg(cfg, video, paths, meta, plan):
    assert_valid_roi(cfg)
    if cfg.get('fingerprint') != paths['fingerprint']:
        raise ValueError('ROI is for a DIFFERENT video: redraw ROIs for this clip.')
    if tuple(cfg[k] for k in ('frame_width','frame_height')) != display_dimensions(meta):
        raise ValueError('ROI working resolution differs from this video.')
    if cfg.get('sampling_plan') != plan:
        raise ValueError('Sampling plan changed: use the original --calib-count/--test-count or --reset-roi to reconfigure.')


def assert_valid_roi(cfg):
    w, h = cfg['frame_width'], cfg['frame_height']
    slots = cfg.get('slots', [])
    if not slots:
        raise ValueError('No ROIs drawn. Draw at least one parking slot.')
    ids = set()
    for s in slots:
        sid = s['id']
        if sid in ids:
            raise ValueError(f'Duplicate slot {sid}')
        ids.add(sid)
        x,y,sw,sh = (int(s[k]) for k in ('x','y','w','h'))
        if x < 0 or y < 0 or sw < 3 or sh < 3 or x+sw > w or y+sh > h:
            raise ValueError(f'Invalid ROI {s} for frame {w}x{h}')


def draw_roi_gui(frame, plan, meta, paths, replace=False):
    target = paths['roi']
    if target.is_file() and not replace:
        cfg = json.loads(target.read_text(encoding='utf-8'))
        check_cfg(cfg, None, paths, meta, plan)
        print(f'Using saved {len(cfg["slots"])} ROI (no redrawing): {target}')
        return cfg
    fh, fw = frame.shape[:2]
    # Draw UI at already downscaled processing resolution.
    banner = 84
    slots = []
    state = {'start': None, 'current': None}
    window = 'NEW VIDEO - DRAW PARKING ROIs - PRESS S TO SAVE'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, max(600, fw), fh+banner)

    def mouse(evt, x, y, flags, data):
        y0 = y-banner
        if evt == cv2.EVENT_LBUTTONDOWN and 0 <= x < fw and 0 <= y0 < fh:
            state['start'] = (x, y0)
            state['current'] = (x, y0)
        elif evt == cv2.EVENT_MOUSEMOVE and state['start']:
            state['current'] = (x, y0)
        elif evt == cv2.EVENT_LBUTTONUP and state['start']:
            x0, y0_start = state['start']
            left, right = sorted((max(0,min(fw,x0)), max(0,min(fw,x))))
            top, bottom = sorted((max(0,min(fh,y0_start)), max(0,min(fh,y0))))
            if right-left >= 9 and bottom-top >= 9:
                slots.append({'id': f'N{len(slots)+1:03d}', 'x': left, 'y': top,
                              'w': right-left, 'h': bottom-top})
            state['start'] = state['current'] = None

    cv2.setMouseCallback(window, mouse)
    try:
        while True:
            canvas = np.full((fh+banner, fw, 3), (27, 34, 44), np.uint8)
            canvas[banner:] = frame
            put(canvas, f'NEW PARKING VIDEO | {len(slots)} ROIs', 12, 27, .68)
            put(canvas, 'Drag ROI | S=save | U=undo | C=clear | Q=cancel', 12, 58, .48, (70,220,250))
            for s in slots:
                x,y,w,h = (s[k] for k in ('x','y','w','h'))
                cv2.rectangle(canvas,(x,y+banner),(x+w,y+h+banner),(0,215,250),2)
                put(canvas,s['id'],x+4,y+banner+17,.46,(0,215,250))
            if state['start'] and state['current']:
                a,b = state['start'],state['current']
                cv2.rectangle(canvas,(a[0],a[1]+banner),(b[0],b[1]+banner),(255,255,255),2)
            cv2.imshow(window, canvas)
            key=cv2.waitKey(25)&0xff
            if key==ord('s'):
                if not slots:
                    print('Draw at least one ROI first.');continue
                cfg = {'frame_width':fw,'frame_height':fh, 'source_video':str(meta['width'])+'x'+str(meta['height']),
                       'fingerprint':paths['fingerprint'], 'sampling_plan':plan,
                       'annotation':'manual ROI created without viewing model predictions',
                       'slots':slots}
                save_json(target, cfg)
                # When explicitly replacing ROI, invalidate any old labels in place by renaming with .bak.
                if paths['labels'].exists():
                    backup = paths['labels'].with_suffix('.json.bak')
                    i=1
                    while backup.exists():
                        backup=paths['labels'].with_suffix(f'.json.bak{i}');i+=1
                    paths['labels'].rename(backup)
                    print('Old labels backed up:', backup)
                print(f'Saved {len(slots)} video ROIs: {target}')
                return cfg
            if key==ord('u') and slots:slots.pop()
            if key==ord('c'):slots.clear()
            if key in (ord('q'),27):raise RuntimeError('ROI drawing canceled; previous saved files not changed.')
    finally:
        cv2.destroyWindow(window)


def put(img, text, x, y, scale=.55, color=(245,245,245), thick=1):
    cv2.putText(img,str(text),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,scale,color,thick,cv2.LINE_AA)


def label_key(phase, fi, sid):
    return f'{phase}:{fi}:{sid}'


def roi_signature(cfg):
    j = json.dumps({'slots': cfg['slots'], 'w':cfg['frame_width'], 'h':cfg['frame_height'],
                    'plan':cfg['sampling_plan']},sort_keys=True)
    return hashlib.sha256(j.encode()).hexdigest()


def load_labels(paths,cfg):
    p=paths['labels']
    if not p.exists():
        return {'fingerprint':paths['fingerprint'], 'roi_signature':roi_signature(cfg),
                'protocol':'Human calibration AND holdout-test truth recorded before displaying model predictions.',
                'labels':{}}
    data=json.loads(p.read_text(encoding='utf-8'))
    if data.get('fingerprint') != paths['fingerprint'] or data.get('roi_signature') != roi_signature(cfg):
        raise ValueError('Saved labels belong to different ROIs or sampling plan. Use --reset-roi then redraw.')
    return data


def labeled_entries(cfg):
    return [(phase, f, s) for phase in ('calibration','test')
            for f in cfg['sampling_plan'][phase] for s in cfg['slots']]


def label_gui(cap,cfg,paths,meta):
    data=load_labels(paths,cfg)
    tasks=labeled_entries(cfg)
    labeled=data['labels']
    index=next((i for i,(phase,fi,s) in enumerate(tasks) if label_key(phase,fi,s['id']) not in labeled),len(tasks))
    if index==len(tasks):
        print(f'Using all {len(tasks)} previously saved labels: {paths["labels"]}')
        return data
    print(f'Label progress: {len(labeled)}/{len(tasks)}; O=occupied, E=empty, U=unknown, B=back, Q=save & quit.')
    window='NEW VIDEO - BLIND HUMAN LABELS (no AI predictions shown)'
    cv2.namedWindow(window,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window,1100,900)
    size=(cfg['frame_width'],cfg['frame_height'])
    cache={}
    try:
        while index<len(tasks):
            phase,fi,s=tasks[index]
            if fi not in cache: cache[fi]=frame_at(cap,fi,size)
            fr=cache[fi]; x,y,w,h=(int(s[k]) for k in ('x','y','w','h'))
            crop=fr[y:y+h,x:x+w]
            if crop.size==0:raise ValueError(f'Empty crop {s["id"]}')
            scene=cv2.resize(fr,(min(850,size[0]),round(size[1]*min(850,size[0])/size[0])))
            ratio=scene.shape[1]/size[0]
            cx,cy,cw,ch=(round(v*ratio) for v in (x,y,w,h))
            cv2.rectangle(scene,(cx,cy),(cx+cw,cy+ch),(0,225,255),3)
            upper_h=max(120,scene.shape[0]); width=max(900,scene.shape[1]); height=upper_h+320
            panel=np.full((height,width,3),(24,31,40),np.uint8)
            panel[:scene.shape[0],:scene.shape[1]]=scene
            patch_scale=min(3.0,340/max(w,1),240/max(h,1))
            zoom=cv2.resize(crop,(max(1,round(w*patch_scale)),max(1,round(h*patch_scale))))
            panel[upper_h+20:upper_h+20+zoom.shape[0],15:15+zoom.shape[1]]=zoom
            put(panel,f'{phase.upper()} | F{fi} | {s["id"]} | {index+1}/{len(tasks)}',15,upper_h+272,.55)
            put(panel,'O = occupied    E = empty    U = unknown',390,upper_h+66,.56,(0,220,255))
            put(panel,'B = go back    Q = save and exit',390,upper_h+111,.52)
            put(panel,f'Previous: {labeled.get(label_key(phase,fi,s["id"]),"none")}',390,upper_h+160,.52)
            cv2.imshow(window,panel)
            key=cv2.waitKey(0)&0xff
            ans={ord('o'):'occupied',ord('e'):'empty',ord('u'):'unknown'}.get(key)
            if ans:
                labeled[label_key(phase,fi,s['id'])]=ans
                save_json(paths['labels'], data)  # No .part file; fixes prior WinError 5 rename.
                index+=1
            elif key==ord('b'): index=max(0,index-1)
            elif key in (ord('q'),27):break
    finally:
        cv2.destroyWindow(window)
    print(f'Saved {len(labeled)}/{len(tasks)} human labels: {paths["labels"]}')
    if len(labeled)<len(tasks):
        raise RuntimeError('Labeling PAUSED. Run the same wizard command to resume saved progress.')
    return data


def save_patches(cap,cfg,paths,data):
    train=paths['train']
    for klass in CLASSES:
        d=train/klass
        d.mkdir(parents=True,exist_ok=True)
        for old in d.glob('*.png'):old.unlink()
    c=Counter()
    size=(cfg['frame_width'],cfg['frame_height'])
    for fi in cfg['sampling_plan']['calibration']:
        fr=frame_at(cap,fi,size)
        for s in cfg['slots']:
            lab=data['labels'][label_key('calibration',fi,s['id'])]
            if lab not in CLASSES:continue
            x,y,w,h=(int(s[k]) for k in ('x','y','w','h'))
            cv2.imwrite(str(train/lab/f'F{fi}_{s["id"]}.png'),fr[y:y+h,x:x+w])
            c[lab]+=1
    print('CALIBRATION (early frames ONLY):',dict(c))
    return c


def feature(patch):
    resized=cv2.resize(patch,(48,24),interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(resized,(3,3),0).astype(np.float32)


def make_gallery(cap,cfg,data):
    gallery=defaultdict(list)
    size=(cfg['frame_width'],cfg['frame_height'])
    for fi in cfg['sampling_plan']['calibration']:
        fr=frame_at(cap,fi,size)
        for s in cfg['slots']:
            lab=data['labels'][label_key('calibration',fi,s['id'])]
            if lab in CLASSES:
                x,y,w,h=(int(s[k]) for k in ('x','y','w','h'))
                gallery[s['id']].append((lab,feature(fr[y:y+h,x:x+w])))
    return gallery


class SceneReference:
    def __init__(self,gallery):
        self.gallery=gallery

    def predict(self,frame,slots):
        results=[]
        for s in slots:
            x,y,w,h=(s[k] for k in ('x','y','w','h'))
            refs=self.gallery.get(s['id'],[])
            label='uncertain'
            if refs:
                img=feature(frame[y:y+h,x:x+w])
                best=min((float(np.mean(np.abs(img-r))),cl) for cl,r in refs)
                # Conservative threshold: large appearance changes are not treated as EMPTY.
                if best[0]<=24.0:label=best[1]
            results.append({'id':s['id'],'status':label,'x':x,'y':y,'w':w,'h':h})
        return results


def evaluate(truth, preds):
    paired=[(truth[k],preds.get(k,'uncertain')) for k in truth if truth[k] in CLASSES]
    cm={v:{u:0 for u in CLASSES} for v in CLASSES}
    absent=0; right=0
    for actual,pred in paired:
        if pred in CLASSES:cm[actual][pred]+=1
        else:absent+=1
        right+=(pred==actual)
    n=len(paired)
    cls={}
    for c in CLASSES:
        tp=cm[c][c]
        fp=sum(cm[a][c] for a in CLASSES if a!=c)
        fn=sum(cm[c][a] for a in CLASSES if a!=c)+sum(a==c and p not in CLASSES for a,p in paired)
        prec=tp/(tp+fp) if tp+fp else None
        rec=tp/(tp+fn) if tp+fn else None
        cls[c]={'precision':prec,'recall':rec,
                'f1':2*prec*rec/(prec+rec) if prec is not None and rec is not None and prec+rec else None,
                'support':sum(a==c for a,_ in paired)}
    return {'tested':n,'correct':right,'incorrect_including_uncertain':n-right,
            'accuracy':right/n if n else None, 'uncertain':absent,
            'coverage':(n-absent)/n if n else None,'confusion_matrix':cm,'per_class':cls,
            'unknown_truth_excluded':len(truth)-n}


def annotate(frame,slot_preds,frame_index,model,training_note=''):
    h,w=frame.shape[:2]
    header=90
    out=np.full((h+header,w,3),(27,31,42),np.uint8)
    out[header:]=frame
    cnt=Counter(p['status'] for p in slot_preds)
    text_scale=.62 if w>=700 else .42
    put(out,'SMART PARKING | NEW LOCATION VIDEO',12,26,text_scale,(250,250,250),2)
    put(out,f'EMPTY {cnt["empty"]} | OCCUPIED {cnt["occupied"]} | UNCERTAIN {cnt["uncertain"]}',12,53,.53 if w>=700 else .36,(90,230,245))
    put(out,f'{model} | FRAME {frame_index} | {training_note}',12,77,.43 if w>=700 else .32,(220,220,220))
    for p in slot_preds:
        x,y,rw,rh=[p[k] for k in ('x','y','w','h')]
        color=COLORS.get(p['status'],COLORS['uncertain'])
        cv2.rectangle(out,(x,y+header),(x+rw,y+rh+header),color,2)
        label=p['id']+':'+p['status'].upper()
        tw,th=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.42,1)[0]
        cv2.rectangle(out,(x,y+header),(min(w-1,x+tw+8),min(h+header-1,y+header+th+9)),color,-1)
        put(out,label,x+4,y+header+th+4,.42,(18,25,32))
    return out


def build_models(cap,cfg,paths,data,mode):
    models={}
    if mode in ('both','cnrpark'):
        if not all(any((SOURCE_TRAIN/cl).glob('*.jpg')) or any((SOURCE_TRAIN/cl).glob('*.png')) for cl in CLASSES):
            if mode=='cnrpark':raise FileNotFoundError(f'CNRPark train folders missing: {SOURCE_TRAIN}')
            print('WARNING: CNRPark baseline not prepared; baseline skipped:',SOURCE_TRAIN)
        else:models['cnrpark_pretrained']=ParkingDetector(SOURCE_TRAIN,k=3)
    if mode in ('both','calibrated'):
        counts=save_patches(cap,cfg,paths,data)
        if all(counts[c]>=1 for c in CLASSES):
            models['new_scene_knn']=ParkingDetector(paths['train'],k=min(3, sum(counts.values())))
        else:
            print('WARNING: KNN cannot be calibrated because early-frame labels contain only one class.')
            print('Make sure some calibration frames contain BOTH empty and occupied slots.')
        models['new_scene_reference']=SceneReference(make_gallery(cap,cfg,data))
    if not models:raise ValueError('No usable classifier; prepare CNRPark or supply both classes for calibration.')
    return models


def get_truth_frames(cfg,data):
    return {fi:{s['id']: data['labels'][label_key('test',fi,s['id'])]
                for s in cfg['slots']} for fi in cfg['sampling_plan']['test']}


def test_models(cap,cfg,data,models,paths):
    truth_frames=get_truth_frames(cfg,data)
    size=(cfg['frame_width'],cfg['frame_height'])
    all_scores={}; all_rows={}
    for name, model in models.items():
        actual={}; predictions={}; rows=[]
        for fi, truth in truth_frames.items():
            fr=frame_at(cap,fi,size)
            mapping={p['id']:p['status'] for p in model.predict(fr,cfg['slots'])}
            for sid,label in truth.items():
                key=f'{fi}:{sid}';actual[key]=label;predictions[key]=mapping[sid]
                rows.append({'frame':fi,'slot_id':sid,'human_truth':label,'prediction':mapping[sid],
                             'result':'excluded' if label=='unknown' else ('correct' if label==mapping[sid] else 'wrong')})
        all_scores[name]=evaluate(actual,predictions)
        all_rows[name]=rows
        print(f'{name}: {all_scores[name]["correct"]}/{all_scores[name]["tested"]} correct; '
              f'accuracy={all_scores[name]["accuracy"]}, uncertain={all_scores[name]["uncertain"]}')
    return all_scores,all_rows


def render_video(video,cap,meta,cfg,paths,models,rows,scores,max_seconds,output_fps):
    out=paths['out'];out.mkdir(parents=True,exist_ok=True)
    size=(cfg['frame_width'],cfg['frame_height'])
    fps=meta['fps']
    step=max(1,int(round(fps/output_fps)))
    video_fps=fps/step
    total=meta['count']
    if max_seconds>0:total=min(total,int(math.ceil(max_seconds*fps)))
    for model_name, samples in rows.items():
        with (out/f'test_predictions_{model_name}.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['frame','slot_id','human_truth','prediction','result'])
            writer.writeheader();writer.writerows(samples)
    csv_names=['frame','seconds','slot_id','predicted_status','model']
    writers={}
    csv_files={}
    try:
        for model_name in models:
            fname=out/f'annotated_{model_name}.mp4'
            vw=cv2.VideoWriter(str(fname),cv2.VideoWriter_fourcc(*'mp4v'),video_fps,(size[0],size[1]+90))
            if not vw.isOpened():raise RuntimeError(f'VideoWriter failed: {fname} (check codecs).')
            writers[model_name]=vw
            f=(out/f'video_predictions_{model_name}.csv').open('w',newline='',encoding='utf-8')
            csv_files[model_name]=f
            csv.writer(f).writerow(csv_names)
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        frame_no=0; saved=0
        while frame_no<total:
            ok, frame=cap.read()
            if not ok:break
            if frame_no%step==0:
                frame=cv2.resize(frame,size,interpolation=cv2.INTER_AREA)
                for name,model in models.items():
                    pred=model.predict(frame,cfg['slots'])
                    annotated=annotate(frame,pred,frame_no,name,'CALIBRATION FROM EARLY VIDEO FRAMES')
                    writers[name].write(annotated)
                    wr=csv.writer(csv_files[name])
                    for p in pred:
                        wr.writerow((frame_no,round(frame_no/fps,3),p['id'],p['status'],name))
                    if saved==0:
                        cv2.imwrite(str(out/f'preview_{name}.jpg'),annotated)
                saved+=1
            frame_no+=1
        print(f'Rendered {saved} output frames for {len(models)} models at {video_fps:.2f} fps.')
    finally:
        for v in writers.values():v.release()
        for f in csv_files.values():f.close()
    if not saved:raise RuntimeError('Zero output frames rendered; input video could not be decoded.')
    for model_name in models:
        verify_mp4(out/f'annotated_{model_name}.mp4')
    chosen=next((name for name in ('new_scene_reference','new_scene_knn','cnrpark_pretrained')
                 if name in models),next(iter(models)))
    shutil.copyfile(out/f'annotated_{chosen}.mp4',out/'annotated_video.mp4')
    verify_mp4(out/'annotated_video.mp4')
    shutil.copyfile(out/f'preview_{chosen}.jpg',out/'preview_annotated_video.jpg')
    print('PRIMARY VIDEO CREATED: ',out/'annotated_video.mp4')


def has_training_images(directory):
    directory=Path(directory)
    return all(any((directory/cl).glob('*.jpg')) or any((directory/cl).glob('*.png')) for cl in CLASSES)


def select_video_detector(cap, cfg, paths, method='auto'):
    """Pick a real classifier for ROI overlay, WITHOUT needing held-out test labels.

    auto: same-scene calibrated reference when both EARLY frames are labeled,
    otherwise pre-trained CNRPark KNN; fallback to trained V3 KNN when available.
    """
    if method not in ('auto','cnrpark','v3','scene_reference','scene_knn'):
        raise ValueError(f'Unknown detector choice: {method}')
    labels=None
    if method in ('auto','scene_reference','scene_knn'):
        labels=load_labels(paths,cfg)
        calibration_keys=[label_key('calibration',fi,s['id'])
                          for fi in cfg['sampling_plan']['calibration'] for s in cfg['slots']]
        calibration_complete=all(k in labels['labels'] for k in calibration_keys)
        if calibration_complete:
            if method in ('auto','scene_reference'):
                print('Using calibrated same-location per-slot reference model.')
                return 'new_scene_reference',SceneReference(make_gallery(cap,cfg,labels))
            if method=='scene_knn':
                counts=save_patches(cap,cfg,paths,labels)
                if all(counts[c] for c in CLASSES):
                    return 'new_scene_knn',ParkingDetector(paths['train'],k=min(3,sum(counts.values())))
                raise ValueError('Scene KNN requires both EMPTY and OCCUPIED calibration samples.')
        if method in ('scene_reference','scene_knn'):
            raise ValueError('Calibration labels are incomplete. Run wizard and finish EARLY-frame labels first, or use --detector cnrpark.')

    v3train=ROOT/'dataset'/'v3_calibrated'/'deploy_train'
    if method in ('auto','cnrpark') and has_training_images(SOURCE_TRAIN):
        print('Using existing CNRPark KNN, NOT trained on this new video.')
        return 'cnrpark_pretrained',ParkingDetector(SOURCE_TRAIN,k=3)
    if method in ('auto','v3') and has_training_images(v3train):
        print('Using existing V3-trained KNN, NOT retrained on this new video.')
        return 'v3_adapted_knn',ParkingDetector(v3train,k=3)
    if method=='cnrpark':
        raise FileNotFoundError(f'CNRPark training images missing from {SOURCE_TRAIN}')
    if method=='v3':
        raise FileNotFoundError(f'V3 training images missing from {v3train}')
    raise FileNotFoundError('No trained model available. Prepare dataset/real_cnrpark/train/{empty,occupied} '
                            'or dataset/v3_calibrated/deploy_train/{empty,occupied}, or finish calibration.')


def verify_mp4(filename):
    filename=Path(filename)
    if not filename.is_file() or filename.stat().st_size<2000:
        raise RuntimeError(f'Annotated MP4 missing/empty: {filename}; close media players and check output permissions.')
    cap=cv2.VideoCapture(str(filename))
    try:
        ok,frame=cap.read()
        if not ok or frame is None:
            raise RuntimeError(f'MP4 was written but cannot be decoded: {filename}. Check local codec support.')
    finally:
        cap.release()
    return filename


def open_video_result(filename, skip=False):
    filename=Path(filename).resolve()
    print('\n============================================')
    print('ANNOTATED VIDEO READY — ROI + RED/GREEN LABELS:')
    print(str(filename))
    print('PowerShell: ii "'+str(filename)+'"')
    print('============================================\n')
    if not skip and os.name=='nt':
        try:
            os.startfile(str(filename))  # noqa: B010 - intended on Windows desktop
        except OSError as exc:
            print('Cannot auto-open player; use PowerShell command above:',exc)


def render_one_video(cap, meta, cfg, paths, model_name, detector, max_seconds=0, output_fps=10):
    """Always create a directly playable annotated_video.mp4 (no CSV-only outcome).

    This is the simple NEW-VIDEO -> ROI -> VIDEO route, with no manual truth needed.
    Accuracy is NOT estimated until separate manual testing is performed.
    """
    out=paths['out'];out.mkdir(parents=True,exist_ok=True)
    filename=out/'annotated_video.mp4'
    size=(cfg['frame_width'],cfg['frame_height'])
    fps=float(meta['fps'])
    step=max(1,int(round(fps/output_fps)))
    actual_fps=fps/step
    frames=int(meta['count'])
    if max_seconds>0:frames=min(frames,int(math.ceil(max_seconds*fps)))
    writer=cv2.VideoWriter(str(filename),cv2.VideoWriter_fourcc(*'mp4v'),actual_fps,(size[0],size[1]+90))
    if not writer.isOpened():
        raise RuntimeError(f'Could not open MP4 writer: {filename}. Close video player, check disk space/codecs.')
    rows=[];saved=0
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    try:
        for idx in range(frames):
            ok,frame=cap.read()
            if not ok:break
            if idx%step:continue
            frame=cv2.resize(frame,size,interpolation=cv2.INTER_AREA) if frame.shape[1::-1]!=size else frame
            preds=detector.predict(frame,cfg['slots'])
            annotated=annotate(frame,preds,idx,model_name,'VIDEO WITH FIXED ROI')
            writer.write(annotated)
            if saved==0:
                if not cv2.imwrite(str(out/'preview_annotated_video.jpg'),annotated):
                    raise RuntimeError('Could not save annotated preview JPG')
            for p in preds:
                rows.append((idx,round(idx/fps,3),p['id'],p['status'],model_name))
            saved+=1
    finally:
        writer.release()
    if not saved:raise RuntimeError('Zero annotated frames generated; check input MP4.')
    verify_mp4(filename)
    with (out/'video_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.writer(f)
        wr.writerow(('frame','seconds','slot_id','predicted_status','model'))
        wr.writerows(rows)
    print(f'VIDEO SUCCESS: {saved} frames, {len(cfg["slots"])} ROI, {model_name} ({actual_fps:.2f} fps)')
    print('IMPORTANT: This output is a VIDEO, not independent accuracy/testing evidence.')
    return filename


def html_report(paths,video,meta,cfg,scores,labels):
    out=paths['out'];out.mkdir(parents=True,exist_ok=True)
    counters={'calibration':Counter(),'test':Counter()}
    for key, value in labels.items():
        counters[key.split(':',1)[0]][value]+=1
    report={
        'source_video':str(video),'video_fingerprint':paths['fingerprint'],
        'frame_count':meta['count'],'fps':meta['fps'],
        'roi_count':len(cfg['slots']), 'frame_sampling_plan':cfg['sampling_plan'],
        'human_label_counts':{k:dict(v) for k,v in counters.items()},
        'holdout_protocol':'EARLY frames (<=34%) calibration; LATE frames (>=57%) test, never used in train or reference.',
        'models':scores,
        'limitations':[
            'Same-video late-frame test remains temporally correlated with calibration frames; not an independent location test.',
            'ROIs are fixed; moving/drone camera requires video stabilization or ROI tracking.',
            'Small test size; confidence intervals / external video tests are needed for deployment claims.',
            'New scene KNN needs manual labels from both classes; missing-class training is skipped.',
            'UNCERTAIN predictions count as incorrect in strict accuracy, and prediction coverage is reported.',
        ],
    }
    save_json(out/'report.json',report)
    safe=html.escape(str(video.name))
    styles='body{font:16px/1.55 Arial;background:#f3f6fb;color:#1e293b;margin:0}main{max-width:1050px;margin:40px auto;padding:0 20px}header{background:#132637;color:white;padding:24px;border-radius:14px}section{background:white;padding:22px;margin:20px 0;border-radius:12px;box-shadow:0 3px 12px #152c4112}table{border-collapse:collapse;width:100%}td,th{padding:9px;border:1px solid #e1e7ed;text-align:left}th{background:#e8eff4}a{color:#1166ae}img{width:100%;max-width:720px;border-radius:9px}.note{background:#fff6e2;padding:12px;border-left:4px solid #dba328}'
    sections=[]
    for name,score in scores.items():
        acc='N/A' if score['accuracy'] is None else f'{score["accuracy"]*100:.1f}%'
        cov='N/A' if score['coverage'] is None else f'{score["coverage"]*100:.1f}%'
        cm=score['confusion_matrix']
        sections.append(f'<section><h2>{html.escape(name)}</h2><p><b>Accuracy {acc}</b> | {score["correct"]}/{score["tested"]} correct | coverage {cov} | uncertain {score["uncertain"]}</p>'
           +f'<p><a href="annotated_{name}.mp4">Video anotasi</a> · <a href="test_predictions_{name}.csv">CSV testing</a></p>'
           +f'<img src="preview_{name}.jpg" alt="First video frame annotated by {name}">'
           +'<h3>Confusion matrix</h3><table><tr><th>Actual \\ Predicted</th><th>EMPTY</th><th>OCCUPIED</th></tr>'
           +f'<tr><th>EMPTY</th><td>{cm["empty"]["empty"]}</td><td>{cm["empty"]["occupied"]}</td></tr>'
           +f'<tr><th>OCCUPIED</th><td>{cm["occupied"]["empty"]}</td><td>{cm["occupied"]["occupied"]}</td></tr></table></section>')
    text=f'<!doctype html><html lang="id"><meta charset="utf-8"><title>Smart Parking NEW VIDEO Test</title><style>{styles}</style><main>' \
         f'<header><h1>Smart Parking · New Video Testing</h1><p>{safe}</p><p>ROI {len(cfg["slots"])} slots · {meta["count"]} original frames</p></header>' \
         '<section><h2>Protocol</h2><p>Manual calibration labels at the beginning of the clip; blind human ground truth on later, temporally held-out frames BEFORE model inference.</p>' \
         '<p class="note">This is <b>within-video testing</b>, not an independent real-world accuracy claim. Use a second recording from another session for stronger evidence. Camera must stay fixed.</p>' \
         '<p><b><a href="annotated_video.mp4">▶ OPEN MAIN ANNOTATED VIDEO (ROI + RED/GREEN)</a></b></p><p><a href="report.json">Raw metrics JSON</a></p></section>'+''.join(sections)+'</main></html>'
    (out/'report.html').write_text(text,encoding='utf-8')
    print('HTML REPORT:',out/'report.html')
    return report


def run(video, cfg, paths, meta, max_seconds, model_mode, output_fps):
    data=load_labels(paths,cfg)
    necessary={label_key(phase,fi,s['id']) for phase,fi,s in labeled_entries(cfg)}
    missing=necessary-set(data['labels'])
    if missing:raise ValueError(f'Incomplete human labels: {len(missing)} missing. Run wizard --video "..." to finish labeling.')
    cap,_=open_video(video)
    try:
        models=build_models(cap,cfg,paths,data,model_mode)
        scores,rows=test_models(cap,cfg,data,models,paths)
        render_video(video,cap,meta,cfg,paths,models,rows,scores,max_seconds,output_fps)
        report=html_report(paths,video,meta,cfg,scores,data['labels'])
        return report
    finally:cap.release()


def main(argv=None):
    parser=argparse.ArgumentParser(description='NEW parking VIDEO: manual ROI, independent temporal calibration/test, annotated videos.')
    parser.add_argument('action',choices=('wizard','run','roi','label','video'),nargs='?',default='wizard')
    parser.add_argument('--pick',action='store_true',help='Choose NEW MP4/AVI via Windows dialog (wizard/roi only)')
    parser.add_argument('--video',help='Path to a NEW mostly fixed-camera parking video')
    parser.add_argument('--calib-count',type=int,default=2,help='Early frames sampled for calibration (default: 2)')
    parser.add_argument('--test-count',type=int,default=2,help='Late frames sampled for blind test (default: 2)')
    parser.add_argument('--reset-roi',action='store_true',help='Explicitly redraw ROI and backup prior ground truth')
    parser.add_argument('--model',choices=('both','cnrpark','calibrated'),default='both',help='Evaluation models for wizard/run')
    parser.add_argument('--detector',choices=('auto','cnrpark','v3','scene_reference','scene_knn'),default='auto',help='Detector for VIDEO action (auto: scene reference if calibrated, else pretrained)')
    parser.add_argument('--no-open',action='store_true',help='Do not auto-open annotated_video.mp4 on Windows')
    parser.add_argument('--max-seconds',type=float,default=0,help='Limit rendered video duration; 0 for whole clip')
    parser.add_argument('--output-fps',type=float,default=10,help='Sampling fps for rendering output (default: 10)')
    args=parser.parse_args(argv)
    try:
        if args.calib_count<1 or args.test_count<1 or args.output_fps<=0 or args.max_seconds<0:
            raise ValueError('Calib count/test count/output FPS must be positive, max seconds nonnegative.')
        if args.video and args.pick:raise ValueError('Choose --pick or --video, not both')
        if args.pick:args.video=choose_video()
        if not args.video:raise ValueError('No video specified. Use --pick or --video "C:/path/parking.mp4"')
        video=abspath(args.video)
        print('NEW PARKING VIDEO:',video)
        paths=paths_for(video)
        cap,meta=open_video(video)
        plan=label_plan(meta['count'],args.calib_count,args.test_count)
        print('Calibration frames:',plan['calibration'],'| Blind TEST frames:',plan['test'])
        try:
            if args.action in ('wizard','roi','video'):
                ref=frame_at(cap,plan['calibration'][0],display_dimensions(meta))
                cfg=draw_roi_gui(ref,plan,meta,paths,replace=args.reset_roi)
            else:
                if not paths['roi'].exists():raise FileNotFoundError(f'ROI missing: {paths["roi"]}. First run wizard.')
                cfg=json.loads(paths['roi'].read_text(encoding='utf-8'))
                check_cfg(cfg,video,paths,meta,plan)
            if args.action=='video':
                detector_name,detector=select_video_detector(cap,cfg,paths,method=args.detector)
                output=render_one_video(cap,meta,cfg,paths,detector_name,detector,args.max_seconds,args.output_fps)
            if args.action in ('wizard','label'):
                label_gui(cap,cfg,paths,meta)
        finally:cap.release()
        if args.action=='video':
            open_video_result(output,skip=args.no_open)
        if args.action in ('wizard','run'):
            result=run(video,cfg,paths,meta,args.max_seconds,args.model,args.output_fps)
            print('COMPLETE. Open report:',paths['out']/'report.html')
            print('NOTE: Late-video temporal test, not independent-camera deployment accuracy.')
            open_video_result(paths['out']/'annotated_video.mp4',skip=args.no_open)
    except (OSError,RuntimeError,ValueError,cv2.error) as exc:
        raise SystemExit(f'ERROR: {exc}') from exc

if __name__=='__main__':main()
