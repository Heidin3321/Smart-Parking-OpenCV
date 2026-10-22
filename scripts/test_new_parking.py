"""Smart Parking: independent new-photo test with manual ROI and blind ground truth.

Usage:
  python scripts/test_new_parking.py wizard --image "C:/Users/lenovo/Downloads/new_lot.jpg"
  python scripts/test_new_parking.py roi   --image "..."
  python scripts/test_new_parking.py label --image "..."
  python scripts/test_new_parking.py test  --image "..." --model both
  python scripts/test_new_parking.py report

Dependencies: cv2, numpy, existing detector.py and main.py.
Does NOT train on the test photo or load Parking V3 same-slot reference gallery.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detector import ParkingDetector
from main import read_slots

CLASS_NAMES = ('empty', 'occupied')
PALETTE = {'empty': (65, 195, 75), 'occupied': (72, 70, 239),
           'unknown': (0, 195, 245), 'uncertain': (0, 195, 245)}
TRAINING = {
    'cnrpark': 'dataset/real_cnrpark/train',
    'v3_adapted': 'dataset/v3_calibrated/deploy_train',
}


def resolved(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def read_image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f'Image not found: {path}')
    # np.fromfile supports Unicode names on Windows; cv2.imread sometimes does not.
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.ndim != 3:
        raise ValueError(f'Cannot read image: {path}. Use JPG/PNG, not a video file.')
    return img


def save_image(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    if ext not in ('.png', '.jpg', '.jpeg'):
        raise ValueError(f'Unsupported image output: {path}')
    ok, bytes_ = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f'Could not encode image: {path}')
    bytes_.tofile(str(path))


def identity(image_path: Path):
    p = resolved(image_path)
    if not p.is_file():
        raise FileNotFoundError(f'Cannot find image: {p}')
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    slug = re.sub(r'[^A-Za-z0-9_-]+', '_', p.stem).strip('_')[:36] or 'parking_image'
    return f'{slug}_{digest[:10]}', digest


def paths_for(image_path: Path):
    name, sha = identity(image_path)
    return {
        'config': ROOT / 'config' / 'new_parking' / f'{name}_rois.json',
        'labels': ROOT / 'evaluation' / 'new_parking' / f'{name}_truth.json',
        'output': ROOT / 'results' / 'new_parking' / name,
        'image_name': name, 'hash': sha,
    }


def save_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.part')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def roi_digest(cfg: dict) -> str:
    data = {key: cfg[key] for key in ('frame_width', 'frame_height', 'slots', 'image_sha256')}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def load_setup(image_path: Path):
    image_path = resolved(image_path)
    paths = paths_for(image_path)
    image = read_image(image_path)
    cfg = read_slots(paths['config'])
    if (image.shape[1], image.shape[0]) != (cfg['frame_width'], cfg['frame_height']):
        raise ValueError('Image dimensions do not match saved ROIs; redraw for this image.')
    if cfg.get('image_sha256') != paths['hash']:
        raise ValueError('Image identity changed; saved ROIs do not belong to this image.')
    return image, paths, cfg


def display_fit(img, max_w=1180, max_h=780):
    h, w = img.shape[:2]
    scale = min(1.0, max_w / w, max_h / h)
    return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA), scale


def line(img, txt, x, y, scale=.6, color=(240, 240, 240), thick=1):
    cv2.putText(img, txt, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thick, cv2.LINE_AA)


def canvas_with_header(img, title, sub, footer=None, max_w=1180, max_h=610):
    view, scale = display_fit(img, max_w, max_h)
    h, w = view.shape[:2]
    w = max(680, w)
    frame = np.full((h + 92, w, 3), (27, 35, 43), dtype=np.uint8)
    frame[64:64+h, :view.shape[1]] = view
    line(frame, title, 12, 28, .69, (255, 255, 255), 2)
    line(frame, sub[:120], 12, 51, .47, (115, 220, 244))
    if footer:
        line(frame, footer[:150], 12, h + 82, .49, (255, 255, 255))
    return frame, scale, 64


def display_roi(image, slots, title, footer, pending=None):
    shown, scale, top = canvas_with_header(image, title,
        'DRAW ROI WITH LEFT MOUSE | IDs assigned in drawing order', footer)
    for slot in slots:
        x,y,w,h = [int(slot[t]) for t in ('x','y','w','h')]
        x,y,w,h = [round(v*scale) for v in (x,y,w,h)]
        cv2.rectangle(shown, (x, y+top), (x+w, y+h+top), (0, 224, 254), 2)
        cv2.rectangle(shown, (x, y+top), (x+70, y+top+23), (0, 224, 254), -1)
        line(shown, slot['id'], x+4, y+top+17, .5, (22, 34, 40), 2)
    if pending:
        (x0,y0),(x1,y1)=pending
        cv2.rectangle(shown, (x0,y0),(x1,y1),(255,255,255),2)
    return shown, scale, top


def draw_rois(image_path: Path, replace=False):
    image_path = resolved(image_path)
    paths = paths_for(image_path)
    if paths['config'].exists() and not replace:
        print(f'ROI already saved: {paths["config"]}')
        print('Keep them and continue with label / test. For redrawing use roi --replace-roi.')
        return paths['config']
    image = read_image(image_path)
    h,w = image.shape[:2]
    slots = []
    state = {'start': None, 'pointer': None}
    window = 'Parking TEST - DRAW ROI (image is NOT training data)'
    _, scale, header = display_roi(image, slots, 'DRAW NEW PARKING ROIs','S save | U undo | C clear | Q cancel')

    def mouse(evt,x,y,flags,param):
        if evt == cv2.EVENT_LBUTTONDOWN:
            if y < header or y >= header + round(h*scale) or x >= round(w*scale):
                return
            state['start']=(x,y)
            state['pointer']=(x,y)
        elif evt == cv2.EVENT_MOUSEMOVE and state['start']:
            state['pointer']=(x,y)
        elif evt == cv2.EVENT_LBUTTONUP and state['start']:
            x0,y0 = state['start']
            x1,y1 = x,y
            left = max(0,min(w,round(min(x0,x1)/scale)))
            right = max(0,min(w,round(max(x0,x1)/scale)))
            top = max(0,min(h,round((min(y0,y1)-header)/scale)))
            bottom = max(0,min(h,round((max(y0,y1)-header)/scale)))
            if right-left >= 8 and bottom-top >= 8:
                slots.append({'id':f'T{len(slots)+1:03d}', 'x':left, 'y':top,
                              'w':right-left,'h':bottom-top})
            state['start']=None;state['pointer']=None

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window,mouse)
    print('Draw ROI boxes with mouse. Press S to save, U to undo, C to clear, Q/Esc to cancel.')
    try:
        while True:
            pending=(state['start'],state['pointer']) if state['start'] else None
            shown,_,_=display_roi(image,slots, f'DRAW NEW PARKING ROIs | {len(slots)} slots',
                         'S save | U undo | C clear | Q cancel', pending)
            cv2.imshow(window,shown)
            key=cv2.waitKey(25)&0xff
            if key==ord('s'):
                if not slots:
                    print('Draw at least one slot before saving.');continue
                cfg={
                    'frame_width':w,'frame_height':h, 'source_image':image_path.name,
                    'image_sha256':paths['hash'],
                    'annotation_method':'manual by human; NO labels or predictions used',
                    'slots':slots,
                }
                if paths['labels'].exists():
                    index=1
                    backup=paths['labels'].with_name(paths['labels'].stem+f'_before_roi_redraw_{index}.json')
                    while backup.exists():
                        index+=1
                        backup=paths['labels'].with_name(paths['labels'].stem+f'_before_roi_redraw_{index}.json')
                    paths['labels'].rename(backup)
                    print(f'Old manual truth backed up to {backup}; please label the new ROIs.')
                save_json(paths['config'],cfg)
                print(f'Saved {len(slots)} ROIs to {paths["config"]}')
                return paths['config']
            if key==ord('u') and slots: slots.pop()
            if key==ord('c'): slots.clear()
            if key in (ord('q'),27):
                raise RuntimeError('ROI drawing cancelled; nothing was saved.')
    finally:
        cv2.destroyWindow(window)


def load_truth(paths, cfg, allow_missing=False):
    dest=paths['labels']
    if not dest.exists():
        if allow_missing: return {}
        raise FileNotFoundError(f'Missing ground truth: {dest}. Run label first.')
    obj=json.loads(dest.read_text(encoding='utf8'))
    if obj.get('image_sha256')!=paths['hash'] or obj.get('roi_sha256')!=roi_digest(cfg):
        raise ValueError('Ground truth belongs to a different image/ROI config; relabel after changing ROI.')
    known={s['id'] for s in cfg['slots']}
    labels=obj.get('truth',{})
    if set(labels)-known or any(v not in (*CLASS_NAMES,'unknown') for v in labels.values()):
        raise ValueError('Invalid ground truth labels in file.')
    return labels


def put_truth(paths,cfg,truth):
    save_json(paths['labels'],{
        'image_sha256':paths['hash'], 'roi_sha256':roi_digest(cfg),
        'protocol':'human ground truth assigned before viewing model predictions',
        'truth':truth,
    })


def label_rois(image_path: Path, start_at=None):
    image,paths,cfg=load_setup(image_path)
    truth=load_truth(paths,cfg,allow_missing=True)
    slots=cfg['slots']
    if start_at:
        ids=[s['id'] for s in slots]
        if start_at not in ids:
            raise ValueError(f'Unknown slot ID {start_at}. Available: {ids}')
        i=ids.index(start_at)
    else:
        i=next((j for j,s in enumerate(slots) if s['id'] not in truth),len(slots))
    if i==len(slots):
        print(f'All {len(truth)} slots already labeled: {paths["labels"]}')
        print('To change one: label --image ... --start-at T003')
        return paths['labels']
    window='Parking TEST - HUMAN GROUND TRUTH (NO MODEL SHOWN)'
    cv2.namedWindow(window,cv2.WINDOW_AUTOSIZE)
    try:
        while i<len(slots):
            slot=slots[i];x,y,w,h=[slot[k] for k in ('x','y','w','h')]
            crop=image[y:y+h,x:x+w]
            if crop.size==0:raise ValueError(f'Invalid crop for {slot["id"]}')
            view,scale,header=canvas_with_header(image,
                f'MANUAL GROUND TRUTH | {slot["id"]} ({i+1}/{len(slots)})',
                'NO AI PREDICTIONS SHOWN. Inspect actual parking slot.',
                'O = OCCUPIED | E = EMPTY | U = UNKNOWN | B = back | Q = save & exit',
                max_w=1000,max_h=400)
            cv2.rectangle(view,(round(x*scale),header+round(y*scale)),
                (round((x+w)*scale),header+round((y+h)*scale)),(0,215,255),3)
            bigger=cv2.resize(crop,(420,180),interpolation=cv2.INTER_LINEAR)
            cw=max(view.shape[1],460)
            full=np.full((view.shape[0]+218,cw,3),(27,35,43),dtype=np.uint8)
            full[:view.shape[0],:view.shape[1]]=view
            full[view.shape[0]+12:view.shape[0]+192,15:435]=bigger
            line(full,f'Zoom: {slot["id"]} | previous: {truth.get(slot["id"],"-").upper()}',
                 448,view.shape[0]+56,.62,(235,235,235),2)
            line(full,'Press O / E / U',448,view.shape[0]+100,.7,(0,230,250),2)
            line(full,'Q saves progress',448,view.shape[0]+144,.52)
            cv2.imshow(window,full)
            key=cv2.waitKey(0)&0xff
            choice={ord('o'):'occupied',ord('e'):'empty',ord('u'):'unknown'}.get(key)
            if choice:
                truth[slot['id']]=choice
                put_truth(paths,cfg,truth)
                i+=1
            elif key==ord('b'):
                i=max(0,i-1)
            elif key in (ord('q'),27): break
    finally:
        cv2.destroyWindow(window)
    print(f'Ground truth saved: {paths["labels"]} ({len(truth)}/{len(slots)})',dict(Counter(truth.values())))
    return paths['labels']


def scores(truth, preds):
    """Full 2x2 confusion counts and strict accuracy; missing/uncertain predictions counted incorrect.

    Unknown ground truth excluded. Undefined precision/recall are None (not 0%).
    """
    evaluated=[(k,actual,preds[k]) for k,actual in truth.items() if actual in CLASS_NAMES]
    total=len(evaluated)
    matrix={actual:{pred:sum(a==actual and p==pred for _,a,p in evaluated)
                    for pred in CLASS_NAMES} for actual in CLASS_NAMES}
    invalid=[(k,a,p) for k,a,p in evaluated if p not in CLASS_NAMES]
    correct=sum(a==p for _,a,p in evaluated)
    per_class={}
    for klass in CLASS_NAMES:
        tp=matrix[klass][klass]
        fp=sum(matrix[a][klass] for a in CLASS_NAMES if a!=klass)
        fn=sum(matrix[klass][p] for p in CLASS_NAMES if p!=klass)+sum(a==klass for _,a,_ in invalid)
        support=sum(a==klass for _,a,_ in evaluated)
        prec=tp/(tp+fp) if tp+fp else None
        rec=tp/(tp+fn) if tp+fn else None
        f1=(2*prec*rec/(prec+rec) if prec+rec else 0.0) if prec is not None and rec is not None else None
        per_class[klass]={
            'support':support,
            'precision':None if prec is None else round(prec,4),
            'recall':None if rec is None else round(rec,4),
            'f1':None if f1 is None else round(f1,4),
        }
    return {
        'n_roi':len(truth), 'n_evaluable':total,
        'n_unknown_ground_truth':len(truth)-total,
        'correct':correct, 'wrong':total-correct,
        'accuracy':round(correct/total,4) if total else None,
        'prediction_coverage':round((total-len(invalid))/total,4) if total else None,
        'class_balance':dict(Counter(a for _,a,_ in evaluated)),
        'confusion_matrix':matrix,'per_class':per_class,
        'note':'Single-photo test. Manually selected ROIs, human labels before inference; no training on this photo. Results may not generalize to other images. Unknown ground truth excluded; uncertain predictions count as incorrect.',
    }


def overlay(image,slots, pred=None, truth=None, title='SMART PARKING TEST', subtitle='', height_limit=840):
    view,scale,top=canvas_with_header(image,title,subtitle,
        'GREEN=EMPTY | RED=OCCUPIED | YELLOW=UNKNOWN; original ROI size preserved',
        max_w=1250,max_h=height_limit)
    for s in slots:
        sid=s['id'];x,y,w,h=[s[z] for z in ('x','y','w','h')]
        x,y,w,h=[round(v*scale) for v in (x,y,w,h)]
        st = pred.get(sid, 'unknown') if pred is not None else truth.get(sid,'unknown')
        col=PALETTE.get(st,PALETTE['unknown'])
        cv2.rectangle(view,(x,y+top),(x+w,y+h+top),col,2)
        text=f'{sid}: {st.upper()}'
        (tw,th),bl=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.48,1)
        yy=y+top
        cv2.rectangle(view,(x,yy),(min(view.shape[1]-1,x+tw+12),min(view.shape[0]-1,yy+th+12)),col,-1)
        line(view,text,x+5,yy+th+6,.48,(16,25,35))
    return view


def safe_html(s):
    import html
    return html.escape(str(s), quote=True)


def write_html(out,stats,model_names, image_name, rows_by_model=None):
    rows_by_model = rows_by_model or {}
    blocks=[]
    for name in model_names:
        info=stats[name]
        accuracy='N/A (no evaluable labels)' if info['accuracy'] is None else f'{100*info["accuracy"]:.1f}%'
        cm=info['confusion_matrix']
        slot_table = ''.join(
            f'<tr><td>{safe_html(row["slot_id"])}</td>'
            f'<td>{safe_html(row["actual_status"])}</td>'
            f'<td>{safe_html(row["predicted_status"])}</td>'
            f'<td>{safe_html("excluded" if row["correct"] is None else ("correct" if row["correct"] else "wrong"))}</td></tr>'
            for row in rows_by_model.get(name, [])
        )
        blocks.append(f'''<section><h2>{safe_html(name)}</h2>
            <div class="metrics"><div><strong>{accuracy}</strong><span>Accuracy</span></div>
            <div><strong>{info['correct']}/{info['n_evaluable']}</strong><span>Correct / evaluated</span></div>
            <div><strong>{info['wrong']}</strong><span>Wrong</span></div></div>
            <h3>Confusion Matrix</h3>
            <table><thead><tr><th>Actual / Predicted</th><th>EMPTY</th><th>OCCUPIED</th></tr></thead>
            <tbody><tr><th>EMPTY</th><td>{cm['empty']['empty']}</td><td>{cm['empty']['occupied']}</td></tr>
            <tr><th>OCCUPIED</th><td>{cm['occupied']['empty']}</td><td>{cm['occupied']['occupied']}</td></tr></tbody></table>
            <h3>Slot-by-slot audit</h3><table><thead><tr><th>ROI</th><th>Human truth</th><th>Model output</th><th>Result</th></tr></thead><tbody>{slot_table}</tbody></table>
            <p><a href="annotated_{name}.jpg">Open annotated image</a> · <a href="predictions_{name}.csv">Prediction CSV</a></p></section>''')
    doc=f'''<!DOCTYPE html><html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Smart Parking — Unseen Image Evaluation</title>
    <style>body{{font:16px/1.55 Arial,sans-serif;margin:0;background:#f4f7fb;color:#182635}}main{{max-width:940px;margin:40px auto;padding:0 18px}}header{{background:#172b3b;color:#fff;padding:28px;border-radius:18px}}header h1{{margin:0 0 4px}}section{{padding:24px;background:white;border-radius:16px;margin-top:18px;box-shadow:0 4px 16px #1333}}section h2{{margin-top:0}}.metrics{{display:flex;gap:12px;flex-wrap:wrap}}.metrics>div{{background:#edf2f7;border-radius:12px;padding:12px 20px;min-width:130px}}strong{{display:block;font-size:24px}}span{{display:block;font-size:12px;color:#506478}}table{{border-collapse:collapse;width:100%;max-width:550px}}td,th{{padding:12px;border:1px solid #dce5ef;text-align:center}}th{{background:#edf2f7}}a{{color:#1164a3}}.notice{{border-left:4px solid #e0a800;padding:14px;background:#fffae7}}</style></head>
    <body><main><header><h1>Smart Parking · New Image Test</h1><div>{safe_html(image_name)}</div><div>Manual ROI · Pre-registered human labels · Existing KNN models</div></header>
    <section><h2>Independent photo workflow</h2><p>New photo is <b>not</b> used to train either tested model. Ground truth was assigned before inference. ROIs were manually configured.</p>
    <p class="notice">One image is a small test set and cannot establish broad deployment accuracy. The V3-adapted model was trained using the V3 video, but not this test photo. Class composition and viewing angle affect results.</p>
    <p><a href="ground_truth.jpg">View manually labeled ground truth</a> · <a href="report.json">Inspect machine-readable metrics</a></p></section>
    {''.join(blocks)}</main></body></html>'''
    (out/'report.html').write_text(doc,encoding='utf-8')


def test_image(image_path: Path, model: str='both'):
    image,paths,cfg=load_setup(image_path)
    truth=load_truth(paths,cfg)
    ids={s['id'] for s in cfg['slots']}
    if set(truth)!=ids:
        missing=sorted(ids-set(truth))
        raise ValueError(f'Manual labeling incomplete ({len(truth)}/{len(ids)}). Missing: {missing}. Run label first.')
    choices=list(TRAINING) if model=='both' else [model]
    out=paths['output'];out.mkdir(parents=True,exist_ok=True)
    save_image(out/'ground_truth.jpg',overlay(image,cfg['slots'],truth=truth,
                                             title='GROUND TRUTH | HUMAN-LABELED',
                                             subtitle='Labels assigned before inference - NOT AI predictions'))
    results={}; details={}
    for name in choices:
        td=resolved(TRAINING[name])
        if not (td/'empty').is_dir() or not (td/'occupied').is_dir():
            raise FileNotFoundError(f'Missing model training patches: {td} (empty/, occupied/). Run CNRPark data preparation or v3_adapt.py prepare first.')
        print(f'Loading existing model: {name} from {td} (never learns from test image).')
        detector=ParkingDetector(td,k=3)
        preds=detector.predict(image,cfg['slots'])
        mapping={p['id']:p['status'] for p in preds}
        score=scores(truth,mapping)
        results[name]=score
        save_image(out/f'annotated_{name}.jpg',overlay(image,cfg['slots'],pred=mapping,
            title=f'PREDICTION | {name.upper()}',
            subtitle='NEW PHOTO NOT USED IN TRAINING | Blind evaluation against manual labels'))
        details[name]=[]
        with (out/f'predictions_{name}.csv').open('w',newline='',encoding='utf-8') as f:
            wr=csv.writer(f)
            wr.writerow(('slot_id','actual_status','predicted_status','evaluation','correct'))
            for s in cfg['slots']:
                sid=s['id'];actual=truth[sid];pred=mapping[sid]
                if actual=='unknown': tag='excluded_unknown_truth';flag=''
                elif pred not in CLASS_NAMES:tag='abstained';flag=0
                else:tag='evaluated';flag=int(actual==pred)
                wr.writerow((sid,actual,pred,tag,flag))
                details[name].append({'slot_id':sid,'actual_status':actual,'predicted_status':pred,
                                      'correct':None if flag=='' else bool(flag)})
        rate=f'{100*score["accuracy"]:.1f}%' if score['accuracy'] is not None else 'N/A'
        print(f'{name}: {score["correct"]}/{score["n_evaluable"]} correct = {rate} | errors={score["wrong"]} | unknown truth={score["n_unknown_ground_truth"]}')
    report={
        'image_name':resolved(image_path).name,'image_sha256':paths['hash'],
        'roi_count':len(cfg['slots']),'label_counts':dict(Counter(truth.values())),
        'test_design':'human-drawn ROI, ground truth labeled BEFORE inference; no training on test photo',
        'models':{m:{'training_directory':TRAINING[m],**data,
                     'per_slot_results':details[m]} for m,data in results.items()},
        'limitations':['Small test size: single photo; use multiple independent scenes for stronger evaluation.',
                       'No ground truth for unknown areas; excluded from denominator.',
                       'Model training samples may differ greatly from new camera angle or weather.',
                       'Not a measurement of same-video per-slot Reference Classifier.'],
    }
    save_json(out/'report.json',report)
    write_html(out,results,choices, resolved(image_path).name, details)
    print('OUTPUT:',out)
    print('REPORT:',out/'report.html')
    print('NOTE: This is new-photo generalization testing, NOT same-video calibration accuracy.')
    return report


def report_all():
    out=ROOT/'results'/'new_parking'
    files=sorted(out.glob('*/report.json')) if out.is_dir() else []
    if not files:raise FileNotFoundError(f'No reports found under {out}. Run test first.')
    rows=[]
    for fn in files:
        data=json.loads(fn.read_text(encoding='utf8'))
        for model,stats in data['models'].items():
            rows.append({'image':data['image_name'],'model':model,
                         'evaluated':stats['n_evaluable'],'correct':stats['correct'],
                         'wrong':stats['wrong'],'accuracy':stats['accuracy'],
                         'report':str(fn.parent/'report.html')})
    csv_dest=out/'ALL_NEW_IMAGE_TESTS.csv'
    with csv_dest.open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]))
        wr.writeheader();wr.writerows(rows)
    print('Individual photo tests (do not call one-photo scores large-scale reliability):')
    for r in rows: print(f"  {r['image']}: {r['model']} {r['correct']}/{r['evaluated']} = {r['accuracy']}")
    print('Summary:',csv_dest)
    return rows


def choose_image():
    """Optional Windows file picker so a lecturer can choose the test image live."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError as e:
        raise RuntimeError('This Python install has no tkinter. Use --image "C:\\Users\\...\\parkir_baru.jpg" instead.') from e
    root=tk.Tk()
    root.withdraw()
    root.attributes('-topmost',True)
    try:
        chosen=filedialog.askopenfilename(parent=root,title='Choose a NEW parking lot image for testing',
            filetypes=[('Parking photo (JPG/PNG)','*.jpg *.jpeg *.png'),('All files','*.*')])
    finally:
        root.destroy()
    if not chosen:raise RuntimeError('No image selected.')
    return chosen


def main(argv=None):
    parser=argparse.ArgumentParser(description='Test existing Smart Parking KNN models on a NEW photo. No training on test photo.')
    sub=parser.add_subparsers(dest='action',required=True)
    for act in ('wizard','roi','label','test'):
        sp=sub.add_parser(act)
        sp.add_argument('--image',help='Path of a DIFFERENT real parking photo (JPG/PNG)')
        sp.add_argument('--pick',action='store_true',help='Open Windows file picker instead of specifying --image')
        if act in ('wizard','roi'):
            sp.add_argument('--replace-roi',action='store_true',help='Explicitly overwrite existing saved ROIs')
        if act in ('wizard','test'):
            sp.add_argument('--model',choices=('both','cnrpark','v3_adapted'),default='both')
        if act=='label': sp.add_argument('--start-at',help='Correct a specific ROI ID, e.g., T003')
    sub.add_parser('report')
    args=parser.parse_args(argv)
    try:
        if args.action!='report':
            if args.image and args.pick:raise ValueError('Specify either --image or --pick, not both.')
            if args.pick:args.image=choose_image()
            if not args.image:raise ValueError('Please specify --image \"path\" or --pick.')
            print('Test image:',args.image)
        if args.action=='roi': draw_rois(args.image,replace=args.replace_roi)
        elif args.action=='label': label_rois(args.image,args.start_at)
        elif args.action=='test': test_image(args.image,model=args.model)
        elif args.action=='wizard':
            draw_rois(args.image,replace=args.replace_roi)
            label_rois(args.image)
            test_image(args.image,model=args.model)
        else:report_all()
    except (OSError,RuntimeError,ValueError,cv2.error) as e:
        raise SystemExit(f'ERROR: {e}') from e


if __name__=='__main__':
    main()
