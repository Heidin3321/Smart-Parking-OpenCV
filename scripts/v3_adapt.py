"""Parking V3 local domain adaptation, manual labels and leakage-aware evaluation.

No data is automatically labelled from existing model predictions.
Holdout is disjoint by slot identity, *not* random crop split.
Use in a local OpenCV project with detector.py / main.py and Parking V3 footage.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from main import read_slots
from detector import ParkingDetector, CLASS_NAMES, extract_features

DEFAULT_SOURCE = 'dataset/parking_v3_stabilized.mp4'
DEFAULT_CONFIG = 'config/parking_v3_slots_REVIEWED.json'
DEFAULT_LABELS = 'evaluation/parking_v3_manual_labels.csv'
DEFAULT_DIR = 'dataset/v3_calibrated'
DEFAULT_OUT = 'results/v3_adaptation'
# Frozen BEFORE the user labels data: evaluation slots never contribute crops to train/.
HOLDOUT_SLOTS = ('P003', 'P010', 'P012')
VALID = ('empty', 'occupied')
HEADER = ['frame', 'slot_id', 'true_status', 'x', 'y', 'w', 'h']


def path(s):
    p = Path(s)
    return p if p.is_absolute() else ROOT / p


def roi_fingerprint(info: dict) -> str:
    basis = {k: info[k] for k in ('frame_width', 'frame_height', 'slots')}
    return hashlib.sha256(json.dumps(basis, sort_keys=True).encode('utf8')).hexdigest()


def video_reader(source: Path):
    if not source.is_file():
        raise FileNotFoundError(f'Missing video: {source}')
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open video: {source}')
    return cap


def read_frame(cap, index: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok: raise RuntimeError(f'Cannot read frame {index}')
    return frame


def frame_plan(total: int, n: int, fps: float) -> list[int]:
    if n < 2: raise ValueError('At least 2 temporal samples required')
    margin = min(round(fps), max(0, total//12))
    first, last = margin, total - 1 - margin
    if last <= first: raise ValueError('Video too short for sampling')
    values = sorted(set(int(round(v)) for v in np.linspace(first, last, n)))
    if len(values) != n: raise ValueError('Too many samples for available frames')
    return values


def load_labels(dest: Path):
    if not dest.is_file(): return {}
    with dest.open(newline='', encoding='utf-8-sig') as f:
        entries = list(csv.DictReader(f))
    labels = {}
    for r in entries:
        try:
            key = (int(r['frame']), r['slot_id'])
            labels[key] = {
                'frame': key[0], 'slot_id': key[1], 'true_status': r['true_status'],
                **{k:int(r[k]) for k in ('x','y','w','h')},
            }
        except (KeyError, ValueError) as e:
            raise ValueError(f'Invalid label entry {r}: {e}') from e
    return labels


def write_labels(dest: Path, labels):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.partial.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        for _, r in sorted(labels.items()): writer.writerow(r)
    tmp.replace(dest)


def check_label_compatibility(labels, info):
    lookup = {s['id']:s for s in info['slots']}
    for key, r in labels.items():
        slot = lookup.get(r['slot_id'])
        if slot is None or any(r[k] != slot[k] for k in ('x','y','w','h')):
            raise ValueError(f'ROI coordinates changed for {key}; existing labels may be stale. Back up labels, relabel or use original ROI JSON.')


def label(args):
    src, config, labels_path = map(path, (args.source, args.config, args.labels))
    info = read_slots(config)
    cap = video_reader(src)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 10.0
    frames = frame_plan(total, args.samples, fps)
    slots = info['slots']
    seen = load_labels(labels_path)
    check_label_compatibility(seen, info)
    tasks = [(fr, s) for fr in frames for s in slots]
    task_keys = {(fr,s['id']) for fr,s in tasks}
    leftover = set(seen)-task_keys
    if leftover: raise ValueError(f'Existing label set has frames from a different sample plan: {sorted(leftover)[:3]}. Use same --samples or a different --labels path.')
    cursor = next((i for i,(fr,s) in enumerate(tasks) if (fr,s['id']) not in seen), len(tasks))
    last_frame_num = None
    frame = None
    name='Parking V3 - LABEL MANUAL (NOT AI)'
    try:
        while cursor < len(tasks):
            fi, slot = tasks[cursor]
            if last_frame_num != fi:
                frame = read_frame(cap, fi)
                last_frame_num = fi
            if frame.shape[:2] != (info['frame_height'], info['frame_width']):
                raise ValueError('ROI JSON video resolution mismatch')
            x,y,w,h = (slot[k] for k in ('x','y','w','h'))
            roi = frame[y:y+h,x:x+w]
            cw,ch = 1030,710
            canvas = np.full((ch,cw,3), (31,36,40),np.uint8)
            view=cv2.resize(frame,(570,678),interpolation=cv2.INTER_LINEAR)
            canvas[24:702,0:570] = view
            scale_x=570/frame.shape[1]
            scale_y=678/frame.shape[0]
            for s in slots:
                xx,yy,ww,hh = (s[k] for k in ('x','y','w','h'))
                col=(0,217,255) if s['id']==slot['id'] else (82,94,106)
                thick=3 if s['id']==slot['id'] else 1
                cv2.rectangle(canvas,(round(xx*scale_x),24+round(yy*scale_y)),(round((xx+ww)*scale_x),24+round((yy+hh)*scale_y)),col,thick)
            patch = cv2.resize(roi,(420,182),interpolation=cv2.INTER_LINEAR)
            canvas[165:347,595:1015]=patch
            def line(txt, yy, size=.65, color=(225,225,225)):
                cv2.putText(canvas,txt,(595,yy),cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
            line('MANUAL GROUND TRUTH',60,.78)
            line(f'SLOT {slot["id"]} / FRAME {fi}',100,.8,(0,215,255))
            line(f'{cursor+1} / {len(tasks)}   labeled: {len(seen)}',130,.58)
            line('O  = OCCUPIED (car present)',394,.63,(90,100,255))
            line('E  = EMPTY (no car)',434,.63,(85,222,120))
            line('K  = UNKNOWN / unclear',474,.63)
            line('B  = BACK and correct',514,.63)
            line('Q  = SAVE and quit',554,.63)
            line('Do not copy model predictions.',607,.55,(0,205,255))
            line('Inspect parking BAY, not just shadow.',635,.5)
            cv2.imshow(name,canvas)
            key=cv2.waitKey(0)&0xff
            status={ord('o'):'occupied',ord('e'):'empty',ord('k'):'unknown'}.get(key)
            if status:
                r={'frame':fi,'slot_id':slot['id'],'true_status':status,**{k:slot[k] for k in ('x','y','w','h')}}
                seen[(fi,slot['id'])]=r
                write_labels(labels_path,seen)
                cursor+=1
                while cursor <len(tasks) and (tasks[cursor][0],tasks[cursor][1]['id']) in seen:cursor+=1
            elif key==ord('b'):
                cursor=max(0,cursor-1)
                prev=tasks[cursor]
                seen.pop((prev[0],prev[1]['id']),None)
                write_labels(labels_path,seen)
            elif key in (ord('q'),27): break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    counts=Counter(r['true_status'] for r in seen.values())
    print(f'Labels saved: {labels_path} ({len(seen)}/{len(tasks)}). Counts: {dict(counts)}')
    print('Re-run the SAME label command to continue unfinished labels.')


def images_from_labels(args, info):
    src = path(args.source)
    labels_path=path(args.labels)
    labels=load_labels(labels_path)
    check_label_compatibility(labels,info)
    if not labels: raise ValueError('No manual labels. First run: python scripts/v3_adapt.py label')
    cap=video_reader(src)
    try:
        fps=float(cap.get(cv2.CAP_PROP_FPS)) or 10
        total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_expected=set(frame_plan(total, args.samples, fps))
        missing=frames_expected-{fr for fr,_ in labels}
        if missing: print(f'NOTE: some frames are not labelled yet: {sorted(missing)}')
        result=[]
        last_fr=-1; frame=None
        for _,r in sorted(labels.items()):
            if r['true_status'] not in VALID:continue
            if r['frame'] != last_fr:
                frame=read_frame(cap,r['frame']);last_fr=r['frame']
            x,y,w,h = (r[k] for k in ('x','y','w','h'))
            if x+w>frame.shape[1] or y+h>frame.shape[0]:raise ValueError('ROI out of frame')
            result.append((r,frame[y:y+h,x:x+w].copy()))
        return result
    finally: cap.release()


def choose_balanced(rows, seed=2026):
    grouped={st: [row for row in rows if row[0]['true_status']==st] for st in VALID}
    n=min(len(grouped['empty']),len(grouped['occupied']))
    if n<2: raise ValueError(f'Not enough training samples from BOTH classes: empty={len(grouped["empty"])}, occupied={len(grouped["occupied"])}. Label more slots/frames.')
    rng=random.Random(seed)
    return [row for st in VALID for row in rng.sample(grouped[st],n)]


def prepare(args):
    info=read_slots(path(args.config))
    records=images_from_labels(args,info)
    known={s['id'] for s in info['slots']}
    test_ids=set(HOLDOUT_SLOTS)
    if not test_ids.issubset(known):
        raise ValueError(f'Required holdout slots missing: {sorted(test_ids-known)}; ROI IDs were changed. See docs before choosing a new split.')
    train=[(r,p) for r,p in records if r['slot_id'] not in test_ids]
    test=[(r,p) for r,p in records if r['slot_id'] in test_ids]
    if not test or len({r['true_status'] for r,_ in test}) < 2:
        raise ValueError('Holdout test slot labels must include BOTH empty and occupied examples. Label P003/P010/P012.')
    balanced=choose_balanced(train)
    deploy=choose_balanced(records)
    base=path(args.data_dir)
    if base.resolve() in [ROOT.resolve(), (ROOT/'dataset').resolve()]:
        raise ValueError('Unsafe output data directory')
    for kind in ('train','test','deploy_train'):
        current=base/kind
        if current.exists():shutil.rmtree(current)
        for c in VALID:(current/c).mkdir(parents=True,exist_ok=True)
    def write(group, kind):
        for r,img in group:
            dest=base/kind/r['true_status']/f"f{r['frame']:04d}_{r['slot_id']}.png"
            if not cv2.imwrite(str(dest),img):raise RuntimeError(f'Failed to write {dest}')
    write(balanced,'train')
    write(test,'test')
    write(deploy,'deploy_train')
    manifest=base/'split_manifest.csv'
    with manifest.open('w', newline='',encoding='utf8') as f:
        writer=csv.writer(f); writer.writerow(['frame','slot_id','true_status','split'])
        for r,_ in records:
            writer.writerow([r['frame'],r['slot_id'],r['true_status'], 'holdout_test' if r['slot_id'] in test_ids else 'training_slot'])
    summary={
        'source':'Parking V3 video, HUMAN-LABELED by team',
        'split_rule':'SLOT-disjoint holdout, pre-specified IDs',
        'holdout_slot_ids':sorted(test_ids),
        'training_slot_ids':sorted(known-test_ids),
        'training_balanced_count':len(balanced),
        'test_count':len(test),
        'deployment_training_count':len(deploy),
        'test_label_counts':dict(Counter(r['true_status'] for r,_ in test)),
        'warning':'Frames within each slot are temporally correlated; test accuracy applies to this particular short video only. deploy_train includes heldout samples and must NOT be evaluated as holdout.'
    }
    (base/'split_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))
    print(f'Created train/, test/, deploy_train/ at {base}')


def calc_scores(records):
    labels=VALID
    correct=sum(t==p for _,_,t,p in records)
    total=len(records)
    matrix={actual:{pred:sum(t==actual and p==pred for _,_,t,p in records) for pred in labels} for actual in labels}
    metrics={}
    for st in labels:
        tp=matrix[st][st]
        fp=sum(matrix[a][st] for a in labels if a!=st)
        fn=sum(matrix[st][a] for a in labels if a!=st)
        prec=tp/(tp+fp) if tp+fp else 0.
        rec=tp/(tp+fn) if tp+fn else 0.
        f1=2*prec*rec/(prec+rec) if prec+rec else 0.
        metrics[st]={k:round(v,4) for k,v in {'precision':prec,'recall':rec,'f1':f1,'support':sum(matrix[st].values())}.items()}
    return {'total':total,'correct':correct,'accuracy':round(correct/total,4) if total else None,'confusion_matrix':matrix,'per_class':metrics}


def evaluate(args):
    base=path(args.data_dir)
    summary_path=base/'split_summary.json'
    if not summary_path.exists():raise FileNotFoundError(f'Run prepare first: {summary_path}')
    summary=json.loads(summary_path.read_text(encoding='utf8'))
    actual_test_files={cls:list((base/'test'/cls).glob('*.png')) for cls in VALID}
    if any(not x for x in actual_test_files.values()):raise ValueError('Both classes must exist in holdout set')
    local=ParkingDetector(base/'train', k=args.k)
    baseline_path=path(args.cnrpark_dir)
    baseline=ParkingDetector(baseline_path,k=args.k) if baseline_path.exists() else None
    cases=[]
    for class_name, files in actual_test_files.items():
        for im in sorted(files):
            patch=cv2.imread(str(im))
            if patch is None: raise RuntimeError(f'Unreadable: {im}')
            frame_id=int(im.stem.split('_')[0][1:]);slot_id=im.stem.split('_')[1]
            feats=extract_features(patch).reshape(1,-1)
            cases.append((frame_id,slot_id,class_name,feats))
    # Clearly distinguish real video samples from real CNRPark *source* training crops.
    predictors={'cnrpark_baseline':baseline,'v3_slot_disjoint':local}
    results={}; rows=[]
    for model_name,model in predictors.items():
        if model is None:continue
        preds=[]
        for fr,slot,true,vec in cases:
            _,r,_,_=model.model.findNearest(vec,k=model.k)
            predicted=CLASS_NAMES[int(r[0,0])]
            preds.append((fr,slot,true,predicted))
            rows.append([model_name,fr,slot,true,predicted,int(predicted==true)])
        results[model_name]=calc_scores(preds)
    report={'split':'strictly distinct slot IDs between evaluation train and test','test_slot_ids':summary['holdout_slot_ids'],
            'results':results,'limitations':'Small, highly correlated 28s video; accuracy does not establish deployment reliability or independent scene generalization.'}
    out=path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    with (out/'v3_model_comparison.csv').open('w',newline='',encoding='utf8') as f:
        wr=csv.writer(f);wr.writerow(['model','frame','slot_id','true_status','predicted_status','correct']);wr.writerows(rows)
    (out/'v3_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    for name,stats in results.items(): print(f'{name}: {stats["correct"]}/{stats["total"]} = {stats["accuracy"]*100:.2f}%  [held-out slots {", ".join(summary["holdout_slot_ids"])}]')
    print('Confusion matrices and class metrics:',out/'v3_evaluation.json')
    print('If adapted accuracy is worse, keep the baseline or try better labels/features; do not claim improvement.')


def parse():
    p=argparse.ArgumentParser(description='Parking V3 labelling, dataset adaptation & held-out evaluation')
    subs=p.add_subparsers(dest='action',required=True)
    for action in ('label','prepare','evaluate'):
        sp=subs.add_parser(action)
        sp.add_argument('--source',default=DEFAULT_SOURCE)
        sp.add_argument('--config',default=DEFAULT_CONFIG)
        sp.add_argument('--labels',default=DEFAULT_LABELS)
        sp.add_argument('--samples',type=int,default=7,help='Number of sampled timestamps (default 7; do not change mid-labeling)')
        sp.add_argument('--data-dir',default=DEFAULT_DIR)
        sp.add_argument('--output-dir',default=DEFAULT_OUT)
        sp.add_argument('--cnrpark-dir',default='dataset/real_cnrpark/train')
        sp.add_argument('--k',type=int,default=3)
    return p.parse_args()

if __name__=='__main__':
    a=parse()
    try:
        {'label':label,'prepare':prepare,'evaluate':evaluate}[a.action](a)
    except (OSError,RuntimeError,ValueError,cv2.error) as ex:
        raise SystemExit(f'ERROR: {ex}') from ex
