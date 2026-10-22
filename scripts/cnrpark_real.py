"""CNRPark real parking-slot image acquisition, grouped split, evaluation, gallery.

Official dataset: https://cnrpark.it/ (Amato et al.).
No dataset is embedded in this code distribution. Please respect source terms.
Requires project root `detector.py`, Python OpenCV 4.12.x and numpy.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
import re
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'dataset' / 'raw' / 'CNRPark-Patches-150x150.zip'
READY = ROOT / 'dataset' / 'real_cnrpark'
REPORTS = ROOT / 'results' / 'real_cnrpark'
SOURCES = (
    'https://cnrpark.it/dataset/CNRPark-Patches-150x150.zip',
    'http://cnrpark.it/dataset/CNRPark-Patches-150x150.zip',
)
IMAGE_EXT = ('.jpg', '.jpeg', '.png')


def source_entries(z: zipfile.ZipFile) -> list[dict]:
    """Read actual CNRPark path convention: <CAMERA>/<CLASS>/YYYYMMDD_HHMM_<SLOT>.jpg.

    Can tolerate root prefixed by a top-level directory, and uppercase labels.
    Group by same camera and capture timestamp, not individual patch image.
    """
    items=[]
    for p in z.namelist():
        if not p.lower().endswith(IMAGE_EXT) or p.endswith('/'):
            continue
        parts=p.replace('\\','/').split('/')
        for idx in range(len(parts)-2):
            camera=parts[idx].upper()
            cls=parts[idx+1].lower()
            if camera not in ('A','B') or cls not in ('free','busy'):
                continue
            filename=parts[-1]
            m=re.match(r'^(\d{8}_\d{4})_\d+\.(?:jpg|jpeg|png)$',filename,re.I)
            if not m:
                continue
            items.append({'member':p,'camera':camera,
                          'label':'empty' if cls=='free' else 'occupied',
                          'group':f'{camera}/{m.group(1)}',
                          'filename':filename})
            break
    return items


def choose_members(entries:list[dict],seed:int,train_per_class:int,test_per_class:int):
    if len(entries)<4:
        raise ValueError('Tidak cukup image berlabel pada ZIP. Pastikan ZIP CNRPark asli.')
    bygroup=defaultdict(list)
    for e in entries:
        bygroup[e['group']].append(e)
    groups=sorted(bygroup)
    if len(groups)<2:
        raise ValueError('Minimal dua capture timestamp berbeda diperlukan untuk split.')
    rng=random.Random(seed)
    rng.shuffle(groups)
    n_test=max(1,int(round(len(groups)*0.20)))
    n_test=min(n_test,len(groups)-1)
    testgroups=set(groups[:n_test])
    buckets={k:[] for k in [('train','empty'),('train','occupied'),('test','empty'),('test','occupied')]}
    for group in groups:
        partition='test' if group in testgroups else 'train'
        for e in bygroup[group]:
            buckets[(partition,e['label'])].append(e)
    chosen=[]
    for key in buckets:
        values=buckets[key]
        rng.shuffle(values)
        limit= train_per_class if key[0]=='train' else test_per_class
        if limit>0: values=values[:limit]
        if not values:
            raise ValueError(f'Split kosong untuk {key}; coba --seed berbeda atau cek ZIP')
        for e in values:
            chosen.append({**e,'split':key[0]})
    return chosen


def download(args):
    dest=Path(args.archive).resolve()
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.is_file() and zipfile.is_zipfile(dest):
        print(f'Archive already available: {dest} ({dest.stat().st_size / 1048576:.1f} MiB)')
        return
    errors=[]
    for url in SOURCES:
        partial=dest.with_suffix(dest.suffix+'.part')
        try:
            print(f'Downloading dataset from: {url}',flush=True)
            req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 SmartParkingClassProject/1.0'})
            with urllib.request.urlopen(req,timeout=50) as response, partial.open('wb') as out:
                while True:
                    buf=response.read(1024*1024)
                    if not buf: break
                    out.write(buf)
                    if out.tell() > 120*1024*1024:
                        raise RuntimeError('Archive larger than expected (120 MB): abort')
            if not zipfile.is_zipfile(partial):
                raise RuntimeError('Not a valid ZIP (may be a server error page)')
            partial.replace(dest)
            print(f'OK: {dest} ({dest.stat().st_size / 1048576:.1f} MiB)')
            return
        except Exception as exc:
            errors.append(f'{url} -> {exc}')
            partial.unlink(missing_ok=True)
    raise RuntimeError('Cannot download official dataset.\n'+'\n'.join(errors)+
                       '\nManual alternative: open https://cnrpark.it/ and save CNRPark-Patches-150x150.zip to '+str(dest))


def prepare(args):
    archive=Path(args.archive).resolve()
    output=Path(args.output).resolve()
    if not archive.is_file() or not zipfile.is_zipfile(archive):
        raise FileNotFoundError(f'Official dataset archive missing/invalid: {archive}\nRun: python scripts/cnrpark_real.py download')
    if (output/'split_manifest.csv').exists() and not args.force:
        raise FileExistsError(f'Split exists: {output}. Use --force only to regenerate it.')
    with zipfile.ZipFile(archive) as z:
        entries=source_entries(z)
        if not entries:
            preview='\n'.join(z.namelist()[:15])
            raise ValueError('Cannot find A/free and A/busy / B/... images in ZIP. First paths:\n'+preview)
        chosen=choose_members(entries,args.seed,args.train_per_class,args.test_per_class)
        output.mkdir(parents=True,exist_ok=True)
        if args.force:
            for folder in ('train','test'):
                shutil.rmtree(output/folder,ignore_errors=True)
        for e in chosen:
            folder=output/e['split']/e['label']
            folder.mkdir(parents=True,exist_ok=True)
            name=e['camera']+'_'+e['filename']
            destination=folder/name
            if destination.exists():
                raise FileExistsError(f'Duplicate filename detected: {destination}')
            with z.open(e['member']) as original, destination.open('wb') as target:
                shutil.copyfileobj(original,target)
            e['output_path']=str(destination.relative_to(output)).replace('\\','/')
    chosen.sort(key=lambda x:(x['split'],x['label'],x['output_path']))
    with (output/'split_manifest.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['split','label','camera','group','member','output_path'],extrasaction='ignore')
        writer.writeheader();writer.writerows(chosen)
    counts=Counter((e['split'],e['label']) for e in chosen)
    summary={'source':'CNRPark original 150x150 JPEG patch dataset, https://cnrpark.it/',
             'images_found_in_source_zip':len(entries),
             'split':'group by camera + capture timestamp, random seeded 80/20 groups',
             'seed':args.seed,'counts':{f'{s}/{c}':counts[(s,c)] for s in ('train','test') for c in ('empty','occupied')},
             'train_capture_groups':len(set(e['group'] for e in chosen if e['split']=='train')),
             'test_capture_groups':len(set(e['group'] for e in chosen if e['split']=='test')),
             'note':'Same slot/camera across timestamps; this is not cross-camera or unseen-site evaluation.'}
    (output/'dataset_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('Source images:',len(entries))
    print('Dataset ready:',output)
    print(json.dumps(summary['counts'],indent=2))


def evaluate(args):
    output=Path(args.output).resolve()
    test_root=output/'test'
    train_root=output/'train'
    for cls in ('empty','occupied'):
        if not list((test_root/cls).glob('*.jpg')) and not list((test_root/cls).glob('*.png')):
            raise FileNotFoundError(f'No test images in {test_root/cls}. Run prepare first.')
    sys.path.insert(0,str(ROOT))
    from detector import ParkingDetector
    model=ParkingDetector(train_root,k=args.k)
    targets=[];predictions=[];details=[]
    for ground_truth in ('empty','occupied'):
        for path in sorted((test_root/ground_truth).iterdir()):
            if path.suffix.lower() not in IMAGE_EXT: continue
            img=cv2.imread(str(path))
            if img is None: raise RuntimeError(f'Unreadable image: {path}')
            h,w=img.shape[:2]
            classified=model.predict(img,[{'id':'test','x':0,'y':0,'w':w,'h':h}])[0]['status']
            targets.append(ground_truth); predictions.append(classified)
            details.append({'image':str(path.relative_to(output)).replace('\\','/'), 'actual':ground_truth,
                            'predicted':classified,'correct':int(ground_truth==classified)})
    if not targets:raise RuntimeError('No images evaluated.')
    classes=['empty','occupied']
    by_class={}
    for cls in classes:
        tp=sum(a==p==cls for a,p in zip(targets,predictions))
        fp=sum(a!=cls and p==cls for a,p in zip(targets,predictions))
        fn=sum(a==cls and p!=cls for a,p in zip(targets,predictions))
        by_class[cls]={'precision':round(tp/(tp+fp),4) if tp+fp else 0.0,
                       'recall':round(tp/(tp+fn),4) if tp+fn else 0.0,
                       'f1':round(2*tp/(2*tp+fp+fn),4) if 2*tp+fp+fn else 0.0,
                       'support':sum(a==cls for a in targets)}
    correct=sum(a==b for a,b in zip(targets,predictions))
    report={'dataset':'REAL CNRPark parking-space patches',
            'validation':'holdout separated by camera+capture timestamp; no identical capture group in training/test',
            'limitations':'same cameras and parking slots may exist across groups; evaluation is crop classification, NOT real-video detection',
            'train_dir':str(train_root),'test_count':len(targets),'correct':correct,
            'accuracy':round(correct/len(targets),4),
            'confusion_matrix':{actual:{pred:sum(a==actual and p==pred for a,p in zip(targets,predictions)) for pred in classes} for actual in classes},
            'per_class':by_class,'knn_k':args.k}
    report_dir=Path(args.reports).resolve()
    report_dir.mkdir(parents=True,exist_ok=True)
    (report_dir/'evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    with (report_dir/'predictions.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['image','actual','predicted','correct'])
        writer.writeheader();writer.writerows(details)
    print('REAL CNRPark patch evaluation (NOT CCTV video evaluation)')
    print('Accuracy:',f'{correct}/{len(targets)} = {100*correct/len(targets):.2f}%')
    print('Confusion matrix:',json.dumps(report['confusion_matrix']))
    print('Reports:',report_dir)


def gallery(args):
    output=Path(args.output).resolve()
    report_dir=Path(args.reports).resolve()
    pred_file=report_dir/'predictions.csv'
    if not pred_file.exists(): raise FileNotFoundError(f'Run evaluate first: {pred_file}')
    with pred_file.open(encoding='utf-8',newline='') as f: rows=list(csv.DictReader(f))
    rng=random.Random(2026)
    rng.shuffle(rows)
    selected=rows[:min(20,len(rows))]
    tile_w,tile_h=210,210
    cols=5; rows_n=max(1,(len(selected)+cols-1)//cols)
    canvas=np.full((60+rows_n*tile_h,cols*tile_w,3),245,dtype=np.uint8)
    cv2.rectangle(canvas,(0,0),(cols*tile_w,58),(32,39,55),-1)
    cv2.putText(canvas,'CNRPark REAL PHOTOS - TEST CROPS (NOT CCTV VIDEO)',(15,36),cv2.FONT_HERSHEY_SIMPLEX,.54,(255,255,255),1,cv2.LINE_AA)
    for i,item in enumerate(selected):
        img=cv2.imread(str(output/item['image']))
        if img is None:continue
        img=cv2.resize(img,(180,150),interpolation=cv2.INTER_AREA)
        x=(i%cols)*tile_w+15;y=60+(i//cols)*tile_h+8
        canvas[y:y+150,x:x+180]=img
        color=(40,145,46) if item['correct']=='1' else (20,20,220)
        label=f"{item['actual']} -> {item['predicted']}"
        cv2.putText(canvas,label,(x,y+174),cv2.FONT_HERSHEY_SIMPLEX,.43,color,1,cv2.LINE_AA)
    report_dir.mkdir(parents=True,exist_ok=True)
    path=report_dir/'real_test_gallery.jpg'
    if not cv2.imwrite(str(path),canvas):raise RuntimeError(f'Cannot write {path}')
    print('Gallery:',path)
    print('Gallery is a collage of actual CNRPark test crops, not a full-frame CCTV detection.')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description='Official REAL CNRPark patches -> KNN experiment')
    sub=p.add_subparsers(dest='command',required=True)
    for action in ('download','prepare','evaluate','gallery','all'):
        s=sub.add_parser(action)
        s.add_argument('--archive',default=str(RAW))
        s.add_argument('--output',default=str(READY))
        s.add_argument('--reports',default=str(REPORTS))
        s.add_argument('--seed',type=int,default=2026)
        s.add_argument('--train-per-class',type=int,default=800,help='0 means all selected train patches')
        s.add_argument('--test-per-class',type=int,default=160,help='0 means all selected test patches')
        s.add_argument('--k',type=int,default=3)
        s.add_argument('--force',action='store_true',help='Overwrite the previously prepared dataset split')
    return p.parse_args(argv)


def main(argv=None):
    a=parse_args(argv)
    if a.command in ('download','all'):download(a)
    if a.command in ('prepare','all'):prepare(a)
    if a.command in ('evaluate','all'):evaluate(a)
    if a.command in ('gallery','all'):gallery(a)

if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError,zipfile.BadZipFile) as e:
        raise SystemExit('ERROR: '+str(e)) from e
