"""Required HUMAN BLIND LABELING + temporal holdout EVALUATION + motion-aware MP4.

Used by parking_roi_wizard.py after manual / automatic / saved ROI selection.
Only CALIBRATION frames enter the scene-specific training gallery.
TEST labels are never added to training, and are recorded before predictions.
"""
from __future__ import annotations

import csv
import html
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

import test_new_parking_video as base
import run_motion_aware_parking as motion


class MotionSampler:
    """Crops ROIs in source-video frames at MOTION-ADJUSTED coordinates."""
    def __init__(self, cap, cfg):
        self.cap = cap
        self.cfg = cfg
        self.size = (cfg['frame_width'], cfg['frame_height'])
        self.anchor = cfg['sampling_plan']['calibration'][0]
        self.ref = base.frame_at(cap, self.anchor, self.size)
        self.tracker = motion.CameraMotion(self.ref)
        self.cache = {}

    def at(self, frame_index):
        if frame_index in self.cache:
            return self.cache[frame_index]
        frame = self.ref.copy() if frame_index == self.anchor else base.frame_at(self.cap, frame_index, self.size)
        if frame_index == self.anchor:
            mat = np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float64)
            quality = 'ANCHOR'
        else:
            mat, quality, _, _ = self.tracker.estimate(frame)
        h, w = frame.shape[:2]
        roi_map = {s['id']:(moving,valid and quality!='LOST') for s in self.cfg['slots']
                   for moving, valid in [motion.transform_slot(s, mat, w, h)]}
        self.cache[frame_index] = frame, roi_map, quality, mat
        return self.cache[frame_index]

    def patch(self, frame_index, slot_id):
        frame, roi_map, quality, _ = self.at(frame_index)
        s, valid = roi_map[slot_id]
        if not valid:
            return None
        x,y,w,h=[s[k] for k in ('x','y','w','h')]
        return frame[y:y+h,x:x+w]


def label_blind(sampler, cfg, paths):
    """Interactively label CALIBRATION then TEST before ANY predictions.

    O/E/U keys; B revises an earlier answer; Q saves partial progress.
    Resume using the same video + existing ROI.
    """
    data = base.load_labels(paths, cfg)
    tasks = base.labeled_entries(cfg)
    labels = data['labels']
    idx = next((i for i,(phase,fi,s) in enumerate(tasks)
                if base.label_key(phase,fi,s['id']) not in labels),len(tasks))
    print('\n'+'='*68)
    print('MANDATORY HUMAN LABELS: CALIBRATION & TEST FRAMES (NO AI PREDICTION).')
    print(f'Progress {len(labels)}/{len(tasks)} | O=Occupied E=Empty U=Unknown B=Back Q=Save/Exit')
    print('Label only the slot highlighted; U when tracking is lost or image is ambiguous.')
    print('='*68)
    if idx == len(tasks):
        print('All labels already saved. To revise a specific entry, edit JSON carefully or use a new video.')
        return data

    window='SMART PARKING | LABEL SLOT (O/E/U), B=BACK, Q=SAVE'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window,1060,850)
    try:
        while idx < len(tasks):
            phase,fi,original = tasks[idx]
            sid = original['id']
            frame, roi_map, quality, _ = sampler.at(fi)
            s, ok = roi_map[sid]
            x,y,w,h=[s[k] for k in ('x','y','w','h')]
            # Scene display is at native resized video width; append status bar.
            H,W=frame.shape[:2]
            canvas=np.full((H+215,max(W,900),3),(27,32,41),dtype=np.uint8)
            canvas[:H,:W] = frame
            color = (0,215,255) if ok else (0,0,255)
            cv2.rectangle(canvas,(x,y),(x+w,y+h),color,3)
            if ok and w>0 and h>0:
                crop = frame[y:y+h,x:x+w]
                factor=min(180/max(h,1),250/max(w,1),3)
                nw,nh=max(1,int(w*factor)),max(1,int(h*factor))
                zoom=cv2.resize(crop,(nw,nh),interpolation=cv2.INTER_LINEAR)
                oy=H+16
                if oy+nh<=canvas.shape[0]:
                    canvas[oy:oy+nh,20:20+nw]=zoom
            status=f'FRAME {fi} | {phase.upper()} | SLOT {sid} | {idx+1}/{len(tasks)}'
            base.put(canvas,status,290,H+42,.55,(235,235,235),2)
            base.put(canvas,'O = OCCUPIED     E = EMPTY     U = UNKNOWN',290,H+77,.48,(0,220,250),1)
            base.put(canvas,'B = BACK          Q = SAVE AND EXIT',290,H+112,.48,(230,230,230),1)
            base.put(canvas,f'TRACKING: {quality}'+(' - PRESS U' if not ok else ''),290,H+147,.43,
                     (50,80,255) if not ok else (130,235,130),1)
            old=labels.get(base.label_key(phase,fi,sid),'-')
            base.put(canvas,f'PREVIOUS LABEL: {old}',290,H+178,.43,(230,230,230),1)
            cv2.imshow(window,canvas)
            key=cv2.waitKey(0)&0xff
            label={ord('o'):'occupied',ord('e'):'empty',ord('u'):'unknown'}.get(key)
            if label:
                if not ok and label != 'unknown':
                    print(f'WARNING: frame {fi}, slot {sid}: TRACKING LOST. Press U for unknown.'); continue
                labels[base.label_key(phase,fi,sid)] = label
                base.save_json(paths['labels'],data)
                idx+=1
            elif key==ord('b'):
                idx=max(0,idx-1)
            elif key in (ord('q'),27):
                break
    finally:
        cv2.destroyWindow(window)
    print(f'Saved {len(labels)}/{len(tasks)} labels at {paths["labels"]}')
    if len(labels)<len(tasks):
        raise RuntimeError('Pelabelan BELUM SELESAI. Ulangi command yang sama -> pilih 3 EXISTING untuk melanjutkan. Tidak ada training/video sebelum labeling selesai.')
    return data


def build_scene_models(sampler,cfg,paths,data):
    """Train solely on calibration crops aligned by camera-motion tracking."""
    gallery=defaultdict(list)
    train=paths['train']
    for label in base.CLASSES:
        folder=train/label
        folder.mkdir(parents=True,exist_ok=True)
        for old in folder.glob('*.png'): old.unlink()
    count=Counter()
    for fi in cfg['sampling_plan']['calibration']:
        for s in cfg['slots']:
            lab=data['labels'][base.label_key('calibration',fi,s['id'])]
            patch=sampler.patch(fi,s['id'])
            if lab not in base.CLASSES or patch is None:continue
            gallery[s['id']].append((lab,base.feature(patch)))
            filename=train/lab/f'F{fi}_{s["id"]}.png'
            if not cv2.imwrite(str(filename),patch):
                raise OSError(f'Cannot save crop {filename}')
            count[lab]+=1
    print('Motion-aligned CALIBRATION train examples:',dict(count))
    models={}
    if sum(count.values())>0:
        models['new_scene_reference']=base.SceneReference(gallery)
    if all(count[klass] for klass in base.CLASSES):
        n=sum(count.values())
        models['new_scene_knn']=base.ParkingDetector(train,k=min(3,n))
    else:
        print('WARNING: Scene KNN skipped: training needs examples of BOTH EMPTY and OCCUPIED.')
    if base.has_training_images(base.SOURCE_TRAIN):
        models['cnrpark_pretrained']=base.ParkingDetector(base.SOURCE_TRAIN,k=3)
    if not models:
        raise ValueError('No usable model: calibration label examples must include at least one valid ROI; or set up CNRPark.')
    return models,count


def evaluate_holdout(sampler,cfg,paths,data,models):
    """All test truths come from later frames, never from calibration frames."""
    out=paths['out'];out.mkdir(parents=True,exist_ok=True)
    scores={}; entries=[]
    for name, model in models.items():
        actual={}; predictions={}
        for fi in cfg['sampling_plan']['test']:
            fr,roi_map,quality,_=sampler.at(fi)
            good=[r for r,ok in roi_map.values() if ok]
            mapping={v['id']:v['status'] for v in model.predict(fr,good)} if good else {}
            for s in cfg['slots']:
                sid=s['id'];key=f'{fi}:{sid}'
                truth=data['labels'][base.label_key('test',fi,sid)]
                pred=mapping.get(sid,'uncertain')
                actual[key]=truth; predictions[key]=pred
                entries.append({'model':name,'frame':fi,'slot_id':sid,
                                'ground_truth':truth,'predicted':pred,
                                'tracking':quality,
                                'verdict':'excluded' if truth=='unknown' else ('correct' if truth==pred else 'wrong')})
        scores[name]=base.evaluate(actual,predictions)
        s=scores[name]
        print(f'Holdout {name}: {s["correct"]}/{s["tested"]} correct, accuracy={s["accuracy"]}, uncertain={s["uncertain"]}')
    with (out/'holdout_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=['model','frame','slot_id','ground_truth','predicted','tracking','verdict'])
        wr.writeheader();wr.writerows(entries)
    return scores


def write_report(paths,cfg,data,scores,count,video_output,stats):
    out=paths['out'];out.mkdir(parents=True,exist_ok=True)
    report={'protocol':'CALIBRATION: early frames / TEST: later frames of SAME VIDEO (temporal holdout).',
            'not_independent_video_test': True,
            'roi_count':len(cfg['slots']),
            'roi_creation_mode':cfg.get('roi_creation_mode','unknown'),
            'sampling_plan':cfg['sampling_plan'],
            'calibration_label_counts':dict(count),
            'test_label_counts':dict(Counter(data['labels'][base.label_key('test',fi,s['id'])]
                                             for fi in cfg['sampling_plan']['test'] for s in cfg['slots'])),
            'scores':scores,'motion_tracking_frames':stats,
            'primary_video':str(video_output)}
    base.save_json(out/'report.json',report)
    sections=[]
    for name,score in scores.items():
        accuracy=f'{score["accuracy"]*100:.1f}%' if score['accuracy'] is not None else 'N/A'
        sections.append(f'<tr><td>{html.escape(name)}</td><td>{score["correct"]}/{score["tested"]}</td>'
                        f'<td>{accuracy}</td><td>{score["uncertain"]}</td></tr>')
    page='''<!doctype html><html lang="id"><meta charset="utf-8"><title>Smart Parking | ROI & Video Testing</title>
    <style>body{font:16px Arial,sans-serif;background:#f4f6fa;color:#172033;margin:40px auto;max-width:940px;padding:0 24px}
    h1{font-size:28px}section{background:#fff;border-radius:12px;padding:22px;margin:18px 0;box-shadow:0 3px 12px #ccd3dd55}
    table{border-collapse:collapse;width:100%}th,td{border:1px solid #ddd;padding:12px;text-align:left}
    th{background:#edf1f9}a{color:#1764bd}video{max-width:100%}</style><main>
    <h1>SMART PARKING — NEW VIDEO TESTING</h1>
    <section><h2>Annotated Video with ROI</h2><video controls src="annotated_video_motion_roi.mp4"></video>
    <p><a href="annotated_video_motion_roi.mp4">Open MP4 video</a> · <a href="holdout_predictions.csv">Per-slot testing CSV</a> · <a href="report.json">JSON metrics</a></p></section>
    <section><h2>Test result</h2><table><tr><th>Model</th><th>Correct / Tested</th><th>Accuracy</th><th>UNCERTAIN</th></tr>'''+''.join(sections)+'''</table>
    <p>This is within-video temporal holdout testing. Calibration comes from early frames only; late-frame test labels do not enter training.
    Frames can be highly correlated and ROI tracking can fail; this does NOT establish generalization to a different recording.</p></section>
    <section><h2>Protocol</h2><p>Manual or Auto ROI -> blind human calibration labels -> blind human holdout labels -> train scene models -> test -> motion-aware annotated MP4.</p></section></main></html>'''
    (out/'report.html').write_text(page,encoding='utf-8')
    print('Evaluation report:',out/'report.html')
    return report


def execute(video, cfg, paths, meta, detector_mode='auto', max_seconds=0, output_fps=10, no_open=False):
    cap,_=base.open_video(video)
    try:
        sampler=MotionSampler(cap,cfg)
        data=label_blind(sampler,cfg,paths)
        models,count=build_scene_models(sampler,cfg,paths,data)
        scores=evaluate_holdout(sampler,cfg,paths,data,models)
        if detector_mode=='auto':
            priority=['new_scene_reference','new_scene_knn','cnrpark_pretrained']
            choice=next((n for n in priority if n in models),None)
        else:
            aliases={'scene_reference':'new_scene_reference','scene_knn':'new_scene_knn',
                     'cnrpark':'cnrpark_pretrained'}
            choice=aliases.get(detector_mode,detector_mode)
        if choice not in models:
            raise ValueError(f'Detector {detector_mode!r} unavailable. Available: {list(models)}')
        print('\nRENDERING ANNOTATED MP4 WITH MOVING ROI (AFTER HUMAN LABELS)...')
        output,stats=motion.run(video,'auto',max_seconds,output_fps,True,
                                detector_override=(choice,models[choice]))
        write_report(paths,cfg,data,scores,count,output,stats)
        print('\n'+'='*70)
        print('SUCCESS: MANDATORY LABELING -> TRAINING -> TESTING -> ANNOTATED MP4')
        print('VIDEO:',output)
        print('REPORT:',paths['out']/'report.html')
        print('='*70)
        if not no_open:base.open_video_result(output,skip=False)
        return output,scores
    finally:
        cap.release()
