"""Compare model predictions CSV to independently labeled manual ground truth."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def load(path,frame_key,slot_key,label_key):
    with path.open(newline='',encoding='utf-8-sig') as f:
        out={}
        for row in csv.DictReader(f):
            label=row[label_key].strip().lower()
            if label not in ('empty','occupied'):continue
            key=(int(row[frame_key]),row[slot_key].strip())
            if key in out:raise ValueError(f'duplicate key: {key}')
            out[key]=label
        return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',default='results/parking_v3/predictions.csv')
    p.add_argument('--ground-truth',default='evaluation/parking_v3_ground_truth_template.csv')
    p.add_argument('--output',default='results/parking_v3/evaluation.json')
    a=p.parse_args()
    fix=lambda x:(Path(x) if Path(x).is_absolute() else ROOT/x)
    gt=load(fix(a.ground_truth),'frame','slot_id','true_status')
    if not gt:raise SystemExit('No manual true_status labels yet. Fill ground truth before evaluation.')
    pred=load(fix(a.predictions),'frame','slot_id','predicted_status')
    shared=gt.keys()&pred.keys();skipped=len(gt)-len(shared)
    if not shared:raise SystemExit('No common (frame, slot_id) between GT and predictions. Check --step and frame IDs.')
    confusion={a:{b:0 for b in ('empty','occupied')} for a in ('empty','occupied')}
    for key in shared:confusion[gt[key]][pred[key]]+=1
    correct=sum(gt[k]==pred[k] for k in shared)
    tp=confusion['occupied']['occupied'];fp=confusion['empty']['occupied'];fn=confusion['occupied']['empty']
    precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn) if tp+fn else 0
    f1=2*precision*recall/(precision+recall) if precision+recall else 0
    result={'labeled':len(gt),'evaluated':len(shared),'unmatched':skipped,'correct':correct,'accuracy':correct/len(shared),
            'precision_occupied':precision,'recall_occupied':recall,'f1_occupied':f1,
            'confusion':confusion,'caution':'Sparse manual sample only; do not generalize to all CCTV.'}
    outfile=fix(a.output);outfile.parent.mkdir(parents=True,exist_ok=True);outfile.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))
    print('Saved:',outfile)
if __name__=='__main__':main()
