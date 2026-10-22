"""Compare manually annotated aerial-video ROI labels with project KNN CSV predictions.

Only human-reviewed labels count. Empty template rows are excluded from metrics.
"""
import argparse
import csv
from pathlib import Path


def evaluate(predictions_csv, ground_truth_csv):
    predictions={}
    with Path(predictions_csv).open(newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            predictions[(int(r['frame']),r['slot_id'])]=r['predicted_status'].strip().lower()
    total=correct=0
    cm={a:{b:0 for b in ('empty','occupied')} for a in ('empty','occupied')}
    missing=[]
    with Path(ground_truth_csv).open(newline='',encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            actual=r['true_status'].strip().lower()
            if actual=='':continue
            if actual not in cm:
                raise ValueError(f"Unknown true_status {actual!r} for {r['frame']} {r['slot_id']}")
            key=(int(r['frame']),r['slot_id'])
            if key not in predictions:
                missing.append(key)
                continue
            predicted=predictions[key]
            if predicted not in ('empty','occupied'):
                raise ValueError(f'Invalid model prediction: {predicted!r}')
            cm[actual][predicted]+=1
            total+=1
            correct+=int(actual==predicted)
    print('Manual labels checked:',total)
    print('Missing prediction pairs:',len(missing))
    if missing:print('First missing pair:',missing[0])
    if total==0:
        print('No manually verified ground-truth labels yet. Fill true_status with empty or occupied.')
        return None
    print(f'Accuracy on manually reviewed ROI samples: {correct}/{total} = {100*correct/total:.2f}%')
    print('Confusion matrix [actual][predicted]:',cm)
    if total < 30:
        print('WARNING: Small manual sample; not a reliable general accuracy estimate.')
    return {'total':total,'correct':correct,'cm':cm}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',default='results/real_video_9010418/predictions.csv')
    p.add_argument('--ground-truth',default='evaluation/real_video_ground_truth_template.csv')
    args=p.parse_args()
    evaluate(args.predictions,args.ground_truth)

if __name__=='__main__':main()
