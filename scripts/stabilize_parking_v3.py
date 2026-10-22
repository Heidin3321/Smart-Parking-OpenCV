"""Stabilize overhead video to the FIRST frame using ORB + RANSAC affine alignment.

Output holds fixed parking coordinates only as long as alignment succeeds.
No vehicle occupancy labels are produced by stabilization.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def resolve(v: str) -> Path:
    p=Path(v)
    return p if p.is_absolute() else ROOT/p


def estimate_reference_to_current(reference_gray, gray, detector, ref_kp, ref_desc, matcher, fallback):
    kp, desc = detector.detectAndCompute(gray, None)
    if desc is None or ref_desc is None or len(kp) < 8:
        return fallback, False
    pairs=matcher.knnMatch(ref_desc,desc,k=2)
    good=[a for a,b in pairs if a.distance < 0.75*b.distance]
    if len(good) < 9:
        return fallback, False
    src=np.float32([ref_kp[m.queryIdx].pt for m in good])
    dst=np.float32([kp[m.trainIdx].pt for m in good])
    affine, inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=2.5,maxIters=1500)
    if affine is None or inliers is None or int(inliers.sum())<9:
        return fallback, False
    scale=(affine[0,0]**2+affine[1,0]**2)**.5
    if not .96<scale<1.04 or max(abs(affine[0,2]),abs(affine[1,2]))>60:
        return fallback, False
    return affine, True


def stabilize(source: Path, output: Path, stride=3, limit=0):
    if stride<1:raise ValueError('stride must be >= 1')
    cap=cv2.VideoCapture(str(source))
    if not cap.isOpened():raise FileNotFoundError(f'Cannot open source: {source}')
    fps=cap.get(cv2.CAP_PROP_FPS) or 30
    ok,reference=cap.read()
    if not ok:raise RuntimeError('Video contains no frames')
    h,w=reference.shape[:2]
    gray0=cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY)
    orb=cv2.ORB_create(nfeatures=1400)
    kp0,desc0=orb.detectAndCompute(gray0,None)
    matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    output.parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),fps/stride,(w,h))
    if not writer.isOpened():raise RuntimeError(f'Cannot write {output}')
    ident=np.array([[1.,0.,0.],[0.,1.,0.]],dtype=np.float64)
    prev=ident.copy(); written=0; aligned=0; failed=0
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    index=0
    try:
        while True:
            if limit and written>=limit:break
            ok,frame=cap.read()
            if not ok:break
            if index%stride:
                index+=1;continue
            if index==0:
                stabilized=frame
            else:
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                matrix,valid=estimate_reference_to_current(gray0,gray,orb,kp0,desc0,matcher,prev)
                if valid:aligned+=1;prev=matrix
                else: failed+=1
                # M maps reference locations -> current positions, so WARP_INVERSE_MAP
                # samples each output coordinate from corresponding source coordinates.
                stabilized=cv2.warpAffine(frame,matrix,(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,
                                          borderMode=cv2.BORDER_CONSTANT,borderValue=(35,35,35))
            writer.write(stabilized)
            written+=1;index+=1
    finally:
        cap.release();writer.release()
    if not written:raise RuntimeError('No frames written')
    print(f'Stabilized frames: {written} | output FPS: {fps/stride:.2f} | aligned: {aligned} | fallback: {failed}')
    print(f'Output: {output} | resolution: {w}x{h}')
    return written


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='dataset/parking_v3_original.mp4')
    p.add_argument('--output',default='dataset/parking_v3_stabilized.mp4')
    p.add_argument('--stride',type=int,default=3)
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args()
    stabilize(resolve(a.source),resolve(a.output),a.stride,a.limit)

if __name__=='__main__':main()
