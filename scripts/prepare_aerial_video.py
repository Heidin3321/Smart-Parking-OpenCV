"""Stabilize aerial parking video relative to a reference frame using OpenCV ORB.

Requires opencv-python and numpy only. Frame transforms are camera alignment,
not vehicle tracking. Downsamples 4K source to 1280x720 at about 10 fps.
"""
import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def resize_frame(frame, width=1280, height=720):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


class FrameStabilizer:
    """Estimate affine map from current frame into 10-second reference image."""
    def __init__(self, reference):
        self.dim=(reference.shape[1],reference.shape[0])
        self.orb=cv2.ORB_create(nfeatures=1600,fastThreshold=10)
        self.matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
        self.ref_small=cv2.resize(reference,(640,360),interpolation=cv2.INTER_AREA)
        self.ref_kp, self.ref_desc=self.orb.detectAndCompute(cv2.cvtColor(self.ref_small,cv2.COLOR_BGR2GRAY),None)
        if self.ref_desc is None or len(self.ref_kp)<60:
            raise ValueError("Cannot detect enough reference features for stabilization")
        self.previous=np.float32([[1,0,0],[0,1,0]])
        self.success=0
        self.failed=0

    def align(self, frame):
        small=cv2.resize(frame,(640,360),interpolation=cv2.INTER_AREA)
        kp,desc=self.orb.detectAndCompute(cv2.cvtColor(small,cv2.COLOR_BGR2GRAY),None)
        affine=None
        if desc is not None and len(kp)>=30:
            pairs=self.matcher.knnMatch(desc,self.ref_desc,k=2)
            good=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance < 0.76*n.distance]
            if len(good)>=18:
                a=np.float32([kp[m.queryIdx].pt for m in good])
                b=np.float32([self.ref_kp[m.trainIdx].pt for m in good])
                mat,mask=cv2.estimateAffinePartial2D(a,b,method=cv2.RANSAC,
                    ransacReprojThreshold=2.5,maxIters=2200,confidence=0.995)
                if mat is not None and mask is not None and mask.sum()>=16:
                    scale=math.hypot(mat[0,0],mat[0,1])
                    rotation=math.degrees(math.atan2(mat[1,0],mat[0,0]))
                    if 0.94 <= scale <=1.06 and abs(rotation)<4 and abs(mat[0,2])<40 and abs(mat[1,2])<40:
                        affine=mat.astype(np.float32)
        if affine is None:
            affine=self.previous
            self.failed+=1
        else:
            self.previous=affine
            self.success+=1
        # Affine was estimated on 640x360; double translation for 1280x720.
        full=affine.copy()
        full[:,2]*=2
        aligned=cv2.warpAffine(frame, full, self.dim,
            flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,
            borderValue=(15,20,20))
        return aligned


def create_stabilized_video(src, dst, ref_seconds=10.0, out_width=1280, out_height=720, target_fps=10.0, max_frames=0):
    cap=cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source video: {src}")
    fps=cap.get(cv2.CAP_PROP_FPS)
    if fps<=0: raise ValueError("Cannot read input FPS")
    interval=max(1,round(fps/target_fps))
    target_fps=fps/interval
    cap.set(cv2.CAP_PROP_POS_MSEC,ref_seconds*1000)
    ok,ref=cap.read()
    if not ok: raise ValueError("Reference frame is not available at selected timestamp")
    if ref.shape[1]*out_height != ref.shape[0]*out_width:
        raise ValueError("Output resolution must have same aspect ratio as input")
    reference=resize_frame(ref,out_width,out_height)
    stabilizer=FrameStabilizer(reference)
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    dst=Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer=cv2.VideoWriter(str(dst),cv2.VideoWriter_fourcc(*'mp4v'),target_fps,(out_width,out_height))
    if not writer.isOpened(): raise RuntimeError(f"Failed to create {dst}")
    index=processed=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok:break
            if index%interval==0:
                resized=resize_frame(frame,out_width,out_height)
                result=stabilizer.align(resized)
                writer.write(result)
                processed+=1
                if max_frames and processed>=max_frames:break
            index+=1
    finally:
        writer.release()
        cap.release()
    print(f"Written {processed} frames at {target_fps:.3f} fps: {dst}")
    print(f"ORB stabilized frames: {stabilizer.success}; fallback frames: {stabilizer.failed}")
    return processed


def main():
    p=argparse.ArgumentParser(description="Generate stabilized 720p fixed-ROI test footage")
    p.add_argument('--source',required=True,help="Original aerial parking MP4 or already downsized MP4")
    p.add_argument('--output',default='dataset/parking_real_720p_stabilized.mp4')
    p.add_argument('--reference-time',type=float,default=10.0)
    p.add_argument('--target-fps',type=float,default=10.0)
    p.add_argument('--max-frames',type=int,default=0)
    args=p.parse_args()
    create_stabilized_video(args.source,args.output,args.reference_time,target_fps=args.target_fps,max_frames=args.max_frames)

if __name__=='__main__':main()
