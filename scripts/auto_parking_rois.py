"""Conservative OpenCV proposals for TOP-DOWN parking ROIs.

Detects bright painted parking-space dividers. Produces rectangular *candidates*,
not guaranteed parking-slot ground truth. Users MUST verify positions before use.
Offline; no model downloads required.
"""
from __future__ import annotations

from collections import defaultdict
import cv2
import numpy as np


def _groups(vals, gap=2):
    vals = sorted({int(v) for v in vals})
    groups = []
    for v in vals:
        if not groups or v - groups[-1][-1] > gap:
            groups.append([v])
        else:
            groups[-1].append(v)
    return groups


def _estimate_center_lines(bright, max_columns=3):
    """Long, bright vertical stripes usually separate facing rows of parking bays."""
    h, w = bright.shape
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, int(h * .14))))
    opening = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k)
    density = (opening > 0).sum(axis=0)
    min_density = h * .17
    xs = np.where(density >= min_density)[0]
    candidates = []
    for g in _groups(xs, gap=max(2, int(w * .008))):
        x = int(np.median(g))
        if w * .19 <= x <= w * .81:
            candidates.append((float(density[g].max()), x))
    candidates.sort(reverse=True)
    selected = []
    for _, x in candidates:
        if all(abs(x - v) > w * .20 for v in selected):
            selected.append(x)
        if len(selected) == max_columns - 1:
            break
    return sorted(selected)


def _row_markings(bright, x0, x1):
    """Thin horizontal white strokes with good span are potential bay boundaries."""
    h, w = bright.shape
    region = bright[:, x0:x1]
    rw = region.shape[1]
    if rw < 30:
        return []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(max(10,round(rw*.20)),1))
    opened = cv2.morphologyEx(region, cv2.MORPH_OPEN, kernel)
    active = (opened > 0)
    row_count = active.sum(axis=1)
    ys = np.where(row_count >= max(16,rw * .34))[0]
    result=[]
    for g in _groups(ys,gap=1):
        if len(g) > max(7,int(h*.018)):
            # Thick roofs/walls, not thin parking lines.
            continue
        y=int(round(np.median(g)))
        xs=np.where(active[g].any(axis=0))[0]
        if len(xs) < rw*.32:
            continue
        left=max(0,int(np.percentile(xs,4))-2)
        right=min(rw,int(np.percentile(xs,96))+3)
        if right-left < rw*.35:
            continue
        result.append((y,x0+left,x0+right,float(row_count[g].max()/rw)))
    return result


def _stable_grid(markings, height):
    """Fit regular y positions to long white divider lines and ignore roof edges."""
    if len(markings) < 3:
        return []
    ys=np.array(sorted(set(int(m[0]) for m in markings)))
    diffs=np.diff(ys)
    # Exclude short false horizontal edges from car roofs and very large gaps.
    valid=diffs[(diffs>=height*.08)&(diffs<=height*.27)]
    if len(valid)<2:
        # Some stripes can be hidden; consider pairwise larger gaps if needed.
        gaps=np.array([abs(int(b-a)) for i,a in enumerate(ys) for b in ys[i+1:]])
        valid=gaps[(gaps>=height*.08)&(gaps<=height*.23)]
    if len(valid)<2:return []
    # Most frequent distance (allow mild line-to-line jitter).
    choices=sorted(set(int(round(v)) for v in valid))
    spacing=max(18.,float(max(choices,key=lambda p:sum(abs(valid-p)<=max(4,p*.09)))))
    if not (height*.045<=spacing<=height*.28):return []
    tolerance=max(4.,spacing*.12)
    anchor=max(ys,key=lambda a:sum(abs(((ys-a+spacing/2)%spacing)-spacing/2)<=tolerance))
    min_y=int(min(ys));max_y=int(max(ys))
    candidates=[int(round(anchor+i*spacing)) for i in range(-round(height/spacing)-2,round(height/spacing)+3)]
    strong=[]
    for c in candidates:
        if not(0<=c<height):continue
        near=[(abs(c-int(m[0])),int(m[0])) for m in markings if abs(c-int(m[0]))<=tolerance]
        if near:strong.append(min(near)[1])
        elif min_y<=c<=max_y:strong.append(c)
    result=sorted(set(strong))
    if len(result)<3:return []
    return result


def _make_bays(markings, band_left, band_right, height, slot_limit, global_dividers=None):
    if not markings and not global_dividers:
        return []
    own_grid=_stable_grid(markings,height)
    if global_dividers and len(global_dividers)>len(own_grid):
        boundaries=list(global_dividers)
    else:
        boundaries=list(own_grid)
    if len(boundaries)<3:return []
    diffs=np.diff(boundaries)
    spacing=float(np.median(diffs))
    if boundaries[0]>spacing*.48 and boundaries[0]<spacing*1.55:
        boundaries.insert(0,max(0,round(boundaries[0]-spacing)))
    if height-boundaries[-1]>spacing*.55:
        boundaries.append(min(height,round(boundaries[-1]+spacing)))
    rois=[]
    # Estimate x-range from painted lines in this column; don't learn from other columns.
    if markings:
        xleft=int(np.median([m[1] for m in markings]));xright=int(np.median([m[2] for m in markings]))
    else:
        xleft=band_left;xright=band_right
    for a,b in zip(boundaries,boundaries[1:]):
        gap=b-a
        if not (.70*spacing<=gap<=1.30*spacing):continue
        adjacent=[m for m in markings if abs(m[0]-a)<spacing*.22 or abs(m[0]-b)<spacing*.22]
        lx=int(np.median([v[1] for v in adjacent])) if adjacent else xleft
        rx=int(np.median([v[2] for v in adjacent])) if adjacent else xright
        lx=max(band_left,lx);rx=min(band_right,rx)
        # Front bumpers can hide painted-line endpoints in the leftmost column.
        # Use the furthest observed line edge plus a small outward margin.
        if band_left == 0 and markings:
            lx=min(lx,min(int(m[1]) for m in markings))-round((band_right-band_left)*.085)
        x0=max(band_left,lx+max(0,round((rx-lx)*.015)))
        x1=rx-max(0,round((rx-lx)*.015))
        y0=max(0,a+2);y1=min(height,b-2)
        if x1-x0>=22 and y1-y0>=14:
            rois.append({'x':x0,'y':y0,'w':x1-x0,'h':y1-y0})
        if len(rois)>=slot_limit:break
    return rois


def suggest_rois(frame, max_rois=15, columns=None):
    """Return (proposals, diagnostics). Empty list signals AUTO could not infer spaces.

    Robustness is limited to top-down views with visible, straight painted lines.
    Always allow manual correction/approval in a separate GUI.
    """
    if frame is None or frame.ndim!=3:
        raise ValueError('Input must be a BGR image.')
    h,w=frame.shape[:2]
    if min(h,w)<80:raise ValueError('Image too small for auto ROI suggestions.')
    if not (1<=max_rois<=100):raise ValueError('max_rois must be 1..100')
    if columns is not None and not (1<=columns<=4):raise ValueError('columns must be 1..4')
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    threshold=int(np.clip(np.percentile(gray,75)+19,145,202))
    _,bright=cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY)
    centers=_estimate_center_lines(bright,max_columns=min(3,columns or 3))
    if columns is not None:
        if columns==1:centers=[]
        elif len(centers)!=columns-1:
            # If no clear vertical separator, use equal width bands as initial proposals.
            centers=[round(w*i/columns) for i in range(1,columns)]
    bands=list(zip([0]+centers,centers+[w]))
    proposed_stripes=[_row_markings(bright,left,right) for left,right in bands]
    # Prefer the row with the most visible painted lines to infer hidden dividers.
    strongest=max(proposed_stripes,key=len) if proposed_stripes else []
    shared_dividers=_stable_grid(strongest,h)
    proposal=[]
    info=[]
    for col_idx,((left,right),stripe) in enumerate(zip(bands,proposed_stripes)):
        boxes=_make_bays(stripe,left,right,h,max_rois,global_dividers=shared_dividers)
        proposal.extend(boxes)
        info.append({'band':col_idx+1,'x':[left,right], 'dividers':len(stripe),'boxes':len(boxes)})
    # Propose at most max_rois slots, by y order per band (already sorted).
    proposal=proposal[:max_rois]
    proposals=[{'id':f'N{i+1:03d}',**r} for i,r in enumerate(proposal)]
    return proposals, {'threshold':threshold,'centerlines':centers,'bands':info,
                       'confidence':'PROPOSAL_ONLY_REVIEW_REQUIRED',
                       'note':'Top-down parking lines heuristic; may miss hidden lines/empty bays.'}


if __name__=='__main__':
    import argparse,json
    ap=argparse.ArgumentParser()
    ap.add_argument('image');ap.add_argument('--columns',type=int,choices=[1,2,3,4]);ap.add_argument('--max-rois',type=int,default=15)
    args=ap.parse_args()
    img=cv2.imread(args.image)
    slots,info=suggest_rois(img,args.max_rois,args.columns)
    print(json.dumps({'slots':slots,'diagnostic':info},indent=2))
