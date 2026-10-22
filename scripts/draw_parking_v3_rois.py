"""Draw Parking V3 ROI rectangles by hand from a blank config.

A source video and blank JSON are required. This does NOT classify parking occupancy.
Controls: left-drag add, right-click remove, U undo, C+C clear all,
S save, Q save+quit, ESC quit without saving, I labels.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    for k in ("frame_width", "frame_height", "slots"):
        if k not in cfg:
            raise ValueError(f"Missing '{k}' in ROI configuration: {path}")
    return cfg


def next_roi_id(slots: list[dict]) -> str:
    used = {str(s["id"]) for s in slots}
    counter = 1
    while f"P{counter:03}" in used:
        counter += 1
    return f"P{counter:03}"


def write_config(path: Path, config: dict, slots: list[dict]) -> None:
    out = copy.deepcopy(config)
    out["slots"] = copy.deepcopy(slots)
    out["roi_status"] = "MANUAL ROI DRAFT - verify positions on 0s, 10s and 20s"
    out["notes"] = "ROI positions drawn by user for stabilized V3 video; classification requires separate evaluation."
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"SAVED {len(slots)} manual ROI -> {path}")


def edit_video(args: argparse.Namespace) -> None:
    empty_path = resolve_path(args.config)
    save_path = resolve_path(args.save_as)
    # Avoid accidentally erasing previously hand-drawn positions on rerun.
    if save_path.is_file() and not args.reset:
        initial_path = save_path
        print(f"RESUMING previously saved manual ROI: {initial_path}")
    else:
        initial_path = empty_path
        if args.reset:
            print(f"STARTING FROM BLANK config (--reset): {initial_path}")
        else:
            print(f"STARTING FROM BLANK config: {initial_path}")
    config = load_config(initial_path)
    slots = copy.deepcopy(config["slots"])

    cap = cv2.VideoCapture(str(resolve_path(args.source)))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {resolve_path(args.source)}")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.time * 1000)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError("Cannot read the frame; choose a time within video length")
    h, w = frame.shape[:2]
    if (w, h) != (int(config["frame_width"]), int(config["frame_height"])):
        raise ValueError(f"Config needs {w}x{h} video frame but has {config['frame_width']}x{config['frame_height']}")

    scale = args.scale
    top_pad = 68
    history: list[list[dict]] = []
    drag_start: tuple[int, int] | None = None
    pointer: tuple[int, int] | None = None
    show_ids = True
    clear_armed = False
    dirty = False

    def checkpoint() -> None:
        history.append(copy.deepcopy(slots))
        if len(history) > 100:
            history.pop(0)

    def to_xy(px: int, py: int) -> tuple[int, int] | None:
        if py < top_pad:
            return None
        x = max(0, min(w-1, px//scale))
        y = max(0, min(h-1, (py-top_pad)//scale))
        return x, y

    def find_slot(x: int, y: int) -> int | None:
        for idx in reversed(range(len(slots))):
            s=slots[idx]
            if s['x'] <= x < s['x']+s['w'] and s['y'] <= y < s['y']+s['h']:
                return idx
        return None

    def mouse(event: int, px: int, py: int, flags: int, userdata: object) -> None:
        nonlocal drag_start, pointer, dirty, clear_armed
        xy = to_xy(px, py)
        if event == cv2.EVENT_RBUTTONDOWN and xy is not None:
            idx = find_slot(*xy)
            if idx is not None:
                checkpoint()
                removed = slots.pop(idx)
                dirty = True
                print("DELETED", removed['id'])
        elif event == cv2.EVENT_LBUTTONDOWN and xy is not None:
            drag_start = xy
            pointer = xy
        elif event == cv2.EVENT_MOUSEMOVE and drag_start is not None:
            pointer = xy
        elif event == cv2.EVENT_LBUTTONUP and drag_start is not None:
            start=drag_start
            drag_start=None
            pointer=None
            if xy is not None:
                x0, y0 = start; x1,y1=xy
                bw,bh=abs(x1-x0),abs(y1-y0)
                if bw >= 8 and bh >= 8:
                    checkpoint()
                    entry=dict(id=next_roi_id(slots), x=min(x0,x1), y=min(y0,y1), w=bw, h=bh)
                    slots.append(entry)
                    dirty=True
                    clear_armed=False
                    print(f"ADDED {entry['id']} ({entry['x']}, {entry['y']}, {entry['w']}, {entry['h']}); total={len(slots)}")

    win="SMART PARKING V3 - DRAW ROI FROM SCRATCH"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, mouse)
    print("ADD: left-click and DRAG around each parking bay | DELETE: right-click inside ROI")
    print("S=SAVE | Q=SAVE & QUIT | ESC=EXIT WITHOUT SAVING | U=UNDO | C twice=CLEAR ALL | I=ID toggle")

    while True:
        resized=cv2.resize(frame,(w*scale,h*scale),interpolation=cv2.INTER_LINEAR)
        canvas=np.zeros((resized.shape[0]+top_pad, resized.shape[1],3),dtype=np.uint8)
        canvas[:top_pad]=(36,40,47)
        canvas[top_pad:]=resized
        n=len(slots)
        cv2.putText(canvas, f"MANUAL ROI: {n} | LEFT DRAG = ADD | RIGHT CLICK = DELETE",(8,23),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(canvas,"S SAVE | Q SAVE+QUIT | ESC DISCARD | U UNDO | C+C CLEAR",(8,47),cv2.FONT_HERSHEY_SIMPLEX,.43,(90,230,230),1,cv2.LINE_AA)
        for s in slots:
            x,y,bw,bh=(int(s[k]) for k in ('x','y','w','h'))
            cv2.rectangle(canvas,(x*scale,y*scale+top_pad),((x+bw)*scale,(y+bh)*scale+top_pad),(20,220,255),2)
            if show_ids:
                cv2.putText(canvas,s['id'],(x*scale+3,y*scale+top_pad+16),cv2.FONT_HERSHEY_SIMPLEX,.46,(255,255,255),1,cv2.LINE_AA)
        if drag_start and pointer:
            x0,y0=drag_start; x1,y1=pointer
            cv2.rectangle(canvas,(x0*scale,y0*scale+top_pad),(x1*scale,y1*scale+top_pad),(60,255,90),2)
        if clear_armed:
            cv2.putText(canvas,"PRESS C AGAIN TO CLEAR ALL",(8,65),cv2.FONT_HERSHEY_SIMPLEX,.45,(40,70,255),1,cv2.LINE_AA)
        cv2.imshow(win,canvas)
        key=cv2.waitKey(30)&0xff
        if key==255:
            continue
        if key in (ord('u'),ord('U')):
            if history:
                slots[:]=history.pop();dirty=True;print("UNDO; now",len(slots),"ROI")
            clear_armed=False
        elif key in (ord('i'),ord('I')):
            show_ids=not show_ids
            clear_armed=False
        elif key in (ord('c'),ord('C')):
            if not clear_armed:
                clear_armed=True
                print("CONFIRM: press C again to remove all current ROI")
            else:
                checkpoint();slots.clear();dirty=True;clear_armed=False
                print("ALL ROIs REMOVED; start drawing")
        elif key in (ord('s'),ord('S')):
            write_config(save_path,config,slots)
            dirty=False;clear_armed=False
        elif key in (ord('q'),ord('Q')):
            write_config(save_path,config,slots)
            print("Saved and quit. Next: run detection with --config", save_path)
            break
        elif key==27:
            print("Quit without saving current changes. Already-saved file was not deleted.")
            break
        else:
            clear_armed=False

    cv2.destroyAllWindows()


def main() -> None:
    parser=argparse.ArgumentParser(description="Draw manual parking ROIs from a blank Parking V3 template")
    parser.add_argument('--source',default='dataset/parking_v3_stabilized.mp4')
    parser.add_argument('--config',default='config/parking_v3_slots_EMPTY.json')
    parser.add_argument('--save-as',default='config/parking_v3_slots_REVIEWED.json')
    parser.add_argument('--time',type=float,default=10.0)
    parser.add_argument('--scale',type=int,default=2,choices=[1,2,3])
    parser.add_argument('--reset',action='store_true',help="Start blank even when REVIEWED.json already exists")
    args=parser.parse_args()
    edit_video(args)


if __name__ == '__main__':
    main()
