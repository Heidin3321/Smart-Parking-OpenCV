"""New V3 checks. Run alongside previous integration tests."""
from pathlib import Path
import sys
import unittest
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from main import read_slots
from scripts.run_parking_v3 import annotate

class ParkingV3Tests(unittest.TestCase):
    def test_config_contains_14_valid_unique_rois(self):
        c=read_slots(ROOT/'config/parking_v3_slots_DRAFT.json')
        self.assertEqual((c['frame_width'],c['frame_height']),(380,452))
        self.assertEqual(len(c['slots']),14)
        self.assertEqual(len({s['id'] for s in c['slots']}),14)

    def test_stabilized_video_matches_roi(self):
        cap=cv2.VideoCapture(str(ROOT/'dataset/parking_v3_stabilized.mp4'))
        try:
            self.assertTrue(cap.isOpened())
            ok,frame=cap.read();self.assertTrue(ok)
            self.assertEqual(frame.shape[:2],(452,380))
        finally:cap.release()

    def test_header_is_outside_frame(self):
        black=np.zeros((452,380,3),np.uint8)
        detections=[dict(id='R01',x=210,y=2,w=140,h=50,status='occupied')]
        view=annotate(black,detections,0)
        self.assertEqual(view.shape,(1009,760,3))
        self.assertTrue(view[105+400:105+410,300:310].max()==0)

if __name__=='__main__':unittest.main()
