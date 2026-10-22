"""Additional compatibility tests for all-slot upgrade. No real dataset needed."""
import unittest
from pathlib import Path
import sys
import tempfile
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from main import read_slots
from detector import ParkingDetector
from scripts.run_all_slots import predict_batch, draw

class AllSlotsTests(unittest.TestCase):
    def test_candidate_roi_json_valid(self):
        cfg=read_slots(ROOT/'config/parking_9010418_all_slots_DRAFT.json')
        ids=[s['id'] for s in cfg['slots']]
        self.assertGreaterEqual(len(ids),200)
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual((cfg['frame_width'],cfg['frame_height']),(1280,720))
    def test_batch_matches_scalar_model_for_vertical_slots(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)
            for name,color in [('empty', (100,100,100)),('occupied',(12,15,230))]:
                directory=path/name;directory.mkdir()
                for n in range(4):
                    im=np.full((48,24,3),color,dtype=np.uint8)
                    cv2.imwrite(str(directory/f'{n:03d}.jpg'),im)
            model=ParkingDetector(path, k=3)
            image=np.full((100,120,3),(100,100,100),np.uint8)
            image[10:50,40:60]=(12,15,230)
            slots=[{'id':'A','x':5,'y':5,'w':20,'h':40}, {'id':'B','x':40,'y':10,'w':20,'h':40}]
            self.assertEqual([p['status'] for p in predict_batch(model,image,slots)],
                             [p['status'] for p in model.predict(image,slots)])
            out=draw(image,predict_batch(model,image,slots),4)
            self.assertEqual(out.shape,image.shape)

if __name__=='__main__':unittest.main()
