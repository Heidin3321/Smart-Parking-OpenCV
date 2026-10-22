"""Pure logic checks; run via python -m unittest discover -s tests -v"""
import sys
import unittest
import tempfile
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from test_new_parking_video import label_plan, sample_indices, evaluate, assert_valid_roi, SceneReference, feature, save_json
import numpy as np


class NewParkingVideoTests(unittest.TestCase):
    def test_time_disjoint_split(self):
        x=label_plan(283,2,2)
        self.assertEqual(len(x['calibration']),2)
        self.assertEqual(len(x['test']),2)
        self.assertLess(max(x['calibration']),min(x['test']))

    def test_strict_uncertain_counts_wrong(self):
        score=evaluate({'one':'empty','two':'occupied','three':'unknown'},
                       {'one':'empty','two':'uncertain','three':'occupied'})
        self.assertEqual(score['correct'],1)
        self.assertEqual(score['tested'],2)
        self.assertEqual(score['uncertain'],1)
        self.assertAlmostEqual(score['accuracy'],.5)
        self.assertAlmostEqual(score['coverage'],.5)

    def test_unique_roi_coordinates(self):
        assert_valid_roi({'frame_width':100,'frame_height':80,'slots':
                        [{'id':'N001','x':2,'y':2,'w':20,'h':30}]})
        with self.assertRaises(ValueError):
            assert_valid_roi({'frame_width':100,'frame_height':80,'slots':
                        [{'id':'N001','x':96,'y':2,'w':20,'h':30}]})

    def test_windows_safe_json_saving(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'human_truth.json'
            save_json(p, {'label':'occupied'})
            save_json(p, {'label':'empty'})
            self.assertEqual(json.loads(p.read_text())['label'],'empty')
            self.assertFalse(p.with_suffix('.json.part').exists())

    def test_reference_exact_patch(self):
        occupied=np.full((40,50,3),180,np.uint8)
        detector=SceneReference({'N001':[('occupied',feature(occupied))]})
        result=detector.predict(occupied,[{'id':'N001','x':0,'y':0,'w':50,'h':40}])
        self.assertEqual(result[0]['status'],'occupied')

    def test_reference_rejects_unfamiliar_patch(self):
        occupied=np.full((40,50,3),180,np.uint8)
        empty=np.zeros_like(occupied)
        detector=SceneReference({'N001':[('occupied',feature(occupied))]})
        result=detector.predict(empty,[{'id':'N001','x':0,'y':0,'w':50,'h':40}])
        self.assertEqual(result[0]['status'],'uncertain')

if __name__=='__main__':unittest.main()
