"""Small deterministic tests for the scene-specific parking reference matcher."""
from pathlib import Path
import importlib.util
import unittest
import numpy as np

MODULE=Path(__file__).resolve().parents[1]/'scripts'/'run_v3_reference.py'
SPEC=importlib.util.spec_from_file_location('run_v3_reference_tested',MODULE)
m=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)

class ReferenceMatcherTests(unittest.TestCase):
    def test_identical_occupied_crop_matches(self):
        car=np.full((32,64,3),80,np.uint8)
        car[7:25,10:54]=180
        model=m.SlotReferenceDetector({'P1':[(20,'occupied',m.roi_feature(car))]})
        label,ref,d,reason=model.classify('P1',car.copy())
        self.assertEqual(label,'occupied')
        self.assertEqual(ref,20)
        self.assertLess(d,.01)

    def test_big_change_is_uncertain_not_empty(self):
        car=np.full((32,64,3),80,np.uint8)
        model=m.SlotReferenceDetector({'P1':[(20,'occupied',m.roi_feature(car))]})
        totally_changed=np.full((32,64,3),245,np.uint8)
        label,_,_,_=model.classify('P1',totally_changed)
        self.assertEqual(label,'uncertain')

    def test_missing_reference_is_uncertain(self):
        model=m.SlotReferenceDetector({})
        self.assertEqual(model.classify('P2',np.zeros((20,20,3),np.uint8))[0],'uncertain')

    def test_equal_feature_distance_zero(self):
        x=np.zeros((4,6,3),np.float32)
        self.assertEqual(m.pixel_difference(x,x),0.)

    def test_class_assignment_per_slot(self):
        empty=np.zeros((24,48,3),np.uint8)
        occupied=np.full((24,48,3),200,np.uint8)
        model=m.SlotReferenceDetector({'A':[(10,'occupied',m.roi_feature(occupied))],
                                       'B':[(10,'empty',m.roi_feature(empty))]})
        self.assertEqual(model.classify('A',occupied)[0],'occupied')
        self.assertEqual(model.classify('B',empty)[0],'empty')

if __name__=='__main__':unittest.main()
