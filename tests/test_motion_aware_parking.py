"""Checks that detected camera motion results in moving ROIs, not fixed boxes."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_motion_aware_parking import CameraMotion,transform_slot,predict_with_motion


class MotionAwareROITests(unittest.TestCase):
    @staticmethod
    def sample_scene():
        rng=np.random.default_rng(12345)
        img=np.zeros((280,400,3),np.uint8)
        img[:]=rng.integers(25,65,size=(280,400,3),dtype=np.uint8)
        for i in range(8):
            x=20+i*45
            cv2.rectangle(img,(x,30),(x+32,100),(50+i*20,160,200-i*19),-1)
            cv2.rectangle(img,(x+2,120),(x+30,180),(150+i*10,90,90),-1)
            cv2.line(img,(x,10),(x,250),(240,240,240),2)
        return img

    def test_translation_follows_moving_camera(self):
        ref=self.sample_scene()
        for dx,dy in ((15,9),(-20,11),(22,-9)):
            frame=cv2.warpAffine(ref,np.float32([[1,0,dx],[0,1,dy]]),(400,280),borderMode=cv2.BORDER_REFLECT_101)
            mat,quality,_,_=CameraMotion(ref).estimate(frame)
            self.assertNotEqual(quality,'LOST')
            self.assertAlmostEqual(mat[0,2],dx,delta=3.5)
            self.assertAlmostEqual(mat[1,2],dy,delta=3.5)

    def test_roi_coordinates_follow_motion(self):
        box={'id':'N001','x':80,'y':100,'w':54,'h':75}
        m=np.array([[1,0,18],[0,1,12]],float)
        moved,valid=transform_slot(box,m,400,280)
        self.assertTrue(valid)
        self.assertEqual((moved['x'],moved['y']), (98,112))

    def test_no_confident_label_when_alignment_lost(self):
        class ExplodingDetector:
            def predict(self,*_):
                raise AssertionError('Detector must not be called with lost tracking')
        f=self.sample_scene()
        slots=[{'id':'N001','x':80,'y':100,'w':54,'h':75}]
        prediction=predict_with_motion(ExplodingDetector(),f,slots,np.array([[1.,0.,0.],[0.,1.,0.]]),False)
        self.assertEqual(prediction[0]['status'],'uncertain')

    def test_partial_out_of_frame_marks_uncertain(self):
        class ExplodingDetector:
            def predict(self,*_):
                raise AssertionError('Detector must not be called outside image')
        f=self.sample_scene()
        slots=[{'id':'N001','x':375,'y':100,'w':23,'h':75}]
        matrix=np.array([[1.,0.,30.],[0.,1.,0.]])
        prediction=predict_with_motion(ExplodingDetector(),f,slots,matrix,True)
        self.assertEqual(prediction[0]['status'],'uncertain')


if __name__=='__main__': unittest.main()
