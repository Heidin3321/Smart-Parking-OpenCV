"""Tests for selectable ROI creation. Run: python -m unittest tests/test_parking_roi_choice.py -v"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from auto_parking_rois import suggest_rois
import parking_roi_wizard as wizard


class AutoRoiTests(unittest.TestCase):
    def test_synthetic_two_column_bays(self):
        im=np.full((450,380,3),110,dtype=np.uint8)
        cv2.line(im,(200,0),(200,449),(250,250,250),4)
        for y in (52,110,169,228,287,346,405):
            cv2.line(im,(51,y),(199,y),(255,255,255),3)
            cv2.line(im,(204,y),(340,y),(255,255,255),3)
        for row in range(7):
            for x in (68,235):
                if row==2 and x==68:continue  # empty stall
                cy=12+row*59
                cv2.rectangle(im,(x,cy),(x+90,cy+31),(40,40,40),-1)
        slots,diag=suggest_rois(im,max_rois=15)
        self.assertGreaterEqual(len(slots),9)
        self.assertLessEqual(len(slots),15)
        self.assertEqual(len({s['id'] for s in slots}),len(slots))
        self.assertTrue(all(0<=s['x'] and s['x']+s['w']<=380 for s in slots))
        self.assertEqual(diag['confidence'],'PROPOSAL_ONLY_REVIEW_REQUIRED')

    def test_low_information_does_not_invent_grid(self):
        image=np.full((240,400,3),125,dtype=np.uint8)
        proposals,info=suggest_rois(image)
        self.assertEqual(proposals,[])

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):suggest_rois(np.zeros((100,100,3),dtype=np.uint8),max_rois=0)
        with self.assertRaises(ValueError):suggest_rois(np.zeros((100,100,3),dtype=np.uint8),columns=5)

    def test_reindex_after_manually_deleting_one(self):
        before=[{'id':'N002','x':2,'y':3,'w':44,'h':23}]
        self.assertEqual(wizard.numbered(before)[0]['id'],'N001')


if __name__=='__main__':unittest.main()
