"""Unit tests for ROI menu and mandatory blind labeling (no video/GUI required)."""
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import parking_roi_wizard as wizard
import video_label_flow as flow
import test_new_parking_video as base
import run_motion_aware_parking as motion


class FakeSampler:
    def at(self, frame_index):
        fr=np.full((120,200,3),110,dtype=np.uint8)
        slots={'N001':({'id':'N001','x':20,'y':20,'w':50,'h':40},True),
               'N002':({'id':'N002','x':90,'y':20,'w':50,'h':40},True)}
        return fr,slots,'ANCHOR',None


class RoiMenuTests(unittest.TestCase):
    def test_auto_is_menu_option_2_no_saved(self):
        with mock.patch('builtins.input',return_value='2'):
            self.assertEqual(wizard.choose_mode(False),'auto')

    def test_existing_is_option_3_when_roi_saved(self):
        with mock.patch('builtins.input',return_value='3'):
            self.assertEqual(wizard.choose_mode(True),'existing')

    def test_auto_proposal_has_slot_ids(self):
        slots=wizard.numbered([{'x':4,'y':10,'w':20,'h':30}])
        self.assertEqual(slots[0]['id'],'N001')

    def test_motion_moves_roi(self):
        mat=np.array([[1,0,9],[0,1,3]],dtype=np.float64)
        result,valid=motion.transform_slot({'id':'N001','x':20,'y':30,'w':25,'h':15},mat,250,150)
        self.assertTrue(valid)
        self.assertEqual((result['x'],result['y']),(29,33))

    def test_training_and_testing_frames_do_not_overlap(self):
        plan=base.label_plan(300,2,2)
        self.assertLess(max(plan['calibration']),min(plan['test']))


class LabelTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg={'frame_width':200,'frame_height':120,
                  'sampling_plan':{'calibration':[10],'test':[100]},
                  'slots':[{'id':sid,'x':20+idx*70,'y':20,'w':50,'h':40}
                           for idx,sid in enumerate(['N001','N002'])]}
        self.paths={'fingerprint':'fake-video-id', 'labels':Path(self.tmp.name)/'labels.json'}

    def gui_mock(self,keys):
        # fake OpenCV only in tests; production still opens real GUI.
        return mock.patch.multiple(cv2,
                 namedWindow=mock.DEFAULT,resizeWindow=mock.DEFAULT,
                 imshow=mock.DEFAULT,destroyWindow=mock.DEFAULT,
                 waitKey=mock.Mock(side_effect=keys))

    def test_all_slots_both_phases_must_be_labeled(self):
        with self.gui_mock([ord('o'),ord('e'),ord('e'),ord('o')]):
            obj=flow.label_blind(FakeSampler(),self.cfg,self.paths)
        self.assertEqual(len(obj['labels']),4)
        self.assertEqual(obj['labels']['calibration:10:N001'],'occupied')
        self.assertEqual(obj['labels']['test:100:N002'],'occupied')
        # Raw labels are saved without external .part rename.
        self.assertTrue(self.paths['labels'].exists())

    def test_resume_when_labeling_paused(self):
        with self.gui_mock([ord('o'),ord('q')]):
            with self.assertRaisesRegex(RuntimeError,'BELUM SELESAI'):
                flow.label_blind(FakeSampler(),self.cfg,self.paths)
        with self.gui_mock([ord('e'),ord('e'),ord('o')]):
            obj=flow.label_blind(FakeSampler(),self.cfg,self.paths)
        self.assertEqual(len(obj['labels']),4)


if __name__=='__main__':unittest.main()
