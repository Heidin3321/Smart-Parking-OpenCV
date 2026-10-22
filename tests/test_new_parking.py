"""Headless regression checks for NEW parking photo test.
Run with: python -m unittest discover -s tests -v
"""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

SCRIPT=Path(__file__).resolve().parents[1]/'scripts'/'test_new_parking.py'
SPEC=importlib.util.spec_from_file_location('test_new_parking',SCRIPT)
TESTER=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TESTER)


class NewParkingTestingTests(unittest.TestCase):
    def test_metrics_include_both_classes(self):
        truth={'T001':'occupied','T002':'empty','T003':'occupied','T004':'unknown'}
        preds={'T001':'occupied','T002':'occupied','T003':'empty','T004':'empty'}
        result=TESTER.scores(truth,preds)
        self.assertEqual(result['correct'],1)
        self.assertEqual(result['n_evaluable'],3)
        self.assertEqual(result['n_unknown_ground_truth'],1)
        self.assertAlmostEqual(result['accuracy'],1/3,places=4)
        self.assertEqual(result['confusion_matrix']['occupied']['empty'],1)
        self.assertEqual(result['confusion_matrix']['empty']['occupied'],1)

    def test_absent_class_metrics_not_claimed(self):
        result=TESTER.scores({'T001':'occupied'},{'T001':'occupied'})
        self.assertIsNone(result['per_class']['empty']['recall'])
        self.assertIsNone(result['per_class']['empty']['precision'])
        self.assertEqual(result['accuracy'],1.)

    def test_uncertain_does_not_inflate_score(self):
        result=TESTER.scores({'T001':'occupied','T002':'empty'},
                              {'T001':'uncertain','T002':'empty'})
        self.assertEqual(result['correct'],1)
        self.assertEqual(result['wrong'],1)
        self.assertEqual(result['prediction_coverage'],0.5)

    def test_ground_truth_fingerprint_protection(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            img=root/'new-parking.jpg'
            TESTER.save_image(img,np.zeros((100,180,3),np.uint8))
            with mock.patch.object(TESTER,'ROOT',root):
                paths=TESTER.paths_for(img)
                cfg={'frame_width':180,'frame_height':100,'image_sha256':paths['hash'],
                     'slots':[{'id':'T001','x':1,'y':1,'w':30,'h':35}]}
                TESTER.save_json(paths['config'],cfg)
                TESTER.put_truth(paths,cfg,{'T001':'occupied'})
                self.assertEqual(TESTER.load_truth(paths,cfg)['T001'],'occupied')
                cfg['slots'][0]['x']+=1
                with self.assertRaisesRegex(ValueError,'different image/ROI'):TESTER.load_truth(paths,cfg)

    def test_end_to_end_new_image_evaluation_with_existing_training(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            image=root/'another_location.png'
            base=np.full((220,360,3),(80,80,80),np.uint8)
            base[18:98,20:150]=(40,40,170)
            base[110:202,185:330]=(70,170,70)
            TESTER.save_image(image,base)
            with mock.patch.object(TESTER,'ROOT',root):
                paths=TESTER.paths_for(image)
                cfg={'frame_width':360,'frame_height':220,'source_image':image.name,
                     'image_sha256':paths['hash'],
                     'slots':[{'id':'T001','x':20,'y':18,'w':130,'h':80},
                              {'id':'T002','x':185,'y':110,'w':145,'h':92}]}
                TESTER.save_json(paths['config'],cfg)
                TESTER.put_truth(paths,cfg,{'T001':'occupied','T002':'empty'})
                # Fixed training directory created before inference; test image is not part of it.
                training=root/'my_preexisting_training'
                for category,color in [('empty',(70,170,70)),('occupied',(40,40,170))]:
                    (training/category).mkdir(parents=True)
                    for i in range(3):
                        patch=np.full((80,130,3),color,np.uint8)
                        TESTER.save_image(training/category/f'existing_{i}.png',patch)
                with mock.patch.dict(TESTER.TRAINING, {'cnrpark':str(training)}):
                    report=TESTER.test_image(image,model='cnrpark')
                self.assertEqual(report['models']['cnrpark']['n_evaluable'],2)
                self.assertEqual(report['models']['cnrpark']['correct'],2)
                self.assertTrue((paths['output']/'annotated_cnrpark.jpg').exists())
                self.assertTrue((paths['output']/'report.html').exists())
                self.assertTrue((paths['output']/'predictions_cnrpark.csv').exists())
                self.assertEqual(len(TESTER.report_all()),1)


if __name__=='__main__':unittest.main()
