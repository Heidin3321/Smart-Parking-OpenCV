"""Validates creation of a real annotated MP4 independently of manual labels."""
import sys
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from test_new_parking_video import render_one_video,verify_mp4,annotate


class FakeModel:
    def predict(self, frame, slots):
        return [{**s,'status':'occupied' if i%2==0 else 'empty'}
                for i,s in enumerate(slots)]


class TestAnnotatedVideo(unittest.TestCase):
    def test_annotate_has_header_and_visible_roi_pixels(self):
        frame=np.zeros((100,150,3),np.uint8)
        marked=annotate(frame,FakeModel().predict(frame,[{'id':'N001','x':10,'y':10,'w':60,'h':50}]),0,'test')
        self.assertEqual(marked.shape,(190,150,3))
        self.assertTrue(np.any(marked[100,10]!=0))

    def test_render_mp4_without_truth_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            vid=base/'fresh_new_lot.mp4'
            w=cv2.VideoWriter(str(vid),cv2.VideoWriter_fourcc(*'mp4v'),10,(160,120))
            self.assertTrue(w.isOpened())
            for i in range(13):
                frame=np.zeros((120,160,3),np.uint8)
                cv2.rectangle(frame,(15+i%2,20),(55,85),(225,225,225),-1)
                w.write(frame)
            w.release()
            cap=cv2.VideoCapture(str(vid))
            self.assertTrue(cap.isOpened())
            cfg={'frame_width':160,'frame_height':120,'slots':[
                 {'id':'N001','x':10,'y':10,'w':50,'h':80},
                 {'id':'N002','x':80,'y':10,'w':50,'h':80}]}
            meta={'fps':10.,'count':13,'width':160,'height':120}
            paths={'out':base/'results'}
            try:
                output=render_one_video(cap,meta,cfg,paths,'test',FakeModel())
            finally:
                cap.release()
            self.assertEqual(verify_mp4(output),output)
            self.assertTrue((base/'results'/'preview_annotated_video.jpg').is_file())
            self.assertTrue((base/'results'/'video_predictions.csv').is_file())
            decoded=cv2.VideoCapture(str(output))
            self.assertEqual(int(decoded.get(cv2.CAP_PROP_FRAME_COUNT)),13)
            ok,fr=decoded.read();decoded.release()
            self.assertTrue(ok)
            self.assertEqual(fr.shape[:2],(210,160))

if __name__=='__main__':unittest.main()
