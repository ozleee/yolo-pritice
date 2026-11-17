import unittest
import cv2
import numpy as np
from target_tracker import TargetTracker


class TrackerTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(4)
        self.patch=rng.integers(0,255,(60,70,3),dtype=np.uint8)
        self.tracker=TargetTracker()

    def image(self,x=40,y=50):
        frame=np.zeros((240,320,3),np.uint8)
        frame[y:y+60,x:x+70]=self.patch
        return frame

    def test_motion_and_recovery_without_detection(self):
        t=self.tracker
        self.assertTrue(t.start(self.image(),dict(box=[40,50,110,110],name='target')))
        for i in range(1,5):
            b=t.update(self.image(40+2*i,50+i))
            self.assertIsNotNone(b)
            self.assertAlmostEqual(b[0],40+2*i,delta=2)
        self.assertIsNone(t.update(np.zeros((240,320,3),np.uint8)))
        self.assertEqual(t.state,'LOST')
        self.assertIsNone(t.recover(np.zeros((240,320,3),np.uint8),[]))
        self.assertIsNone(t.recover(self.image(180,130),[]))
        b=t.recover(self.image(180,130),[])
        self.assertIsNotNone(b)
        np.testing.assert_allclose(b,[180,130,250,190],atol=2)
        self.assertEqual(t.state,'TRACK')

    def test_ambiguous_duplicates_not_recovered(self):
        t=self.tracker;t.start(self.image(),dict(box=[40,50,110,110],name='target'))
        t.state='LOST';frame=self.image(10,30)
        frame[140:200,200:270]=self.patch
        self.assertIsNone(t.recover(frame,[]))
        self.assertIsNone(t.recover(frame,[]))

    def test_reset(self):
        t=self.tracker;t.start(self.image(),dict(box=[40,50,110,110],name='target'))
        t.reset();self.assertFalse(t.active);self.assertIsNone(t.update(self.image()))


if __name__=='__main__':unittest.main()
