import unittest
import numpy as np
from detection_lock import DetectionLock

class LockTests(unittest.TestCase):
    def setUp(self):
        self.frame=np.zeros((200,300,3),np.uint8)
        self.frame[40:100,40:100]=(40,100,200)
        self.d=dict(id=1,name='cup',box=[40,40,100,100])
        self.lock=DetectionLock();self.assertTrue(self.lock.start(self.frame,self.d))

    def test_same_id_and_new_id_recovery(self):
        d,recovered=self.lock.update(self.frame,[self.d]);self.assertEqual(d['id'],1);self.assertFalse(recovered)
        self.assertIsNone(self.lock.update(self.frame,[])[0])
        new=dict(self.d,id=99)
        for _ in range(2):self.assertIsNone(self.lock.update(self.frame,[new])[0])
        d,recovered=self.lock.update(self.frame,[new])
        self.assertEqual(d['id'],99);self.assertTrue(recovered)

    def test_reject_wrong_class_or_appearance(self):
        self.lock.update(self.frame,[])
        for _ in range(4):self.assertIsNone(self.lock.update(self.frame,[dict(self.d,id=9,name='mouse')])[0])
        black=np.zeros_like(self.frame)
        for _ in range(4):self.assertIsNone(self.lock.update(black,[dict(self.d,id=9)])[0])

    def test_ambiguity_and_unlock(self):
        self.lock.update(self.frame,[])
        for _ in range(4):
            self.assertIsNone(self.lock.update(self.frame,[dict(self.d,id=8),dict(self.d,id=9)])[0])
        self.lock.reset();self.assertFalse(self.lock.active)

    def test_recent_appearance_replaces_initial_and_freezes_during_loss(self):
        changed=self.frame.copy()
        changed[40:100,40:100]=(200,40,40)
        initial=self.lock.reference.copy()
        for _ in range(7):self.lock.update(changed,[dict(self.d,confidence=.9)])
        self.assertFalse(np.allclose(initial,self.lock.reference))
        recent=self.lock.reference.copy()
        self.lock.update(changed,[])
        # Original appearance must no longer be used to recover.
        for _ in range(3):
            self.assertIsNone(self.lock.update(self.frame,[dict(self.d,id=9)])[0])
            np.testing.assert_array_equal(recent,self.lock.reference)
        for _ in range(2):self.assertIsNone(self.lock.update(changed,[dict(self.d,id=9)])[0])
        d,recovered=self.lock.update(changed,[dict(self.d,id=9)])
        self.assertTrue(recovered);self.assertEqual(d['id'],9)

    def test_low_confidence_does_not_update_reference(self):
        reference=self.lock.reference.copy()
        changed=np.zeros_like(self.frame)
        for _ in range(8):self.lock.update(changed,[dict(self.d,confidence=.2)])
        np.testing.assert_array_equal(reference,self.lock.reference)

if __name__=='__main__':unittest.main()
