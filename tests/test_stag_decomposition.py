import unittest
import numpy as np
from touch_research.decomposition.stag import (
    GraspEvent, event_window, normalize_pressure, official_grasp_events,
    smoothed_mean_pressure, decompose_event,
)

class StagDecompositionTests(unittest.TestCase):
    def test_normalization_and_sensor_selection(self):
        p=np.array([[[500,650],[700,510]],[[575,650],[500,510]]],dtype=np.int16)
        mask=np.array([[True,True],[True,False]])
        x=normalize_pressure(p,mask)
        self.assertEqual(x.shape,(2,3))
        self.assertTrue(np.allclose(x[0],[0,1,1]))
        self.assertTrue(np.allclose(x[1],[0.5,1,0]))

    def test_official_event_offset(self):
        events=official_grasp_events(7,np.array([0,1,0,1]),np.array([0,4,0,3]),np.array([0,1,0,0]))
        self.assertEqual(len(events),2)
        self.assertEqual(events[0].peak_index,5)
        self.assertTrue(events[0].reliable)
        self.assertFalse(events[1].reliable)

    def test_event_relative_window_sensitivity(self):
        y=np.array([0,0,0.2,0.4,0.6,1.0,0.7,0.4,0.1,0.0])
        self.assertEqual(event_window(y,0,5,threshold=0.5,normalization="event"),(4,6))

    def test_recording_relative_is_default(self):
        y=np.array([0.0,0.3,0.4,0.45,0.3,1.0,0.0])
        # The event peak is locally high but remains below 0.5 of the
        # recording-wide dynamic range; the paper-aligned default rejects it.
        self.assertIsNone(event_window(y,1,3,threshold=0.5))
        self.assertEqual(event_window(y,1,3,threshold=0.5,normalization="event"),(2,3))

    def test_decomposition_subtracts_baseline(self):
        x=np.array([[0.,1.],[0.2,0.8],[0.5,0.7],[1.,0.5],[0.7,0.3],[0.2,0.1]],dtype=np.float32)
        y=np.array([0,0.2,0.5,1,0.7,0.2])
        e=GraspEvent(1,0,0,3,3,True)
        d=decompose_event(x,y,e,radius=3,threshold=0.5)
        self.assertIsNotNone(d)
        self.assertTrue(np.allclose(d.baseline,[0,1]))
        self.assertTrue(np.allclose(d.object_pressure,x[d.window_start:d.window_end+1]-x[0]))

    def test_non_reliable_event_is_not_decomposed(self):
        x=np.arange(12,dtype=np.float32).reshape(6,2)
        e=GraspEvent(1,0,0,3,3,False)
        self.assertIsNone(decompose_event(x,np.arange(6),e))

if __name__ == "__main__":
    unittest.main()
