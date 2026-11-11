import unittest
import numpy as np
from touch_research.classification.data import (
    build_window_dataset, split_by_recording, split_by_official_recording,
    two_fold_recording_cv
)

class PipelineTests(unittest.TestCase):
    def metadata(self):
        frames=20; nrec=6; n=frames*nrec
        p=np.full((n,32,32),900,dtype=np.int16)
        rng=np.random.default_rng(9)
        flat=p.reshape(n,-1)
        flat[:,:548]+=rng.integers(0,20,size=(n,548))
        return {"pressure":p,"hasValidLabel":np.ones(n,bool),
          "recordingId":np.repeat(np.arange(nrec),frames),
          "objectId":np.repeat(np.repeat([1,2],3),frames),
          "splitId":np.repeat([0,0,1,0,0,1],frames),
          "isBalanced":np.ones(n,bool),
          "objects":np.array(["empty","object-a","object-b"],dtype=object)}
    def test_window_build_and_group_split(self):
        data,meta=build_window_dataset(self.metadata(),k_frames=5,stride=3)
        self.assertGreater(meta["n_windows"],0)
        split=split_by_recording(data,seed=3)
        sets={k:set(data["recording_id"][split[k]].tolist()) for k in ("train","val","test")}
        self.assertFalse(sets["train"]&sets["val"])
        self.assertFalse(sets["train"]&sets["test"])
        self.assertFalse(sets["val"]&sets["test"])
    def test_official_split_keeps_recordings_disjoint(self):
        data,_=build_window_dataset(self.metadata(),k_frames=5,stride=3)
        split=split_by_official_recording(data)
        train=set(data["recording_id"][split["train"]].tolist())
        test=set(data["recording_id"][split["test"]].tolist())
        self.assertFalse(train & test)
        self.assertEqual(len(train),4)
        self.assertEqual(len(test),2)
    def test_official_split_rejects_recording_mixed_between_splits(self):
        data,_=build_window_dataset(self.metadata(),k_frames=5,stride=3)
        rid=int(data["recording_id"][0])
        data["split_id"][np.flatnonzero(data["recording_id"]==rid)[0]]=1
        with self.assertRaises(ValueError):
            split_by_official_recording(data)

    def test_two_fold_cv_uses_only_official_train_recordings(self):
        data,_=build_window_dataset(self.metadata(),k_frames=5,stride=3)
        folds=two_fold_recording_cv(data)
        self.assertEqual(len(folds),2)
        official_test=set(data["recording_id"][data["split_id"]==1].tolist())
        for fold in folds:
            tr=set(data["recording_id"][fold["train"]].tolist())
            va=set(data["recording_id"][fold["validation"]].tolist())
            self.assertFalse(tr & va)
            self.assertFalse(tr & official_test)
            self.assertFalse(va & official_test)
            self.assertEqual(set(np.unique(data["object_id"][fold["train"]])),
                             set(np.unique(data["object_id"][fold["validation"]])))

if __name__ == "__main__": unittest.main()
