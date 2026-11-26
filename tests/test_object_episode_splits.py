import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from build_object_episode_splits import make_object_split,enumerate_episodes,validate_manifests

class ObjectEpisodeSplitTests(unittest.TestCase):
    def setUp(self):
        self.split,_=make_object_split(list(range(1,27)),seed=20261118)
        self.recordings=[];rid=0
        for oid in range(1,27):
            for rep in range(3):
                self.recordings.append({"recording_id":rid,"object_id":oid,"object_name":f"object_{oid}",
                  "recording_name":f"{rid:03d}-object_{oid}","n_future_valid_frames":100+rep})
                rid+=1
    def test_split_is_deterministic_and_18_4_4(self):
        other,_=make_object_split(list(range(1,27)),seed=20261118)
        self.assertEqual(self.split,other)
        self.assertEqual([len(self.split[k]) for k in ("meta_train","meta_validation","meta_test")],[18,4,4])
    def test_all_objects_partitioned_once(self):
        flat=sum(self.split.values(),[])
        self.assertEqual(len(flat),26);self.assertEqual(len(set(flat)),26)
    def test_three_recordings_make_six_episodes_per_object(self):
        eps=enumerate_episodes(self.recordings,self.split,"meta_test")
        self.assertEqual(len(eps),24)
        self.assertEqual(sum(e["support_shots"]==1 for e in eps),12)
        self.assertEqual(sum(e["support_shots"]==2 for e in eps),12)
    def test_support_query_are_disjoint(self):
        for e in enumerate_episodes(self.recordings,self.split,"meta_validation"):
            s=set(map(int,e["support_recording_ids"].split(";")))
            q=set(map(int,e["query_recording_ids"].split(";")))
            self.assertTrue(s.isdisjoint(q));self.assertEqual(len(s),e["support_shots"]);self.assertEqual(len(s|q),3)
    def test_validator_accepts_complete_manifest(self):
        eps=[e for split in self.split for e in enumerate_episodes(self.recordings,self.split,split)]
        checks=validate_manifests(self.recordings,self.split,eps)
        self.assertTrue(checks["support_query_recording_disjoint"]);self.assertEqual(checks["n_recordings"],78)
if __name__=="__main__": unittest.main()
