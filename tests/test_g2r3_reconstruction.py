import sys,unittest
from pathlib import Path
import numpy as np
import networkx as nx
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_g2r3_reconstruction import spatial_graph,rewired_graph,adjacency,PairNet

class G2R3ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.coords=np.argwhere(np.ones((8,8),dtype=bool)).astype(np.float32)
        self.graph=spatial_graph(self.coords,6)
    def test_spatial_graph_shape_and_connectedness(self):
        self.assertEqual(self.graph.number_of_nodes(),64)
        self.assertTrue(nx.is_connected(self.graph))
        self.assertTrue(all(d>=6 for _,d in self.graph.degree()))
        self.assertFalse(nx.number_of_selfloops(self.graph)>0)
    def test_rewire_preserves_degree_and_connectivity(self):
        h=rewired_graph(self.graph,2026112201)
        self.assertEqual(self.graph.number_of_edges(),h.number_of_edges())
        self.assertEqual(sorted(dict(self.graph.degree()).values()),sorted(dict(h.degree()).values()))
        self.assertTrue(nx.is_connected(h))
        self.assertEqual(nx.number_of_selfloops(h),0)
    def test_pair_decoder_is_symmetric(self):
        torch.manual_seed(7)
        adj=adjacency(self.graph,torch.device("cpu"))
        model=PairNet("spatial_k6",adj).eval()
        x=torch.randn(64,10)
        a=torch.tensor([0,1,4]);b=torch.tensor([5,7,9])
        geom=torch.randn(3,5)
        with torch.no_grad():
            p1=model(x,a,b,geom);p2=model(x,b,a,geom)
        self.assertTrue(torch.allclose(p1,p2,atol=1e-7,rtol=1e-6))
    def test_node_and_graph_forward_backward(self):
        x=torch.randn(64,10);a=torch.tensor([0,1,2]);b=torch.tensor([3,4,5]);geom=torch.randn(3,5)
        for kind,adj in [("node_mlp",None),("spatial_k6",adjacency(self.graph,torch.device("cpu")))]:
            model=PairNet(kind,adj);loss=model(x,a,b,geom).square().mean();loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(p.grad is not None for p in model.parameters()))
if __name__=="__main__": unittest.main()
