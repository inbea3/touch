#!/usr/bin/env python3
"""Reconstructed full-548 STAG spatial-graph robustness experiment.

New experiment; not a recovery of the unavailable historical G2-R3 source.
"""
from __future__ import annotations
import argparse,csv,hashlib,json,os,platform,random,sys,time
from pathlib import Path
import numpy as np
import scipy.io as sio
import torch
from torch import nn
import torch.nn.functional as F
import networkx as nx

SPLIT_SEEDS=[20261119,20261120,20261121]
INIT_SEEDS=[20261123,20261124,20261125]
REWIRE_SEEDS=[2026112201,2026112202,2026112203]
CONDITIONS=["node_mlp","spatial_k6","rewired_1","rewired_2","rewired_3"]

def sha256(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
 torch.use_deterministic_algorithms(True,warn_only=True)

def as_vec(m,k,dtype):
 if k not in m: raise KeyError(k)
 return np.asarray(m[k]).reshape(-1).astype(dtype)

def load_recordings(path):
 m=sio.loadmat(path,simplify_cells=True)
 p=np.asarray(m["pressure"]); rec=as_vec(m,"recordingId",np.int64)
 obj=as_vec(m,"objectId",np.int64); ts=as_vec(m,"ts",np.float64)
 frame=as_vec(m,"frame",np.int64); valid=as_vec(m,"hasValidLabel",bool)
 names=np.asarray(m["objects"],dtype=object).reshape(-1)
 mask=np.nanstd(p.astype(np.float32),axis=0)>0
 if p.shape!=(135187,32,32) or int(mask.sum())!=548: raise ValueError("Unexpected STAG pressure shape or active sensor count")
 coords=np.argwhere(mask).astype(np.float32)
 if coords.shape!=(548,2): raise ValueError("548-node coordinate shape mismatch")
 out=[]
 for rid in sorted(np.unique(rec).tolist()):
  ix=np.flatnonzero(rec==rid); oidset=np.unique(obj[ix])
  if len(oidset)!=1: raise ValueError(f"recording {rid} maps to multiple objects")
  oid=int(oidset[0])
  if oid==0: continue
  ix=ix[np.argsort(ts[ix],kind="stable")]
  vp=np.flatnonzero(valid[ix])
  if len(vp)<9: continue
  ei=ix[vp[:5]]; fi=ix[vp[5:]]
  if len(fi)<4: continue
  x=np.clip((p[ei][:,mask].astype(np.float32)-500.)/150.,0.,1.)
  y=np.clip((p[fi][:,mask].astype(np.float32)-500.)/150.,0.,1.)
  # Node features: eight pressure summaries + normalized physical coordinates.
  t=ts[ei];tc=t-t.mean();den=float(tc@tc)
  slope=(tc[:,None]*(x-x.mean(0))).sum(0)/den if den>0 else np.zeros(548,np.float32)
  xy=coords.copy();xy[:,0]/=31.;xy[:,1]/=31.
  feat=np.column_stack([x.mean(0),x.std(0),x.min(0),x.max(0),np.ptp(x,axis=0),slope,x[-1],x[0],xy]).astype(np.float32)
  z=y-y.mean(0,keepdims=True);sd=np.sqrt(np.sum(z*z,axis=0))
  good=sd>1e-10
  corr=np.full((548,548),np.nan,np.float32)
  if good.sum()>=2:
   zn=z[:,good]/sd[good]
   c=np.clip(zn.T@zn,-1,1)
   gi=np.flatnonzero(good);corr[np.ix_(gi,gi)]=c
   np.fill_diagonal(corr,np.nan)
  a,b=np.triu_indices(548,1);target=corr[a,b];ok=np.isfinite(target)
  if ok.sum()<1000: continue
  out.append({"rid":int(rid),"oid":oid,"name":str(names[oid]),"x":feat,
    "a":a[ok].astype(np.int64),"b":b[ok].astype(np.int64),"y":target[ok].astype(np.float32),
    "input_span":float(ts[ei[-1]]-ts[ei[0]]),"input_contiguous":bool(np.all(np.diff(frame[ei])==1)),
    "n_future":int(len(fi))})
 if len(out)!=78: raise ValueError(f"Expected 78 non-empty recordings; constructed {len(out)}")
 return out,coords

def spatial_graph(coords,k=6):
 g=nx.Graph();g.add_nodes_from(range(len(coords)))
 d=np.sqrt(((coords[:,None,:]-coords[None,:,:])**2).sum(-1))
 for i in range(len(coords)):
  for j in np.argsort(d[i])[1:k+1]:g.add_edge(i,int(j))
 return g

def rewired_graph(g,seed):
 h=g.copy()
 nx.connected_double_edge_swap(h,nswap=10*h.number_of_edges(),seed=seed)
 if not nx.is_connected(h): raise RuntimeError("Rewired graph disconnected")
 if sorted(dict(g.degree()).values())!=sorted(dict(h.degree()).values()): raise RuntimeError("Degree sequence changed")
 return h

def adjacency(g,device):
 n=g.number_of_nodes();a=np.zeros((n,n),np.float32)
 for i,j in g.edges():a[i,j]=1.;a[j,i]=1.
 a+=np.eye(n,dtype=np.float32)
 deg=a.sum(1)
 a=a/np.sqrt(deg[:,None]*deg[None,:])
 return torch.tensor(a,device=device)

class PairNet(nn.Module):
 def __init__(self,kind,adj=None):
  super().__init__();self.kind=kind
  self.node=nn.Sequential(nn.Linear(10,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
  if kind!="node_mlp":
   self.register_buffer("adj",adj)
   self.msg1=nn.Linear(64,64);self.msg2=nn.Linear(64,64)
  self.decoder=nn.Sequential(nn.Linear(64*3+5,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1),nn.Tanh())
 def forward(self,x,a,b,geom):
  h=self.node(x)
  if self.kind!="node_mlp":
   h=F.relu(self.msg1(self.adj@h)+h)
   h=F.relu(self.msg2(self.adj@h)+h)
  hi,hj=h[a],h[b]
  pair=torch.cat([hi+hj,torch.abs(hi-hj),hi*hj,geom],dim=1)
  return self.decoder(pair).squeeze(1)

def pair_geometry(coords,a,b,device):
 c=coords
 dx=np.abs(c[a,1]-c[b,1])/31.;dy=np.abs(c[a,0]-c[b,0])/31.
 geom=np.column_stack([c[a,1]/31.+c[b,1]/31.,c[a,0]/31.+c[b,0]/31.,dx,dy,np.hypot(dx,dy)/np.sqrt(2)]).astype(np.float32)
 return torch.tensor(geom,device=device)

def metrics(y,p):
 e=np.asarray(p)-np.asarray(y)
 if len(y)>1 and np.std(y)>1e-12 and np.std(p)>1e-12:
  from scipy.stats import spearmanr
  rho=float(spearmanr(y,p).statistic)
 else:rho=float("nan")
 return {"mae":float(np.mean(np.abs(e))),"rmse":float(np.sqrt(np.mean(e*e))),"spearman":rho}

def fit_one(data,train_oids,val_oids,test_oids,condition,seed,coords,args,outdir,split_seed,fold_i):
 seed_all(seed);device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 graph=None
 if condition=="spatial_k6":graph=spatial_graph(coords,6)
 elif condition.startswith("rewired_"):graph=rewired_graph(spatial_graph(coords,6),REWIRE_SEEDS[int(condition[-1])-1])
 adj=adjacency(graph,device) if graph is not None else None
 model=PairNet(condition,adj).to(device)
 opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
 # Training-object-only feature scaling; coordinates remain on the same scale.
 train=[d for d in data if d["oid"] in train_oids]
 val=[d for d in data if d["oid"] in val_oids]
 test=[d for d in data if d["oid"] in test_oids]
 xx=np.concatenate([d["x"][:,:8] for d in train])
 mu=xx.mean(0);sd=xx.std(0);sd[sd<1e-6]=1.
 for group in (train,val,test):
  for d in group:
   d["xt"]=d["x"].copy();d["xt"][:,:8]=(d["x"][:,:8]-mu)/sd
 # coordinates per pair, compute per recording as pairs have valid target subset.
 def eval_group(group):
  model.eval();ys=[];ps=[];rec_rows=[]
  with torch.no_grad():
   for d in group:
    a=torch.tensor(d["a"],device=device);b=torch.tensor(d["b"],device=device)
    x=torch.tensor(d["xt"],device=device);geom=pair_geometry(coords,d["a"],d["b"],device)
    pred=[]
    for st in range(0,len(a),args.eval_chunk):
     pred.append(model(x,a[st:st+args.eval_chunk],b[st:st+args.eval_chunk],geom[st:st+args.eval_chunk]).cpu().numpy())
    p=np.concatenate(pred);m=metrics(d["y"],p);rec_rows.append({"recording_id":d["rid"],"object_id":d["oid"],**m})
    ys.append(d["y"]);ps.append(p)
  # macro average recording-level metrics within each object
  obj={}
  for r in rec_rows:obj.setdefault(r["object_id"],[]).append(r)
  obj_rows=[]
  for oid,rs in obj.items():
   obj_rows.append({"object_id":oid,**{k:float(np.mean([r[k] for r in rs])) for k in ("mae","rmse","spearman")}})
  return rec_rows,obj_rows
 base_name=f"split{split_seed}_fold{fold_i+1}_{condition}_init{seed}"
 run_dir=outdir/base_name
 if run_dir.exists() and (run_dir/"summary.json").exists():
  return json.loads((run_dir/"summary.json").read_text())
 if run_dir.exists():
  attempt=2
  while (outdir/f"{base_name}_attempt{attempt}").exists(): attempt+=1
  run_dir=outdir/f"{base_name}_attempt{attempt}"
 run_dir.mkdir(parents=True,exist_ok=False)
 best=float("inf");best_epoch=0;wait=0;trace=[]
 log_path=run_dir/"run.log"
 with log_path.open("a") as lf: lf.write(f"START condition={condition} seed={seed} split_seed={split_seed} fold={fold_i+1}"+chr(10))
 config={"condition":condition,"seed":seed,"train_objects":sorted(train_oids),"validation_objects":sorted(val_oids),
  "test_objects":sorted(test_oids),"epochs":args.epochs,"patience":args.patience,"lr":args.lr,
  "weight_decay":args.weight_decay,"pair_sample_per_recording_epoch":args.pairs_per_recording,
  "eval_chunk":args.eval_chunk,"architecture":"node encoder 10-64-64; two residual mean-aggregating graph layers for graph conditions; symmetric pair decoder [sum,abs-difference,product,5D geometry] 197-128-64-1 tanh",
  "normalization":"first eight pressure features standardized using outer training objects only; xy coordinates fixed normalized grid"}
 (run_dir/"config.json").write_text(json.dumps(config,indent=2))
 for epoch in range(1,args.epochs+1):
  model.train();losses=[]
  for d in train:
   n=len(d["y"]);m=min(args.pairs_per_recording,n)
   ids=np.random.choice(n,m,replace=False)
   a=torch.tensor(d["a"][ids],device=device);b=torch.tensor(d["b"][ids],device=device)
   y=torch.tensor(d["y"][ids],device=device);x=torch.tensor(d["xt"],device=device)
   geom=pair_geometry(coords,d["a"][ids],d["b"][ids],device)
   opt.zero_grad(set_to_none=True);pred=model(x,a,b,geom);loss=F.smooth_l1_loss(pred,y)
   loss.backward();opt.step();losses.append(float(loss.detach().cpu()))
  vr,vo=eval_group(val);v=float(np.mean([r["mae"] for r in vo]))
  trace.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_object_macro_mae":v})
  with log_path.open("a") as lf: lf.write(f"epoch={epoch} train_loss={float(np.mean(losses)):.8f} val_object_macro_mae={v:.8f} best={best:.8f}"+chr(10))
  with (run_dir/"training_trace.csv").open("w",newline="") as f:
   w=csv.DictWriter(f,fieldnames=list(trace[0]));w.writeheader();w.writerows(trace)
  if v<best-args.min_delta:
   best=v;best_epoch=epoch;wait=0
   torch.save({"model":model.state_dict(),"epoch":epoch,"validation_mae":v},run_dir/"best_checkpoint.pt")
  else:
   wait+=1
   if wait>=args.patience:break
 ck=torch.load(run_dir/"best_checkpoint.pt",map_location=device,weights_only=False);model.load_state_dict(ck["model"])
 val_rec,val_obj=eval_group(val);test_rec,test_obj=eval_group(test)
 def write_rows(name,rows):
  if rows:
   with (run_dir/name).open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 write_rows("validation_recording_metrics.csv",val_rec);write_rows("validation_object_metrics.csv",val_obj)
 write_rows("test_recording_metrics.csv",test_rec);write_rows("test_object_metrics.csv",test_obj)
 summary={"condition":condition,"seed":seed,"best_epoch":best_epoch,"best_validation_object_macro_mae":best,
  "validation_object_macro":{k:float(np.mean([r[k] for r in val_obj])) for k in ("mae","rmse","spearman")},
  "test_object_macro_exploratory":{k:float(np.mean([r[k] for r in test_obj])) for k in ("mae","rmse","spearman")},
  "n_train_objects":len(train_oids),"n_validation_objects":len(val_oids),"n_test_objects":len(test_oids)}
 (run_dir/"summary.json").write_text(json.dumps(summary,indent=2))
 with log_path.open("a") as lf: lf.write(f"COMPLETE best_epoch={best_epoch} best_val_mae={best:.8f}"+chr(10))
 run_files=[q for q in run_dir.rglob("*") if q.is_file() and q.name!="manifest.sha256"]
 (run_dir/"manifest.sha256").write_text(chr(10).join(f"{sha256(q)}  {q.relative_to(run_dir)}" for q in sorted(run_files))+chr(10))
 (run_dir/"environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\nnumpy={np.__version__}\ntorch={torch.__version__}\nnetworkx={nx.__version__}\ndevice={device}\n")
 return summary

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--metadata",required=True);ap.add_argument("--output",default="results/g2r3_reconstruction")
 ap.add_argument("--epochs",type=int,default=100);ap.add_argument("--patience",type=int,default=15)
 ap.add_argument("--min-delta",type=float,default=1e-5);ap.add_argument("--lr",type=float,default=5e-4)
 ap.add_argument("--weight-decay",type=float,default=1e-3);ap.add_argument("--pairs-per-recording",type=int,default=8192)
 ap.add_argument("--eval-chunk",type=int,default=32768);ap.add_argument("--smoke-only",action="store_true")
 a=ap.parse_args()
 if a.smoke_only:
  a.epochs=1;a.patience=1;a.pairs_per_recording=256
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 data,coords=load_recordings(a.metadata)
 graph=spatial_graph(coords,6)
 graph_info={"nodes":graph.number_of_nodes(),"edges":graph.number_of_edges(),"connected":nx.is_connected(graph),
  "degree_min":min(dict(graph.degree()).values()),"degree_max":max(dict(graph.degree()).values()),
  "coordinate_order":"numpy.argwhere(active_mask), row-major"}
 (out/"data_audit.json").write_text(json.dumps({"recordings":len(data),"objects":len(set(d["oid"] for d in data)),
  "sensors":len(coords),"graph":graph_info,"recording_coverage":[{k:d[k] for k in ("rid","oid","name","input_span","input_contiguous","n_future")} for d in data]},indent=2))
 np.save(out/"sensor_coordinates.npy",np.asarray(coords,dtype="<f4"))
 for seed in REWIRE_SEEDS:
  rg=rewired_graph(graph,seed)
  edges=np.array(sorted((min(i,j),max(i,j)) for i,j in rg.edges()),dtype="<i4")
  np.save(out/f"rewired_edges_{seed}.npy",edges)
  if not nx.is_connected(rg) or sorted(dict(rg.degree()).values())!=sorted(dict(graph.degree()).values()):raise RuntimeError("Rewire invariant failed")
 np.save(out/"spatial_edges.npy",np.array(sorted((min(i,j),max(i,j)) for i,j in graph.edges()),dtype="<i4"))
 # One deterministic grouped split for smoke test; full 225 grid uses the frozen 5x3x3x5 design.
 oids=sorted(set(d["oid"] for d in data));summaries=[]\n status_path=out/"run_status.json"\n if status_path.exists():\n  try: summaries=json.loads(status_path.read_text()).get("runs",[])\n  except Exception: summaries=[]\n completed_keys={(r.get("split_seed"),r.get("fold"),r.get("condition"),r.get("seed")) for r in summaries}
 split_count=0
 for split_seed in SPLIT_SEEDS:
  perm=np.random.default_rng(split_seed).permutation(oids);folds=np.array_split(perm,5)
  for fold_i,test_arr in enumerate(folds):
   test_oids=set(map(int,test_arr));remaining=[x for x in oids if x not in test_oids]
   rng=np.random.default_rng(split_seed+fold_i*100);val_oids=set(map(int,rng.choice(remaining,max(1,round(.2*len(remaining))),replace=False)))
   train_oids=set(remaining)-val_oids
   for init_seed in INIT_SEEDS:
    for condition in CONDITIONS:
     if a.smoke_only and (split_seed!=SPLIT_SEEDS[0] or fold_i!=0 or init_seed!=INIT_SEEDS[0] or condition not in ("node_mlp","spatial_k6")):continue
     run_summary=fit_one(data,train_oids,val_oids,test_oids,condition,init_seed,coords,a,out,split_seed,fold_i)
     run_summary.update({"split_seed":split_seed,"fold":fold_i+1})
     summaries.append(run_summary);completed_keys.add(key);split_count+=1
     (out/"run_status.json").write_text(json.dumps({"completed_fits":len(completed_keys),"expected_fits":2 if a.smoke_only else 225,"runs":summaries},indent=2))
     print(json.dumps({"completed":split_count,"condition":condition,"split_seed":split_seed,"fold":fold_i+1,"init_seed":init_seed,**run_summary}),flush=True)
     if a.smoke_only and split_count>=2:break
    if a.smoke_only and split_count>=2:break
   if a.smoke_only and split_count>=2:break
  if a.smoke_only and split_count>=2:break
 if not a.smoke_only and len(completed_keys)!=225:raise RuntimeError(f"Expected 225 completed fits, got {len(completed_keys)}")
 (out/"final_summary.json").write_text(json.dumps({"experiment_id":"G2-R3-RECONSTRUCTION-20261124",
  "status":"smoke_complete" if a.smoke_only else "complete","n_fits":len(completed_keys),"runs":summaries},indent=2))
 files=[p for p in out.rglob("*") if p.is_file()]
 (out/"manifest.sha256").write_text("\n".join(f"{sha256(p)}  {p.relative_to(out)}" for p in sorted(files))+"\n")
if __name__=="__main__":main()
