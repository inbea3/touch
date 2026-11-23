#!/usr/bin/env python3
"""Stage 20: object-disjoint MLP baselines for HAS future pressure and correlation."""
from __future__ import annotations
import argparse,csv,hashlib,json,platform,random,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from scipy.stats import spearmanr

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()

def seed_all(seed):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None

class MLP(nn.Module):
 def __init__(self,nin,nout):
  super().__init__()
  self.net=nn.Sequential(nn.Linear(nin,256),nn.ReLU(),nn.Dropout(.2),
    nn.Linear(256,128),nn.ReLU(),nn.Dropout(.2),nn.Linear(128,nout))
 def forward(self,x): return self.net(x)

def metrics(y,p):
 err=p-y
 per_sample_mae=np.mean(np.abs(err),axis=1)
 per_sample_rmse=np.sqrt(np.mean(err**2,axis=1))
 spr=[]
 for a,b in zip(y,p):
  q=spearmanr(a,b).statistic
  if np.isfinite(q): spr.append(float(q))
 return {"mae":float(np.mean(np.abs(err))),"rmse":float(np.sqrt(np.mean(err**2))),
         "mean_episode_spearman":float(np.mean(spr)) if spr else None,
         "episode_mae":per_sample_mae,"episode_rmse":per_sample_rmse}

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--data",default="results/has_feature_dataset/episode_features_targets.npz")
 ap.add_argument("--index",default="results/has_feature_dataset/episode_index.csv")
 ap.add_argument("--out",default="results/has_mlp_baseline")
 ap.add_argument("--epochs",type=int,default=200)
 ap.add_argument("--patience",type=int,default=20)
 ap.add_argument("--batch-size",type=int,default=128)
 ap.add_argument("--seeds",default="20261123,20261124,20261125")
 a=ap.parse_args()
 out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
 start=time.time(); logp=out/"run.log"
 def log(m):
  line=f"[{datetime.now(timezone.utc).isoformat()}] {m}"; print(line,flush=True)
  with logp.open("a") as f:f.write(line+"\n")
 with np.load(a.data) as z:
  X=z["X_early"].astype(np.float32); Ym=z["Y_future_mean"].astype(np.float32)
  R=z["Y_future_corr_upper"].astype(np.float32); ci=z["corr_i"]; cj=z["corr_j"]
 rows=list(csv.DictReader(open(a.index,encoding="utf-8")))
 if len(rows)!=len(X) or len(rows)!=4192: raise RuntimeError("Feature/index row mismatch")
 splits=np.array([r["grouped_split"] for r in rows]); objects=np.array([r["object_group"] for r in rows])
 train=np.where(splits=="train")[0]; val=np.where(splits=="validation")[0]; test=np.where(splits=="test")[0]
 objsets={s:set(objects[splits==s]) for s in ("train","validation","test")}
 if any(objsets[x]&objsets[y] for x,y in (("train","validation"),("train","test"),("validation","test"))): raise RuntimeError("Object leakage")
 # Primary analysis excludes the two globally constant channels; all-channel sensitivity is retained.
 policies={"primary_42ch":np.array([i for i in range(44) if i not in (18,25)]),
           "sensitivity_all44":np.arange(44)}
 tasks={}
 for policy,ch in policies.items():
  xcols=np.concatenate([ch,44+ch,88+ch,132+ch])
  paircols=np.where(np.isin(ci,ch)&np.isin(cj,ch))[0]
  tasks[f"{policy}_future_mean"]={"X":X[:,xcols],"Y":Ym[:,ch],"kind":"future_mean","nch":len(ch)}
  tasks[f"{policy}_future_corr"]={"X":X[:,xcols],"Y":R[:,paircols],"kind":"future_corr","nch":len(ch)}
 config={"data":"HAS Stage19 artifact","n_episodes":len(rows),"split_seed":20261123,
  "channel_policy":{"primary_42ch":"exclude channels 18 and 25","sensitivity_all44":"retain all 44 channels"},
  "dimensions":{"primary_42ch":{"input":168,"future_mean":42,"future_corr":861},
                "sensitivity_all44":{"input":176,"future_mean":44,"future_corr":946}},
  "architecture":"input -> 256 ReLU -> dropout(0.2) -> 128 ReLU -> dropout(0.2) -> output","optimizer":"AdamW","dropout":0.2,
  "learning_rate":0.001,"weight_decay":0.001,"batch_size":a.batch_size,
  "max_epochs":a.epochs,"early_stopping_patience":a.patience,"seeds":[int(x) for x in a.seeds.split(",")],
  "selection":"minimum validation standardized MSE; test evaluated only after checkpoint selection",
  "normalization":"X and Y mean/std fitted on Train objects only",
  "metrics":["MAE","RMSE","mean episode-wise Spearman"],"unit":"episode; object-level aggregation"}
 (out/"config.json").write_text(json.dumps(config,indent=2))
 all_results={}
 for task,t in tasks.items():
  xx=t["X"]; yy=t["Y"]
  xm=xx[train].mean(0); xs=xx[train].std(0); xs[xs<1e-8]=1
  ym=yy[train].mean(0); ys=yy[train].std(0); ys[ys<1e-8]=1
  xn=((xx-xm)/xs).astype(np.float32); yn=((yy-ym)/ys).astype(np.float32)
  np.savez_compressed(out/f"{task}_scalers.npz",x_mean=xm,x_std=xs,y_mean=ym,y_std=ys)
  # simple non-neural reference baselines
  if t["kind"]=="future_mean":
   last=xx[:,3*t["nch"]:4*t["nch"]] # last-value block after channel filtering
   base_preds={"train_mean":np.repeat(yy[train].mean(0,keepdims=True),len(test),axis=0),"last_value":last[test]}
  else:
   base_preds={"zero_corr":np.zeros_like(yy[test]),"train_mean_corr":np.repeat(yy[train].mean(0,keepdims=True),len(test),axis=0)}
  base_metrics={k:{q:v for q,v in metrics(yy[test],p).items() if q not in ("episode_mae","episode_rmse")} for k,p in base_preds.items()}
  seed_results=[]
  for sd in config["seeds"]:
   seed_all(sd); model=MLP(xn.shape[1],yn.shape[1])
   opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
   loss_fn=nn.MSELoss()
   dl=DataLoader(TensorDataset(torch.from_numpy(xn[train]),torch.from_numpy(yn[train])),batch_size=a.batch_size,shuffle=True)
   best=float("inf"); best_epoch=0; wait=0; trace=[]
   for ep in range(1,a.epochs+1):
    model.train(); losses=[]
    for xb,yb in dl:
     opt.zero_grad(); loss=loss_fn(model(xb),yb); loss.backward(); opt.step(); losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad(): vl=float(loss_fn(model(torch.from_numpy(xn[val])),torch.from_numpy(yn[val])).item())
    tr=float(np.mean(losses)); trace.append({"epoch":ep,"train_mse":tr,"validation_mse":vl})
    if vl<best-1e-7:
     best=vl;best_epoch=ep;wait=0;torch.save(model.state_dict(),out/f"{task}_seed{sd}_best.pt")
    else:
     wait+=1
     if wait>=a.patience: break
   model.load_state_dict(torch.load(out/f"{task}_seed{sd}_best.pt",map_location="cpu",weights_only=True));model.eval()
   with torch.no_grad(): pred=model(torch.from_numpy(xn[test])).numpy()*ys+ym
   met=metrics(yy[test],pred)
   per_obj=[]
   for ob in sorted(objsets["test"]):
    ix=np.where(objects[test]==ob)[0]; mm=metrics(yy[test][ix],pred[ix])
    per_obj.append({"object_group":ob,"n_episodes":len(ix),"mae":mm["mae"],"rmse":mm["rmse"],"mean_episode_spearman":mm["mean_episode_spearman"]})
   with (out/f"{task}_seed{sd}_trace.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["epoch","train_mse","validation_mse"]);w.writeheader();w.writerows(trace)
   with (out/f"{task}_seed{sd}_test_predictions.npz").open("wb") as f:
    np.savez_compressed(f,y_true=yy[test],y_pred=pred)
   seed_results.append({"seed":sd,"best_epoch":best_epoch,"best_validation_mse":best,
    "test":{"mae":met["mae"],"rmse":met["rmse"],"mean_episode_spearman":met["mean_episode_spearman"]},
    "per_object":per_obj})
   log(f"task={task} seed={sd} best_epoch={best_epoch} test_mae={met['mae']:.6f} test_rmse={met['rmse']:.6f}")
  # mean/std across initialization seeds, not independent test objects
  all_results[task]={"test_baselines":base_metrics,"runs":seed_results}
  del xn,yn
 with (out/"test_episode_index.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows([rows[i] for i in test])
 (out/"metrics.json").write_text(json.dumps(all_results,indent=2))
 (out/"environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\ntorch={torch.__version__}\nnumpy={np.__version__}\n")
 log(f"COMPLETE elapsed_seconds={time.time()-start:.1f}")
 files=sorted(p for p in out.iterdir() if p.is_file() and p.name!="manifest.sha256")
 (out/"manifest.sha256").write_text("\n".join(f"{sha(p)}  {p.name}" for p in files)+"\n")
 print(json.dumps({"tasks":list(tasks),"episodes":len(rows),"test_objects":sorted(objsets["test"]),"results":all_results},indent=2))

if __name__=="__main__":main()
