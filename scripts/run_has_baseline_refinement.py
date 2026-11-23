#!/usr/bin/env python3
"""Stage 20b: stronger, validation-selected HAS regression baselines."""
from __future__ import annotations
import argparse,csv,hashlib,json,platform,random,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset

def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def seed_all(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s)
class ResidualMLP(nn.Module):
 def __init__(self,nin,nout):
  super().__init__();self.net=nn.Sequential(nn.Linear(nin,256),nn.ReLU(),nn.Dropout(.15),nn.Linear(256,128),nn.ReLU(),nn.Dropout(.15),nn.Linear(128,nout))
 def forward(self,x):return self.net(x)
def scores(y,p):
 e=p-y; sp=[]
 for a,b in zip(y,p):
  q=spearmanr(a,b).statistic
  if np.isfinite(q):sp.append(float(q))
 return {"mae":float(np.mean(np.abs(e))),"rmse":float(np.sqrt(np.mean(e*e))),"episode_spearman":float(np.mean(sp)) if sp else None}
def object_scores(y,p,objects):
 out=[]
 for ob in sorted(set(objects)):
  ix=np.where(objects==ob)[0];m=scores(y[ix],p[ix]);out.append({"object":ob,"n":len(ix),**m})
 return out
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--data",default="results/has_feature_dataset/episode_features_targets.npz")
 ap.add_argument("--index",default="results/has_feature_dataset/episode_index.csv")
 ap.add_argument("--out",default="results/has_baseline_refinement")
 ap.add_argument("--epochs",type=int,default=200);ap.add_argument("--patience",type=int,default=20)
 ap.add_argument("--seeds",default="20261123,20261124,20261125")
 a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);t0=time.time()
 def log(msg):
  line=f"[{datetime.now(timezone.utc).isoformat()}] {msg}";print(line,flush=True)
  with (out/"run.log").open("a") as f:f.write(line+"\n")
 with np.load(a.data) as z:
  X=z["X_early"].astype(np.float32);Ym=z["Y_future_mean"].astype(np.float32);R=z["Y_future_corr_upper"].astype(np.float32);ci=z["corr_i"];cj=z["corr_j"]
 rows=list(csv.DictReader(open(a.index,encoding="utf-8")))
 if len(rows)!=4192 or len(rows)!=len(X):raise RuntimeError("Expected 4192 aligned feature rows")
 split=np.array([r["grouped_split"] for r in rows]);objs=np.array([r["object_group"] for r in rows])
 tr=np.where(split=="train")[0];va=np.where(split=="validation")[0];te=np.where(split=="test")[0]
 objsets={s:set(objs[split==s]) for s in ("train","validation","test")}
 if any(objsets[x]&objsets[y] for x,y in (("train","validation"),("train","test"),("validation","test"))):raise RuntimeError("Object leakage")
 ch=np.array([i for i in range(44) if i not in (18,25)])
 xcols=np.concatenate([ch,44+ch,88+ch,132+ch]);last=X[:,132+ch]
 Xp=X[:,xcols]
 # Train-only scaling
 xm=Xp[tr].mean(0);xs=Xp[tr].std(0);xs[xs<1e-8]=1;Z=((Xp-xm)/xs).astype(np.float32)
 # Future mean: residual target to last observed value
 Ym=Ym[:,ch];resid=Ym-last
 ym=resid[tr].mean(0);ys=resid[tr].std(0);ys[ys<1e-8]=1
 Yr=((resid-ym)/ys).astype(np.float32)
 # Ridge alpha chosen only on validation
 ridge_results={}
 for task,Y in [("future_mean_residual_ridge",resid),("future_corr_ridge",R[:,np.where(np.isin(ci,ch)&np.isin(cj,ch))[0]])]:
  ymu=Y[tr].mean(0);yst=Y[tr].std(0);yst[yst<1e-8]=1;Yz=(Y-ymu)/yst
  candidates=[]
  for alpha in [0.01,0.1,1.0,10.0,100.0,1000.0]:
   model=Ridge(alpha=alpha,solver="lsqr")
   model.fit(Z[tr],Yz[tr]);pv=model.predict(Z[va])
   candidates.append((float(np.mean((pv-Yz[va])**2)),alpha,model))
  bestval,alpha,model=min(candidates,key=lambda x:x[0])
  pred=model.predict(Z[te])*yst+ymu
  if task=="future_mean_residual_ridge":pred=pred+last[te]
  truth=Ym[te] if task=="future_mean_residual_ridge" else Y[te]
  m=scores(truth,pred);per=object_scores(truth,pred,objs[te])
  ridge_results[task]={"selected_alpha":alpha,"validation_standardized_mse":bestval,"test":m,"object_macro":{"mae":float(np.mean([x["mae"] for x in per])),"rmse":float(np.mean([x["rmse"] for x in per]))},"per_object":per}
  import joblib;joblib.dump(model,out/f"{task}_ridge.joblib")
  log(f"{task} alpha={alpha} test_mae={m['mae']:.6f} test_rmse={m['rmse']:.6f}")
 # Correlation shrinkage: select lambda multiplying train-mean template, validation only.
 Rc=R[:,np.where(np.isin(ci,ch)&np.isin(cj,ch))[0]]
 train_template=Rc[tr].mean(0)
 shrink=[]
 for lam in [0,.1,.25,.5,.75,1.0]:
  pv=np.repeat((lam*train_template)[None,:],len(va),axis=0)
  shrink.append((float(np.mean((pv-Rc[va])**2)),lam))
 _,best_lam=min(shrink)
 pred=np.repeat((best_lam*train_template)[None,:],len(te),axis=0)
 per=object_scores(Rc[te],pred,objs[te])
 ridge_results["correlation_shrinkage"]={"selected_lambda":best_lam,"validation_mse":min(shrink)[0],"test":scores(Rc[te],pred),
  "object_macro":{"mae":float(np.mean([x["mae"] for x in per])),"rmse":float(np.mean([x["rmse"] for x in per]))},"per_object":per}
 # Residual MLP predicts future_mean - last; checkpoint selection on validation residual standardized MSE.
 runs=[]
 for sd in [int(v) for v in a.seeds.split(",")]:
  seed_all(sd);model=ResidualMLP(Z.shape[1],Yr.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-3)
  dl=DataLoader(TensorDataset(torch.from_numpy(Z[tr]),torch.from_numpy(Yr[tr])),batch_size=128,shuffle=True)
  best=float("inf");be=0;wait=0;trace=[]
  for ep in range(1,a.epochs+1):
   model.train();ls=[]
   for xb,yb in dl:
    opt.zero_grad();loss=nn.functional.mse_loss(model(xb),yb);loss.backward();opt.step();ls.append(float(loss.detach()))
   model.eval()
   with torch.no_grad():vl=float(nn.functional.mse_loss(model(torch.from_numpy(Z[va])),torch.from_numpy(Yr[va])).item())
   trace.append({"epoch":ep,"train_mse":float(np.mean(ls)),"validation_mse":vl})
   if vl<best-1e-7:
    best=vl;be=ep;wait=0;torch.save(model.state_dict(),out/f"residual_mlp_seed{sd}.pt")
   else:
    wait+=1
    if wait>=a.patience:break
  model.load_state_dict(torch.load(out/f"residual_mlp_seed{sd}.pt",map_location="cpu",weights_only=True));model.eval()
  with torch.no_grad():pr=model(torch.from_numpy(Z[te])).numpy()*ys+ym+last[te]
  m=scores(Ym[te],pr);per=object_scores(Ym[te],pr,objs[te])
  with (out/f"residual_mlp_seed{sd}_trace.csv").open("w",newline="") as f:
   w=csv.DictWriter(f,fieldnames=["epoch","train_mse","validation_mse"]);w.writeheader();w.writerows(trace)
  np.savez_compressed(out/f"residual_mlp_seed{sd}_predictions.npz",y_true=Ym[te],y_pred=pr)
  runs.append({"seed":sd,"best_epoch":be,"validation_mse":best,"test":m,"object_macro":{"mae":float(np.mean([x["mae"] for x in per])),"rmse":float(np.mean([x["rmse"] for x in per]))},"per_object":per})
  log(f"residual_mlp seed={sd} epoch={be} test_mae={m['mae']:.6f} test_rmse={m['rmse']:.6f}")
 result={"ridge":ridge_results,"residual_mlp":runs}
 (out/"config.json").write_text(json.dumps({"input":"168-D primary 42-channel","mean_target":"future mean minus last observed value; reconstruct by adding last value","corr_target":"861 upper-triangle Pearson values","ridge_alpha_grid":[.01,.1,1,10,100,1000],"correlation_shrinkage_grid":[0,.1,.25,.5,.75,1],"residual_mlp":{"layers":[168,256,128,42],"dropout":.15,"optimizer":"AdamW","lr":.0005,"weight_decay":.001,"max_epochs":a.epochs,"patience":a.patience},"seeds":[int(v) for v in a.seeds.split(",")],"scaling":"train-only","selection":"validation only","test_objects":sorted(objsets["test"])},indent=2))
 (out/"metrics.json").write_text(json.dumps(result,indent=2))
 (out/"environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\nnumpy={np.__version__}\ntorch={torch.__version__}\n")
 log(f"COMPLETE elapsed={time.time()-t0:.1f}s")
 files=sorted(p for p in out.iterdir() if p.is_file() and p.name!="manifest.sha256")
 (out/"manifest.sha256").write_text("\n".join(f"{sha(p)}  {p.name}" for p in files)+"\n")
 print(json.dumps(result,indent=2))
if __name__=="__main__":main()
