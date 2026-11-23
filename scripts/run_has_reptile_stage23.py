#!/usr/bin/env python3
"""Stage 23: HAS cross-object Reptile vs update-matched pooled MLP."""
from __future__ import annotations
import argparse,csv,copy,hashlib,json,platform,random,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.stats import spearmanr

class Net(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(168,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,42))
 def forward(self,x):return self.net(x)
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def seed_all(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s)
def met(y,p):
 e=p-y;sp=[]
 for a,b in zip(y,p):
  if np.std(a)>1e-12 and np.std(b)>1e-12:
   v=spearmanr(a,b).statistic
   if np.isfinite(v):sp.append(float(v))
 return {"mae":float(np.mean(np.abs(e))),"rmse":float(np.sqrt(np.mean(e*e))),"spearman":float(np.mean(sp)) if sp else None}
def mean_state(states):
 out={}
 for k in states[0]:
  out[k]=torch.stack([s[k].detach().cpu() for s in states]).mean(0)
 return out
def adapt(model,X,Y,steps,lr,wd):
 m=copy.deepcopy(model);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=wd)
 m.train()
 for _ in range(steps):
  opt.zero_grad();loss=nn.functional.mse_loss(m(X),Y);loss.backward();opt.step()
 return m
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--data",default="results/has_window_dataset")
 ap.add_argument("--window",type=int,default=5)
 ap.add_argument("--out",default="results/has_reptile_stage23")
 ap.add_argument("--iterations",type=int,default=300)
 ap.add_argument("--meta-batch",type=int,default=4)
 ap.add_argument("--inner-steps",type=int,default=5)
 ap.add_argument("--inner-lr",type=float,default=1e-3)
 ap.add_argument("--outer-step",type=float,default=.05)
 ap.add_argument("--adapt-steps",type=int,default=10)
 ap.add_argument("--seeds",default="20261123,20261124,20261125")
 ap.add_argument("--support-repeats",type=int,default=5)
 a=ap.parse_args();root=Path(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);t0=time.time()
 def log(s):
  line=f"[{datetime.now(timezone.utc).isoformat()}] {s}";print(line,flush=True)
  with (out/"run.log").open("a") as f:f.write(line+"\n")
 with np.load(root/f"window_{a.window}.npz") as z:
  X=z["X"].astype(np.float32);Y=z["Y_future_mean"].astype(np.float32)
 rows=list(csv.DictReader(open(root/f"window_{a.window}_index.csv",encoding="utf-8")))
 if len(rows)!=len(X) or len(rows)!=4192:raise RuntimeError("Dataset/index alignment failure")
 split=np.array([r["grouped_split"] for r in rows]);obj=np.array([r["object_group"] for r in rows])
 ids=np.array([r["official_split"]+":"+r["episode_index"] for r in rows])
 sets={s:set(obj[split==s]) for s in ("train","validation","test")}
 if any(sets[x]&sets[y] for x,y in (("train","validation"),("train","test"),("validation","test"))):raise RuntimeError("Object leakage")
 tr=np.where(split=="train")[0];va=np.where(split=="validation")[0];te=np.where(split=="test")[0]
 # Train-only standardization.
 xm=X[tr].mean(0);xs=X[tr].std(0);xs[xs<1e-8]=1
 ym=Y[tr].mean(0);ys=Y[tr].std(0);ys[ys<1e-8]=1
 Z=((X-xm)/xs).astype(np.float32);T=((Y-ym)/ys).astype(np.float32)
 xt=torch.from_numpy(Z);yt=torch.from_numpy(T)
 task_idx={s:{o:np.where((split==s)&(obj==o))[0] for o in sorted(sets[s])} for s in sets}
 config={"window":a.window,"input_dim":168,"target_dim":42,"model":[168,128,64,42],"iterations":a.iterations,
 "meta_batch_tasks":a.meta_batch,"inner_steps":a.inner_steps,"inner_lr":a.inner_lr,"outer_step":a.outer_step,
 "inner_batch_size":8,"pooled_updates_per_iteration":a.meta_batch*a.inner_steps,"pooled_batch_size":8,
 "adapt_steps":a.adapt_steps,"shots":[1,5,10],"support_repeats":a.support_repeats,
 "seeds":[int(v) for v in a.seeds.split(",")],"train_only_scalers":True,
 "object_counts":{s:len(sets[s]) for s in sets},"episode_counts":{s:int(np.sum(split==s)) for s in sets},
 "test_status":"exploratory only; fixed test objects were inspected in prior stages"}
 (out/"config.json").write_text(json.dumps(config,indent=2))
 all_episode=[];all_object=[];trace_all=[];val_all=[]
 for seed in [int(v) for v in a.seeds.split(",")]:
  seed_all(seed);init=Net();initial=copy.deepcopy(init.state_dict())
  # Reptile outer loop: uniform task sampling, support-only inner updates.
  rept=Net();rept.load_state_dict(initial);rtrace=[]
  train_objects=sorted(sets["train"])
  for it in range(1,a.iterations+1):
   chosen=np.random.choice(train_objects,size=min(a.meta_batch,len(train_objects)),replace=False)
   adapted_states=[];losses=[]
   for ob in chosen:
    ix=task_idx["train"][ob];sup=np.random.choice(ix,size=8,replace=len(ix)<8)
    task=Net();task.load_state_dict(rept.state_dict());opt=torch.optim.AdamW(task.parameters(),lr=a.inner_lr,weight_decay=1e-3)
    task.train()
    for _ in range(a.inner_steps):
     opt.zero_grad();loss=nn.functional.mse_loss(task(xt[sup]),yt[sup]);loss.backward();opt.step();losses.append(float(loss.detach()))
    adapted_states.append(copy.deepcopy(task.state_dict()))
   base=rept.state_dict()
   with torch.no_grad():
    for name,param in rept.named_parameters():
     avg=torch.stack([st[name].to(param.device) for st in adapted_states]).mean(0)
     param.add_(avg-param,alpha=a.outer_step)
   rtrace.append({"iteration":it,"support_loss_mean":float(np.mean(losses)),"sampled_tasks":";".join(chosen)})
   if it%50==0:log(f"seed={seed} reptile iteration={it}/{a.iterations}")
  torch.save(rept.state_dict(),out/f"reptile_seed{seed}.pt")
  # Update-matched pooled MLP: exactly B*K gradient steps per meta-iteration.
  pooled=Net();pooled.load_state_dict(initial);opt=torch.optim.AdamW(pooled.parameters(),lr=a.inner_lr,weight_decay=1e-3);ptrace=[]
  for step in range(a.iterations*a.meta_batch*a.inner_steps):
   batch=np.random.choice(tr,size=8,replace=len(tr)<8)
   opt.zero_grad();loss=nn.functional.mse_loss(pooled(xt[batch]),yt[batch]);loss.backward();opt.step()
   if step%100==0 or step==a.iterations*a.meta_batch*a.inner_steps-1:
    ptrace.append({"update":step+1,"train_loss":float(loss.detach())})
  torch.save(pooled.state_dict(),out/f"pooled_seed{seed}.pt")
  with (out/f"reptile_seed{seed}_trace.csv").open("w",newline="") as f:
   w=csv.DictWriter(f,fieldnames=list(rtrace[0]));w.writeheader();w.writerows(rtrace)
  with (out/f"pooled_seed{seed}_trace.csv").open("w",newline="") as f:
   w=csv.DictWriter(f,fieldnames=list(ptrace[0]));w.writeheader();w.writerows(ptrace)
  # Validation and test: identical support IDs for Reptile and pooled initializations.
  for sp,ixs in (("validation",va),("test",te)):
   for ob,oi in task_idx[sp].items():
    if len(oi)<12:raise RuntimeError(f"Too few episodes for support/query in {ob}")
    for rep in range(a.support_repeats):
     rng=np.random.default_rng(seed*100000+rep*1000+sum(map(ord,ob))+(0 if sp=="validation" else 500000))
     perm=rng.permutation(oi)
     for shots in (1,5,10):
      support=perm[:shots];query=perm[shots:]
      if len(query)==0:continue
      # Same support set and same adaptation budget for both initializations.
      rm=Net();rm.load_state_dict(rept.state_dict());pm=Net();pm.load_state_dict(pooled.state_dict())
      ra=adapt(rm,xt[support],yt[support],a.adapt_steps,a.inner_lr,1e-3)
      pa=adapt(pm,xt[support],yt[support],a.adapt_steps,a.inner_lr,1e-3)
      with torch.no_grad():
       ytrue=Y[query]
       preds={"reptile_zero_shot":rm(xt[query]).numpy()*ys+ym,
        "reptile_adapted":ra(xt[query]).numpy()*ys+ym,
        "pooled_zero_shot":pm(xt[query]).numpy()*ys+ym,
        "pooled_adapted":pa(xt[query]).numpy()*ys+ym,
        "persistence":X[query,126:168],
        "train_mean":np.repeat(ym[None,:],len(query),axis=0)}
      support_ids=";".join(ids[support]);query_ids=";".join(ids[query])
      for model_name,pred in preds.items():
       m=met(ytrue,pred)
       all_episode.append({"split":sp,"seed":seed,"repeat":rep,"object":ob,"shots":shots,
        "model":model_name,"n_support":len(support),"n_query":len(query),
        "support_ids":support_ids,"query_ids":query_ids,**m})
  log(f"seed={seed} training complete; validation/test episodes evaluated")
 # Aggregate: average repeats and seeds within object first; object remains inferential unit.
 buckets={}
 for r in all_episode:
  k=(r["split"],r["object"],r["shots"],r["model"]);buckets.setdefault(k,[]).append(r)
 for k,vals in buckets.items():
  sp,ob,shots,model=k
  all_object.append({"split":sp,"object":ob,"shots":shots,"model":model,
   "n_repeats_seeds":len(vals),**{m:float(np.mean([v[m] for v in vals if v[m] is not None])) for m in ("mae","rmse","spearman")}})
 def paired(sp,shots,left,right):
  lookup={(r["object"],r["model"]):r["mae"] for r in all_object if r["split"]==sp and r["shots"]==shots}
  dif=[{"object":o,"ordinary_minus_reptile":lookup[(o,left)]-lookup[(o,right)]}
   for o in sorted({r["object"] for r in all_object if r["split"]==sp and r["shots"]==shots})
   if (o,left) in lookup and (o,right) in lookup]
  v=np.array([d["ordinary_minus_reptile"] for d in dif])
  return {"split":sp,"shots":shots,"left":left,"right":right,"positive_favors_reptile":True,
   "objects_improved":int(np.sum(v>0)),"n_objects":len(v),"mean_difference":float(v.mean()) if len(v) else None,"object_differences":dif}
 comparisons=[paired(sp,sh,"pooled_adapted","reptile_adapted") for sp in ("validation","test") for sh in (1,5,10)]
 write=lambda path,rows: (lambda f:(csv.DictWriter(f,fieldnames=list(rows[0])).writeheader(),csv.DictWriter(f,fieldnames=list(rows[0])).writerows(rows)))(path.open("w",newline="",encoding="utf-8"))
 write(out/"episode_metrics.csv",all_episode);write(out/"object_metrics.csv",all_object)
 (out/"paired_comparisons.json").write_text(json.dumps(comparisons,indent=2))
 (out/"environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\nnumpy={np.__version__}\ntorch={torch.__version__}\n")
 log(f"COMPLETE elapsed={time.time()-t0:.1f}s; total parameter updates per model/seed={a.iterations*a.meta_batch*a.inner_steps}")
 files=sorted(p for p in out.iterdir() if p.is_file() and p.name!="manifest.sha256")
 (out/"manifest.sha256").write_text("\n".join(f"{sha(p)}  {p.name}" for p in files)+"\n")
 print(json.dumps({"n_episode_rows":len(all_episode),"n_object_rows":len(all_object),"paired":comparisons},indent=2))
if __name__=="__main__":main()
