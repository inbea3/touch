#!/usr/bin/env python3
"""Stage 24B: persistence-residual Ridge/MLP benchmark on HAS."""
import argparse,csv,hashlib,json,platform,sys,time,copy
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import Ridge
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def metrics(y,p):
 e=p-y
 return {"mae":float(np.mean(np.abs(e))),"rmse":float(np.sqrt(np.mean(e*e)))}
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--data",default="results/has_window_dataset")
 ap.add_argument("--out",default="results/has_residual_stage24b")
 ap.add_argument("--seeds",default="20261126,20261127,20261128")
 ap.add_argument("--epochs",type=int,default=400)
 ap.add_argument("--patience",type=int,default=35)
 a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False);t0=time.time()
 def log(msg):
  line=f"[{datetime.now(timezone.utc).isoformat()}] {msg}";print(line,flush=True)
  with (out/"run.log").open("a") as f:f.write(line+"\n")
 z=np.load(Path(a.data)/"window_5.npz");X=z["X"].astype(np.float32);Y=z["Y_future_mean"].astype(np.float32)
 rows=list(csv.DictReader(open(Path(a.data)/"window_5_index.csv",encoding="utf-8")))
 if len(X)!=4192 or X.shape[1]!=168 or Y.shape[1]!=42 or len(rows)!=len(X):raise RuntimeError("Dataset shape/index mismatch")
 split=np.array([r["grouped_split"] for r in rows]);obj=np.array([r["object_group"] for r in rows])
 if len(set(obj[split=="train"])&set(obj[split=="validation"])) or len(set(obj[split=="train"])&set(obj[split=="test"])) or len(set(obj[split=="validation"])&set(obj[split=="test"])):raise RuntimeError("Object split leakage")
 tr=np.where(split=="train")[0];va=np.where(split=="validation")[0];te=np.where(split=="test")[0]
 # Persistence baseline is the last 42-D pressure vector, defined in the frozen feature schema.
 P=X[:,126:168].copy();R=Y-P
 # Standardize predictors and residual targets using Train only.
 xm=X[tr].mean(0);xs=X[tr].std(0);xs[xs<1e-8]=1
 rm=R[tr].mean(0);rs=R[tr].std(0);rs[rs<1e-8]=1
 Z=((X-xm)/xs).astype(np.float32);T=((R-rm)/rs).astype(np.float32)
 config={"stage":"24B","primary_window":5,"n_episodes":len(X),"input_dim":168,"target_dim":42,
 "residual_definition":"Y_future_mean - X_last_pressure_channels_126_168","train_only_scalers":True,
 "ridge_alphas":[0.01,0.1,1,10,100,1000],"mlp_architecture":[168,128,64,42],
 "mlp_seeds":[int(s) for s in a.seeds.split(",")],"epochs":a.epochs,"patience":a.patience,
 "split_objects":{s:sorted(set(obj[split==s])) for s in ("train","validation","test")},
 "test_status":"exploratory only; prior test inspection"}
 (out/"config.json").write_text(json.dumps(config,indent=2))
 def predict_metrics(ix,pred):
  outrows=[]
  for ob in sorted(set(obj[ix])):
   ii=ix[obj[ix]==ob];m=metrics(Y[ii],pred[ii]);outrows.append({"object":ob,"n_episodes":len(ii),**m})
  return {"object_macro_mae":float(np.mean([r["mae"] for r in outrows])),
   "object_macro_rmse":float(np.mean([r["rmse"] for r in outrows])),"per_object":outrows}
 results=[];predictions=[]
 # Persistence and direct train-mean references.
 for sp,ix in (("validation",va),("test",te)):
  for name,p in (("persistence",P),("train_mean",np.repeat(Y[tr].mean(0,keepdims=True),len(Y),axis=0))):
   mm=predict_metrics(ix,p);results.append({"split":sp,"model":name,"seed":"", "alpha":"",**{k:v for k,v in mm.items() if k!="per_object"}})
   for q in mm["per_object"]:predictions.append({"split":sp,"model":name,"seed":"","alpha":"",**q})
 # Ridge on residuals, alpha selected only on Validation macro-MAE.
 ridge_candidates=[]
 for alpha in config["ridge_alphas"]:
  model=Ridge(alpha=alpha);model.fit(Z[tr],T[tr])
  pred=P+model.predict(Z)*rs+rm
  for sp,ix in (("validation",va),("test",te)):
   mm=predict_metrics(ix,pred);ridge_candidates.append({"alpha":alpha,"split":sp,"prediction":pred,"metrics":mm})
   results.append({"split":sp,"model":"ridge_residual","seed":"","alpha":alpha,"object_macro_mae":mm["object_macro_mae"],"object_macro_rmse":mm["object_macro_rmse"]})
   for q in mm["per_object"]:predictions.append({"split":sp,"model":"ridge_residual","seed":"","alpha":alpha,**q})
 best=min((x for x in ridge_candidates if x["split"]=="validation"),key=lambda x:x["metrics"]["object_macro_mae"])
 (out/"ridge_selection.json").write_text(json.dumps({"selected_alpha":best["alpha"],"criterion":"lowest Validation object-macro MAE","validation_object_macro_mae":best["metrics"]["object_macro_mae"]},indent=2))
 # MLP residual and direct-target MLP use identical architecture/optimizer budget; validation early stopping.
 xt=torch.from_numpy(Z);rt=torch.from_numpy(T)
 ymean=Y[tr].mean(0);ystd=Y[tr].std(0);ystd[ystd<1e-8]=1
 yt=torch.from_numpy(((Y-ymean)/ystd).astype(np.float32))
 for seed in config["mlp_seeds"]:
  torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(2)
  for target_kind in ("residual","direct"):
   target=rt if target_kind=="residual" else yt
   net=nn.Sequential(nn.Linear(168,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,42))
   opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-3)
   best_state=None;best_val=float("inf");bad=0;trace=[]
   for epoch in range(1,a.epochs+1):
    net.train();perm=np.random.permutation(tr)
    losses=[]
    for start in range(0,len(perm),32):
     ix=perm[start:start+32];opt.zero_grad();loss=nn.functional.mse_loss(net(xt[ix]),target[ix]);loss.backward();opt.step();losses.append(float(loss.detach()))
    net.eval()
    with torch.no_grad():
     raw=net(xt[va]).numpy()
     vp=(P[va]+raw*rs+rm) if target_kind=="residual" else raw*ystd+ymean
    val=predict_metrics(va,np.concatenate([P[:0],vp],axis=0))["object_macro_mae"] if False else float(np.mean([metrics(Y[va][obj[va]==o],vp[obj[va]==o])["mae"] for o in sorted(set(obj[va]))]))
    trace.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"validation_object_macro_mae":val})
    if val<best_val-1e-8:
     best_val=val;best_state=copy.deepcopy(net.state_dict());bad=0
    else:bad+=1
    if bad>=a.patience:break
   net.load_state_dict(best_state);net.eval()
   torch.save(net.state_dict(),out/f"{target_kind}_mlp_seed{seed}.pt")
   with (out/f"{target_kind}_mlp_seed{seed}_trace.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(trace[0]));w.writeheader();w.writerows(trace)
   with torch.no_grad():raw=net(xt).numpy()
   pred=(P+raw*rs+rm) if target_kind=="residual" else raw*ystd+ymean
   for sp,ix in (("validation",va),("test",te)):
    mm=predict_metrics(ix,pred);results.append({"split":sp,"model":target_kind+"_mlp","seed":seed,"alpha":"","object_macro_mae":mm["object_macro_mae"],"object_macro_rmse":mm["object_macro_rmse"]})
    for q in mm["per_object"]:predictions.append({"split":sp,"model":target_kind+"_mlp","seed":seed,"alpha":"",**q})
   log(f"seed={seed} model={target_kind}_mlp best_val={best_val:.6f} epochs={len(trace)}")
 # Summarize Validation-selected method; test is descriptive only.
 with (out/"model_metrics.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
 with (out/"object_metrics.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=list(predictions[0]));w.writeheader();w.writerows(predictions)
 summary={"n_episodes":len(X),"train_validation_test_episodes":[len(tr),len(va),len(te)],
 "train_only_scaling":True,"ridge_selected_alpha":best["alpha"],
 "validation_model_metrics":[r for r in results if r["split"]=="validation"],
 "test_model_metrics_exploratory":[r for r in results if r["split"]=="test"],
 "interpretation":"Select candidate using Validation object-macro MAE only. Test metrics are exploratory and must not drive model selection.",
 "elapsed_seconds":time.time()-t0}
 (out/"summary.json").write_text(json.dumps(summary,indent=2))
 (out/"environment.txt").write_text(f"python={sys.version}\nplatform={platform.platform()}\nnumpy={np.__version__}\ntorch={torch.__version__}\n")
 log("COMPLETE")
 files=sorted(p for p in out.iterdir() if p.is_file() and p.name!="manifest.sha256")
 (out/"manifest.sha256").write_text("\n".join(f"{sha(p)}  {p.name}" for p in files)+"\n")
 print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
