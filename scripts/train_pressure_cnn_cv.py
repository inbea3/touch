#!/usr/bin/env python3
"""Pressure-map CNN with recording-level two-fold CV on official train recordings only."""
import argparse, json, random, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from touch_research.classification.data import build_window_dataset, two_fold_recording_cv
from touch_research.classification.models import SpatialPressureCNN
from scipy.io import loadmat

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(4)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata",type=Path,default=Path("data/interim/metadata.mat"))
    p.add_argument("--output",type=Path,default=Path("results/early_classification/pressure_cnn_cv.json"))
    p.add_argument("--epochs",type=int,default=15)
    a=p.parse_args()
    if not a.metadata.is_file(): p.error(f"找不到 {a.metadata}")
    arrays,_=build_window_dataset(loadmat(a.metadata,simplify_cells=True),k_frames=5,stride=10)
    classes=np.unique(arrays["object_id"])
    mapping={int(v):i for i,v in enumerate(classes)}
    y=np.asarray([mapping[int(v)] for v in arrays["object_id"]],dtype=np.int64)
    X=arrays["pressure_maps"].astype(np.float32)
    folds=two_fold_recording_cv(arrays)
    out={
      "protocol":"two_fold_recording_cv_official_train_only",
      "official_test_used":False,
      "input":"K=5 pressure maps, 32x32; force=(1024-pressure)/524 without clipping; per-pixel z-score fitted on fold training recordings only and clipped to [-5,5]",
      "architecture":"SpatialPressureCNN: conv channels 8/16/16, two 2x2 max-pools, spatial flatten, dense 64, dropout 0.5",
      "optimizer":"AdamW(lr=0.0005, weight_decay=0.001)",
      "batch_size":256,"max_epochs":a.epochs,"early_stopping_patience":6,
      "class_loss_weights":"square-root inverse-frequency weights calculated from fold training windows",
      "folds":[]}
    for fi,fold in enumerate(folds,1):
        seed_all(2026+fi)
        tr,va=fold["train"],fold["validation"]
        yv=y[va]
        # All pixel statistics are estimated from the current fold's training data.
        pix_mean=X[tr].mean(axis=(0,1),keepdims=True)
        pix_std=np.maximum(X[tr].std(axis=(0,1),keepdims=True),1e-4)
        X_fold=np.clip((X-pix_mean)/pix_std,-5,5).astype(np.float32)
        rng=np.random.default_rng(2026+fi)
        model=SpatialPressureCNN(len(classes),k_frames=5)
        counts=np.bincount(y[tr],minlength=len(classes))
        weights=torch.tensor(np.sqrt(len(tr)/(len(classes)*np.maximum(counts,1))),dtype=torch.float32)
        loss_fn=nn.CrossEntropyLoss(weight=weights)
        opt=torch.optim.AdamW(model.parameters(),lr=0.0005,weight_decay=0.001)
        best=-1.0; best_state=None; patience=0; history=[]
        for epoch in range(1,a.epochs+1):
            model.train(); losses=[]
            order=rng.permutation(len(tr))
            for start in range(0,len(order),256):
                ids=tr[order[start:start+256]]
                xb=torch.from_numpy(X_fold[ids]); yb=torch.from_numpy(y[ids])
                opt.zero_grad(set_to_none=True)
                loss=loss_fn(model(xb),yb); loss.backward(); opt.step()
                losses.append(float(loss.item()))
            model.eval(); val_parts=[]
            with torch.no_grad():
                for start in range(0,len(va),256):
                    ids=va[start:start+256]
                    val_parts.append(model(torch.from_numpy(X_fold[ids])).argmax(1).numpy())
            val_pred=np.concatenate(val_parts)
            val_macro=f1_score(yv,val_pred,average="macro",zero_division=0)
            history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),
                            "validation_macro_f1":float(val_macro)})
            if val_macro>best+1e-6:
                best=val_macro
                best_state={k:v.detach().clone() for k,v in model.state_dict().items()}
                patience=0
            else:
                patience+=1
                if patience>=6: break
        model.load_state_dict(best_state); model.eval(); pred_parts=[]
        with torch.no_grad():
            for start in range(0,len(va),256):
                ids=va[start:start+256]
                pred_parts.append(model(torch.from_numpy(X_fold[ids])).argmax(1).numpy())
        pred=np.concatenate(pred_parts)
        rec=arrays["recording_id"][va]; rec_rows=[]
        for rid in sorted(np.unique(rec).tolist()):
            mask=rec==rid
            rec_rows.append({"recording_id":int(rid),"n_windows":int(mask.sum()),
                             "accuracy":float(accuracy_score(yv[mask],pred[mask]))})
        row={
          "fold":fi,"n_train_windows":int(len(tr)),"n_validation_windows":int(len(va)),
          "train_recordings":fold["train_recordings"],
          "validation_recordings":fold["validation_recordings"],
          "best_epoch":int(np.argmax([h["validation_macro_f1"] for h in history])+1),
          "epochs_run":len(history),"validation_macro_f1_at_selection":float(best),
          "accuracy":float(accuracy_score(yv,pred)),
          "balanced_accuracy":float(balanced_accuracy_score(yv,pred)),
          "macro_f1":float(f1_score(yv,pred,average="macro",zero_division=0)),
          "weighted_f1":float(f1_score(yv,pred,average="weighted",zero_division=0)),
          "recording_metrics":rec_rows,
          "classification_report":classification_report(yv,pred,output_dict=True,zero_division=0),
          "history":history}
        out["folds"].append(row)
        print(f"fold {fi}: accuracy={row['accuracy']:.4f}, macro-F1={row['macro_f1']:.4f}, best_epoch={row['best_epoch']}")
    out["summary"]={}
    for metric in ("accuracy","balanced_accuracy","macro_f1","weighted_f1"):
        vals=[f[metric] for f in out["folds"]]
        out["summary"][metric+"_mean"]=float(np.mean(vals))
        out["summary"][metric+"_fold_sd"]=float(np.std(vals,ddof=1))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"结果写入 {a.output}; 官方 test 未参与模型训练、早停或选择。")

if __name__=="__main__": main()
