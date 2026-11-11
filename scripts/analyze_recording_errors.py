#!/usr/bin/env python3
"""Recording-level error analysis from two-fold OOF predictions on official train recordings."""
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from touch_research.classification.data import two_fold_recording_cv

def object_names(path):
    if not path.is_file(): return {}
    from scipy.io import loadmat
    raw=np.asarray(loadmat(path,simplify_cells=True).get("objects",[])).reshape(-1)
    return {i:(x.decode("utf-8",errors="replace") if isinstance(x,bytes) else str(x)) for i,x in enumerate(raw)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=Path("data/processed/early_classification.npz"))
    p.add_argument("--metadata",type=Path,default=Path("data/interim/metadata.mat"))
    p.add_argument("--output-dir",type=Path,default=Path("results/early_classification/error_analysis"))
    a=p.parse_args()
    if not a.data.is_file(): p.error(f"找不到 {a.data}；先运行 scripts/build_early_classification_data.py")
    with np.load(a.data,allow_pickle=False) as z: data={k:z[k] for k in z.files}
    y,rec,split=data["object_id"],data["recording_id"],data["split_id"]
    x=data["force_vectors"]; feat=np.concatenate((x.mean(axis=1),x.std(axis=1)),axis=1)
    folds=two_fold_recording_cv(data); train_idx=np.flatnonzero(split==0); classes=np.unique(y[train_idx])
    names=object_names(a.metadata)
    label=lambda c:names.get(int(c),f"class_{int(c)}")
    factories={
      "logistic_regression":lambda:make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced")),
      "knn":lambda:make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=5,weights="distance"))}
    oof={name:np.full(y.shape,-1,dtype=np.int64) for name in factories}; fold_rows=[]
    for k,fold in enumerate(folds,1):
        tr,va=fold["train"],fold["validation"]; row={"fold":k,"n_train_windows":int(len(tr)),
          "n_validation_windows":int(len(va)),"n_train_recordings":len(fold["train_recordings"]),
          "n_validation_recordings":len(fold["validation_recordings"])}
        for name,factory in factories.items():
            model=factory(); model.fit(feat[tr],y[tr]); pred=model.predict(feat[va]); oof[name][va]=pred
            row[name]={"accuracy":float(accuracy_score(y[va],pred)),
                       "macro_f1":float(f1_score(y[va],pred,average="macro",zero_division=0))}
        fold_rows.append(row)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    class_rows=[]; recording_rows=[]; summaries={}; difficulty={}; cm_paths={}
    for name,pred_all in oof.items():
        pred,truth,rr=pred_all[train_idx],y[train_idx],rec[train_idx]
        precision,recall,f1,support=precision_recall_fscore_support(truth,pred,labels=classes,zero_division=0)
        for rid in sorted(np.unique(rr).tolist()):
            m=rr==rid; cls=int(truth[m][0])
            recording_rows.append({"model":name,"recording_id":int(rid),"object_id":cls,
              "object_name":label(cls),"n_windows":int(m.sum()),
              "accuracy":float(accuracy_score(truth[m],pred[m])),
              "correct_windows":int(np.sum(truth[m]==pred[m]))})
        for i,cls0 in enumerate(classes):
            cls=int(cls0); rs=[r["accuracy"] for r in recording_rows if r["model"]==name and r["object_id"]==cls]
            class_rows.append({"model":name,"object_id":cls,"object_name":label(cls),
              "n_recordings":len(rs),"n_windows":int(support[i]),"precision":float(precision[i]),
              "recall":float(recall[i]),"f1":float(f1[i]),
              "recording_accuracy_mean":float(np.mean(rs)),
              "recording_accuracy_sd":float(np.std(rs,ddof=1)) if len(rs)>1 else None})
        cm=confusion_matrix(truth,pred,labels=classes)
        cm_df=pd.DataFrame(cm,index=[f"{int(c)}_{label(c)}" for c in classes],
                           columns=[f"{int(c)}_{label(c)}" for c in classes])
        cp=a.output_dir/f"{name}_confusion_matrix.csv"; cm_df.to_csv(cp,encoding="utf-8-sig"); cm_paths[name]=str(cp)
        summaries[name]={"oof_window_accuracy":float(accuracy_score(truth,pred)),
          "oof_balanced_accuracy":float(balanced_accuracy_score(truth,pred)),
          "oof_macro_f1":float(f1_score(truth,pred,labels=classes,average="macro",zero_division=0)),
          "oof_weighted_f1":float(f1_score(truth,pred,average="weighted",zero_division=0)),
          "n_oof_windows":int(len(truth)),"n_oof_recordings":int(len(np.unique(rr))),
          "classification_report":classification_report(truth,pred,labels=classes,output_dict=True,zero_division=0)}
        difficulty[name]=sorted([r for r in class_rows if r["model"]==name],key=lambda r:(r["recall"],r["f1"]))
    class_path=a.output_dir/"per_class_metrics.csv"; recording_path=a.output_dir/"per_recording_metrics.csv"
    pd.DataFrame(class_rows).to_csv(class_path,index=False,encoding="utf-8-sig")
    pd.DataFrame(recording_rows).to_csv(recording_path,index=False,encoding="utf-8-sig")
    result={"protocol":"pooled_two_fold_oof_on_official_train_recordings","official_test_used":False,
      "unit_of_split":"recording_id","n_classes":int(len(classes)),"n_oof_windows":int(len(train_idx)),
      "n_oof_recordings":int(len(np.unique(rec[train_idx]))),"folds":fold_rows,"models":summaries,
      "hardest_classes_by_lowest_recall":difficulty,
      "output_files":{"per_class_metrics":str(class_path),"per_recording_metrics":str(recording_path),"confusion_matrices":cm_paths},
      "limitations":["每类仅两个官方训练 Recording，OOF 估计对 Recording 身份敏感。",
        "同一 Recording 内窗口存在时间相关性，不能作为独立重复样本。",
        "官方测试 Recording 未参与本误差分析，仍须保留作最终一次评估。",
        "单个 Recording 只有一个真实类别，因此逐 Recording Accuracy 是描述性指标，不是独立总体推断。"]}
    jp=a.output_dir/"error_analysis.json"; jp.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    for name,m in summaries.items():
        hard=difficulty[name][:5]
        print(f"{name}: OOF accuracy={m['oof_window_accuracy']:.4f}, balanced accuracy={m['oof_balanced_accuracy']:.4f}, macro-F1={m['oof_macro_f1']:.4f}")
        print("  最难类别: "+", ".join(f"{r['object_name']} (Recall={r['recall']:.3f})" for r in hard))
    print(f"报告目录: {a.output_dir}")
if __name__=="__main__": main()
