#!/usr/bin/env python3
"""Two-fold recording-level CV on official training recordings; official test is untouched."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from touch_research.classification.data import two_fold_recording_cv

def evaluate_recording_metrics(y_true, y_pred, recording_ids):
    rows = []
    for rid in sorted(np.unique(recording_ids).tolist()):
        mask = recording_ids == rid
        rows.append({
            "recording_id": int(rid),
            "n_windows": int(mask.sum()),
            "accuracy": float(accuracy_score(y_true[mask], y_pred[mask])),
            "macro_f1": float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)),
        })
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=Path("data/processed/early_classification.npz"))
    p.add_argument("--output",type=Path,default=Path("results/early_classification/recording_cv.json"))
    a=p.parse_args()
    if not a.data.is_file(): p.error(f"找不到 {a.data}；先运行 scripts/build_early_classification_data.py")
    with np.load(a.data,allow_pickle=False) as z: data={k:z[k] for k in z.files}
    x=data["force_vectors"]; y=data["object_id"]; rec=data["recording_id"]
    features=np.concatenate((x.mean(axis=1),x.std(axis=1)),axis=1)
    folds=two_fold_recording_cv(data)
    model_factories={
      "logistic_regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,class_weight="balanced")),
      "knn": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5,weights="distance"))
    }
    output={"protocol":"two_fold_class_stratified_recording_cv_on_official_train_only",
            "unit_of_split":"recording_id","official_test_used":False,
            "note":"仅用于训练集内部模型比较/诊断；每类仅两个训练 Recording，估计不稳定，不作为最终泛化性能结论。",
            "folds":[],"models":{}}
    for fold_id,fold in enumerate(folds, start=1):
        tr,va=fold["train"],fold["validation"]
        fold_row={"fold":fold_id,"n_train_windows":int(len(tr)),"n_validation_windows":int(len(va)),
                  "train_recordings":fold["train_recordings"],
                  "validation_recordings":fold["validation_recordings"],
                  "official_test_recordings":fold["official_test_recordings"],
                  "models":{}}
        for name,factory in model_factories.items():
            model=factory(); model.fit(features[tr],y[tr]); pred=model.predict(features[va])
            fold_row["models"][name]={
              "window_accuracy":float(accuracy_score(y[va],pred)),
              "window_macro_f1":float(f1_score(y[va],pred,average="macro",zero_division=0)),
              "recording_metrics":evaluate_recording_metrics(y[va],pred,rec[va]),
              "classification_report":classification_report(y[va],pred,output_dict=True,zero_division=0)
            }
        output["folds"].append(fold_row)
    for name in model_factories:
        acc=[f["models"][name]["window_accuracy"] for f in output["folds"]]
        f1=[f["models"][name]["window_macro_f1"] for f in output["folds"]]
        rec_acc=[r["accuracy"] for f in output["folds"] for r in f["models"][name]["recording_metrics"]]
        rec_f1=[r["macro_f1"] for f in output["folds"] for r in f["models"][name]["recording_metrics"]]
        output["models"][name]={
          "fold_window_accuracy_mean":float(np.mean(acc)),
          "fold_window_accuracy_sd":float(np.std(acc,ddof=1)),
          "fold_window_macro_f1_mean":float(np.mean(f1)),
          "fold_window_macro_f1_sd":float(np.std(f1,ddof=1)),
          "recording_accuracy_mean":float(np.mean(rec_acc)),
          "recording_accuracy_sd":float(np.std(rec_acc,ddof=1)),
          "recording_macro_f1_mean":float(np.mean(rec_f1)),
          "recording_macro_f1_sd":float(np.std(rec_f1,ddof=1))
        }
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    for name,metrics in output["models"].items():
        print(f"{name}: CV window accuracy={metrics['fold_window_accuracy_mean']:.4f} ± {metrics['fold_window_accuracy_sd']:.4f}; "
              f"recording accuracy={metrics['recording_accuracy_mean']:.4f} ± {metrics['recording_accuracy_sd']:.4f}")
    print(f"结果写入 {a.output}; 官方 test 完全未参与本次 CV。")
if __name__=="__main__": main()
