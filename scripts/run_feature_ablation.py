#!/usr/bin/env python3
"""Feature ablations with two-fold Recording-level CV; official test remains sealed."""
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import numpy as np
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from touch_research.classification.data import build_window_dataset, two_fold_recording_cv

def metrics(y,p):
    return {"accuracy":float(accuracy_score(y,p)),
            "balanced_accuracy":float(balanced_accuracy_score(y,p)),
            "macro_f1":float(f1_score(y,p,average="macro",zero_division=0)),
            "weighted_f1":float(f1_score(y,p,average="weighted",zero_division=0))}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata",type=Path,default=Path("data/interim/metadata.mat"))
    p.add_argument("--output",type=Path,default=Path("results/early_classification/feature_ablation.json"))
    a=p.parse_args()
    if not a.metadata.is_file(): p.error(f"找不到 {a.metadata}")
    arrays,_=build_window_dataset(loadmat(a.metadata,simplify_cells=True),k_frames=5,stride=10)
    X=arrays["pressure_maps"].astype(np.float32); y=arrays["object_id"]
    folds=two_fold_recording_cv(arrays)
    sensor=arrays["sensor_mask"].reshape(-1)
    flat=X.reshape(len(X),5,-1)
    mean=flat.mean(axis=1)[:,sensor]; sd=flat.std(axis=1)[:,sensor]
    reps={
      "K1_last_frame":flat[:,-1][:,sensor],
      "K5_mean_only":mean,
      "K5_mean_plus_sd":np.concatenate([mean,sd],axis=1),
      "K5_spatial_mean_plus_sd":np.concatenate([X.mean(axis=1).reshape(len(X),-1),
                                                  X.std(axis=1).reshape(len(X),-1)],axis=1)
    }
    experiments=[]
    for rep_name,features in reps.items():
      for model_name,kind in [("KNN_k5_standardized","knn_scaled"),
                              ("MLP_64","mlp")]:
        for fold_id,(tr,va) in enumerate(folds,1):
          if kind=="knn_scaled":
            model=make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=5,weights="distance"))
          else:
            model=make_pipeline(StandardScaler(),MLPClassifier(
              hidden_layer_sizes=(64,),alpha=.01,learning_rate_init=.001,
              max_iter=250,early_stopping=True,n_iter_no_change=15,random_state=2027))
          model.fit(features[tr],y[tr]); pred=model.predict(features[va])
          experiments.append({"representation":rep_name,"model":model_name,"fold":fold_id,
                              "n_train_windows":int(len(tr)),"n_validation_windows":int(len(va)),
                              **metrics(y[va],pred)})
    # Explicit scaling ablation on the reference K=5 mean+SD representation.
    for fold_id,(tr,va) in enumerate(folds,1):
      for scaled in (True,False):
        estimator=KNeighborsClassifier(n_neighbors=5,weights="distance")
        model=make_pipeline(StandardScaler(),estimator) if scaled else estimator
        model.fit(reps["K5_mean_plus_sd"][tr],y[tr])
        pred=model.predict(reps["K5_mean_plus_sd"][va])
        experiments.append({"representation":"K5_mean_plus_sd",
          "model":"KNN_standardized" if scaled else "KNN_unscaled","fold":fold_id,
          "n_train_windows":int(len(tr)),"n_validation_windows":int(len(va)),
          **metrics(y[va],pred)})
    grouped={}
    for row in experiments: grouped.setdefault((row["representation"],row["model"]),[]).append(row)
    summary=[]
    for (rep,model),rows in grouped.items():
      item={"representation":rep,"model":model}
      for key in ("accuracy","balanced_accuracy","macro_f1","weighted_f1"):
        vals=[r[key] for r in rows]; item[key+"_mean"]=float(np.mean(vals))
      for key in ("accuracy","macro_f1"):
        item[key+"_fold_sd"]=float(np.std([r[key] for r in rows],ddof=1))
      summary.append(item)
    out={"protocol":"two_fold_recording_level_cv_on_official_train_only",
      "official_test_used":False,"n_classes":int(len(np.unique(y))),
      "n_windows":int(len(y)),"n_valid_sensors":int(sensor.sum()),
      "folds":[{"fold":i+1,"train_recordings":len(f["train_recordings"]),
        "validation_recordings":len(f["validation_recordings"]),
        "train_windows":len(f["train"]),"validation_windows":len(f["validation"])}
        for i,f in enumerate(folds)],
      "results":experiments,"summary":summary,
      "limitations":[
        "每类只有两个官方训练 Recording；两折结果仅作描述性探索，不构成独立重复样本推断。",
        "K=1 是同一 K=5 窗口末帧，不是另行采集的一帧任务。",
        "KNN 与 MLP 均在每折训练数据上拟合 StandardScaler。",
        "空间展平统计特征与 548 维有效传感器统计特征仅是补零位置差异，数值上等价；它不单独证明 CNN 的局部空间建模价值。",
        "所有模型参数预先固定；未使用官方 test。"]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
    for r in sorted(summary,key=lambda x:(x["model"],x["representation"])):
      print(f'{r["representation"]} | {r["model"]}: Acc={r["accuracy_mean"]:.4f}, Macro-F1={r["macro_f1_mean"]:.4f}')
    print(f"结果写入 {a.output}; 官方测试集未使用。")
if __name__=="__main__": main()
