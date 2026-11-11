#!/usr/bin/env python3
"""Fixed Logistic Regression/KNN baselines using the official recording-level split."""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from touch_research.classification.data import split_by_official_recording

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=Path("data/processed/early_classification.npz"))
    p.add_argument("--output",type=Path,default=Path("results/early_classification/baselines.json"))
    a=p.parse_args()
    if not a.data.is_file(): p.error(f"找不到 {a.data}；先运行 scripts/build_early_classification_data.py")
    with np.load(a.data,allow_pickle=False) as z: data={k:z[k] for k in z.files}
    split=split_by_official_recording(data); x=data["force_vectors"]
    feat=np.concatenate((x.mean(axis=1),x.std(axis=1)),axis=1); y=data["object_id"]
    tr,te=split["train"],split["test"]
    if not tr.size or not te.size: raise RuntimeError("官方 train/test 样本不足")
    if set(np.unique(y[tr])) != set(np.unique(y[te])):
        raise RuntimeError("官方 train/test 类别不一致，需先审查类别覆盖，不能静默评估")
    result={"protocol":"official_splitId_recording_disjoint","n_train":int(tr.size),
            "n_test":int(te.size),"recordings_per_class":split["recordings_per_class"],
            "models":{},"selection_note":"固定超参数；未使用测试集调参或选择模型"}
    models={
      "logistic_regression":make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight="balanced")),
      "knn":make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=min(5,tr.size),weights="distance"))}
    for name,model in models.items():
        model.fit(feat[tr],y[tr]); pred=model.predict(feat[te])
        result["models"][name]={"accuracy":float(accuracy_score(y[te],pred)),
          "macro_f1":float(f1_score(y[te],pred,average="macro",zero_division=0)),
          "classification_report":classification_report(y[te],pred,output_dict=True,zero_division=0)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    for name,m in result["models"].items(): print(f"{name}: accuracy={m['accuracy']:.4f}, macro_f1={m['macro_f1']:.4f}")
    print(f"报告：{a.output}；固定参数，仅使用官方 test 做最终一次评估。")
if __name__ == "__main__": main()
