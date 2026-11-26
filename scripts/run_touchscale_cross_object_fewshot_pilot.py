#!/usr/bin/env python3
"""TouchScale exploratory cross-object few-shot adaptation pilot.
Secret injection is handled by the workflow environment (no token literal in source).
Run manifest is frozen to the reviewed 150-row candidate sample; no full dataset download.

Uses 27 metadata-reviewed candidate episodes across seven normalized target
categories. One example is one complete interaction episode. Input is early
continuous 15-pad pressure summary; target is the later-window 15-pad
pairwise Pearson association vector (105 upper-triangle entries).

Evaluation is leave-one-category-out (LOCO). Reptile and pooled MLP receive
the same number of training gradient updates. Support and Query are split by
whole episode. This is an exploratory pilot, not a confirmatory result:
several categories have very few episodes and physical instance IDs are absent.
"""
from __future__ import annotations
import copy, io, json, os, random, tarfile, time, hashlib
from collections import defaultdict
from pathlib import Path

import fsspec
import h5py
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_url
from sklearn.preprocessing import StandardScaler

REPO = "2077AIDataFoundation/TouchScale"
OUT = Path("artifacts/touchscale_cross_object_fewshot_pilot")
SEED_LIST = [17, 29, 43]
META_ITERATIONS = 160
META_BATCH = 4
INNER_STEPS = 5
INNER_LR = 1e-3
OUTER_STEP = 0.1
ADAPT_STEPS = 20
BATCH_SIZE = 32
N_PADS = 15
EARLY_FRAC = 0.40
LATE_START = 0.60

# High/medium-confidence candidate rows from the 150-Episode normalization sample.
# Each row is traceable to the retained metadata audit; no low-confidence row is used.
EPISODES = [
 {"id":"bf9023ff-bc6e-4dc1-a742-216957a8800d","actor":"actor_01","date":"2026-08-27","label":"A4 paper sheet set","shard":"data/train-00092-of-00430.tar"},
 {"id":"b9abf17b-dac4-4b82-90d3-637a2c88b65b","actor":"actor_09","date":"2026-09-14","label":"A4 paper sheet set","shard":"data/train-00411-of-00430.tar"},
 {"id":"17c1e6b9-c265-444d-a1c2-767c5343ad15","actor":"actor_09","date":"2026-09-14","label":"A4 paper sheet set","shard":"data/train-00324-of-00430.tar"},
 {"id":"a88f2927-4f9f-4d17-9fde-376a0ad55b13","actor":"actor_04","date":"2026-08-27","label":"container closure (lid/cap)","shard":"data/train-00159-of-00430.tar"},
 {"id":"73106a61-1e8f-48a1-b7b6-43f98fe8a643","actor":"actor_04","date":"2026-08-21","label":"container closure (lid/cap)","shard":"data/train-00202-of-00430.tar"},
 {"id":"37958245-c507-4b1c-8461-1d4de959cf0c","actor":"actor_08","date":"2026-07-31","label":"container closure (lid/cap)","shard":"data/train-00101-of-00430.tar"},
 {"id":"af4a5203-5da2-42d0-8ccc-8b67d94541a8","actor":"actor_08","date":"2026-07-31","label":"container closure (lid/cap)","shard":"data/train-00387-of-00430.tar"},
 {"id":"eaba8c49-3c5a-4da6-a29c-989f1dfcf879","actor":"actor_06","date":"2026-09-09","label":"hex nut","shard":"data/train-00150-of-00430.tar"},
 {"id":"9ebb1e82-5688-4fd2-ba2e-8465db4ee64a","actor":"actor_07","date":"2026-07-31","label":"hex nut","shard":"data/train-00015-of-00430.tar"},
 {"id":"91e20a35-f0c9-4c62-8424-996714132d84","actor":"actor_07","date":"2026-07-31","label":"hex nut","shard":"data/train-00127-of-00430.tar"},
 {"id":"9ad26515-1577-49d4-9eae-b4e1c7441a61","actor":"actor_14","date":"2026-09-22","label":"hex nut","shard":"data/train-00289-of-00430.tar"},
 {"id":"f823586d-91f5-409d-8843-27fe84f8d788","actor":"actor_14","date":"2026-09-22","label":"hex nut","shard":"data/train-00087-of-00430.tar"},
 {"id":"5ef3045b-c63e-4847-8e69-64affdc03e5a","actor":"actor_01","date":"2026-08-10","label":"lever switch","shard":"data/train-00254-of-00430.tar"},
 {"id":"3e597535-8e08-4c8b-a730-a97168fbc102","actor":"actor_23","date":"2026-08-03","label":"lever switch","shard":"data/train-00016-of-00430.tar"},
 {"id":"2cb90ef0-80dd-40ef-8566-8f48db57b4dc","actor":"actor_01","date":"2026-08-21","label":"lid / latch component","shard":"data/train-00244-of-00430.tar"},
 {"id":"3b7d4145-052a-48e6-bb11-5339be5e8237","actor":"actor_02","date":"2026-08-07","label":"lid / latch component","shard":"data/train-00049-of-00430.tar"},
 {"id":"7e6e0b97-9972-44e8-ad04-470e4cc4f09e","actor":"actor_02","date":"2026-08-24","label":"lid / latch component","shard":"data/train-00334-of-00430.tar"},
 {"id":"acc872ea-1453-4f0b-a12b-9d26ac782e0b","actor":"actor_03","date":"2026-08-24","label":"lid / latch component","shard":"data/train-00304-of-00430.tar"},
 {"id":"2e415736-052a-48e6-bb11-5339be5e8237","actor":"actor_05","date":"2026-09-04","label":"lid / latch component","shard":"data/train-00376-of-00430.tar"},
 {"id":"45e75a86-4b7b-4879-9ef9-69a3a7d98d02","actor":"actor_21","date":"2026-09-15","label":"lid / latch component","shard":"data/train-00335-of-00430.tar"},
 {"id":"7202416f-8edd-4666-8836-9533937b56ba","actor":"actor_07","date":"2026-09-04","label":"shipping label","shard":"data/train-00403-of-00430.tar"},
 {"id":"f3695b1a-94b6-4aa5-861b-3cefb3de76da","actor":"actor_11","date":"2026-09-16","label":"shipping label","shard":"data/train-00259-of-00430.tar"},
 {"id":"968f7e5b-40bc-4c5f-969c-a94bea6671db","actor":"actor_11","date":"2026-09-16","label":"shipping label","shard":"data/train-00020-of-00430.tar"},
 {"id":"c7338d17-aec5-42fd-aee5-57a7e6448a9f","actor":"actor_11","date":"2026-09-15","label":"shipping label","shard":"data/train-00344-of-00430.tar"},
 {"id":"6ba348ca-d1b8-4ae8-886b-d3a06089bd9d","actor":"actor_03","date":"2026-09-04","label":"zipper / zipper pull","shard":"data/train-00190-of-00430.tar"},
 {"id":"7de30e25-7b5c-415e-aabd-bd82ef5403c8","actor":"actor_09","date":"2026-09-20","label":"zipper / zipper pull","shard":"data/train-00151-of-00430.tar"},
]

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(1)

def corr_target(x):
    c = np.corrcoef(x, rowvar=False)
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    iu = np.triu_indices(c.shape[0], k=1)
    return np.clip(c[iu], -1.0, 1.0).astype(np.float32)

def episode_xy(h5, meta):
    hand_candidates = []
    for hand in ("left", "right"):
        key = f"obs/tactile/raw/{hand}"
        if key not in h5:
            continue
        g = h5[key]
        pads = sorted([k for k in g if k.startswith("pad_")],
                      key=lambda k: int(k.split("_")[1]))
        if len(pads) != N_PADS:
            continue
        arrays = [np.asarray(g[k][()], dtype=np.float32) for k in pads]
        lengths = {a.shape[0] for a in arrays}
        if len(lengths) != 1:
            continue
        t = lengths.pop()
        if t < 80:
            continue
        loads = np.column_stack([np.maximum(a.reshape(t, -1), 0).sum(axis=1) for a in arrays])
        if not np.isfinite(loads).all():
            continue
        if "t_epoch" not in g:
            continue
        ts = np.asarray(g["t_epoch"][()], dtype=np.float64).reshape(-1)
        if len(ts) != t or not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
            continue
        early_end = int(np.floor(t * EARLY_FRAC))
        late_start = int(np.floor(t * LATE_START))
        early, late = loads[:early_end], loads[late_start:]
        if len(early) < 20 or len(late) < 20:
            continue
        # Hand selection uses only the early input window, never the future target.
        early_activity = float(np.mean(early.sum(axis=1)))
        hand_candidates.append((early_activity, hand, early, late, t, float(ts[-1]-ts[0])))
    if not hand_candidates:
        raise ValueError("no_valid_hand_or_timebase")
    _, hand, early, late, t, duration = max(hand_candidates, key=lambda z: z[0])
    if float(np.mean(early.sum(axis=1))) <= 1e-6:
        raise ValueError("no_early_pressure_signal")
    # Three groups of early-window features, all continuous and threshold-free.
    x = np.concatenate([
        np.log1p(early.mean(axis=0)),
        np.log1p(np.quantile(early, 0.95, axis=0)),
        np.log1p(early.std(axis=0)),
    ]).astype(np.float32)
    y = corr_target(late)
    audit = {
        **meta, "selected_hand": hand, "frames": int(t), "duration_s": duration,
        "early_mean_total_load": float(early.sum(axis=1).mean()),
        "early_active_fraction_0p1N": float((early.sum(axis=1) > 0.1).mean()),
        "early_active_fraction_3N": float((early.sum(axis=1) > 3.0).mean()),
        "late_mean_total_load": float(late.sum(axis=1).mean()),
        "target_edges": int(len(y)), "status": "ok"
    }
    return x, y, audit

class SmallMLP(torch.nn.Module):
    def __init__(self, nx, ny):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(nx, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, ny)
        )
    def forward(self, x): return self.net(x)

def state_copy(m):
    return {k:v.detach().clone() for k,v in m.state_dict().items()}

def reptile_fit(train, n_features, n_targets, seed):
    seed_all(seed)
    rng = np.random.default_rng(seed + 1001)
    tasks = sorted(train)
    model = SmallMLP(n_features, n_targets)
    # Fixed scaler is fitted only on outer-training object categories.
    for _ in range(META_ITERATIONS):
        chosen = rng.choice(tasks, size=min(META_BATCH, len(tasks)), replace=False)
        base = state_copy(model)
        adapted_states = []
        for label in chosen:
            candidates = train[label]
            inner = SmallMLP(n_features, n_targets)
            inner.load_state_dict(base)
            opt = torch.optim.Adam(inner.parameters(), lr=INNER_LR)
            for step in range(INNER_STEPS):
                ex = candidates[int(rng.integers(0, len(candidates)))]
                xb = torch.tensor(ex["x"], dtype=torch.float32).unsqueeze(0)
                yb = torch.tensor(ex["y"], dtype=torch.float32).unsqueeze(0)
                loss = torch.mean((inner(xb)-yb)**2)
                opt.zero_grad(); loss.backward(); opt.step()
            adapted_states.append(state_copy(inner))
        with torch.no_grad():
            for name, param in model.named_parameters():
                delta = torch.stack([s[name]-base[name] for s in adapted_states]).mean(0)
                param.add_(delta, alpha=OUTER_STEP)
    return model

def pooled_fit(train, n_features, n_targets, seed, total_steps):
    seed_all(seed + 5)
    rng = np.random.default_rng(seed + 2003)
    examples = [e for arr in train.values() for e in arr]
    model = SmallMLP(n_features, n_targets)
    opt = torch.optim.Adam(model.parameters(), lr=INNER_LR)
    for _ in range(total_steps):
        ex = examples[int(rng.integers(0, len(examples)))]
        xb = torch.tensor(ex["x"], dtype=torch.float32).unsqueeze(0)
        yb = torch.tensor(ex["y"], dtype=torch.float32).unsqueeze(0)
        loss = torch.mean((model(xb)-yb)**2)
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def adapt_and_score(model, support, query, seed, target_mu, target_sd):
    adapted = copy.deepcopy(model)
    opt = torch.optim.Adam(adapted.parameters(), lr=5e-4)
    rng = np.random.default_rng(seed + 77)
    for _ in range(ADAPT_STEPS):
        ex = support[int(rng.integers(0, len(support)))]
        xb = torch.tensor(ex["x"], dtype=torch.float32).unsqueeze(0)
        yb = torch.tensor(ex["y"], dtype=torch.float32).unsqueeze(0)
        loss = torch.mean((adapted(xb)-yb)**2)
        opt.zero_grad(); loss.backward(); opt.step()
    preds=[]; targets=[]
    with torch.no_grad():
        for ex in query:
            x=torch.tensor(ex["x"],dtype=torch.float32).unsqueeze(0)
            preds.append(adapted(x).numpy()[0])
            targets.append(ex["y"])
    p=np.asarray(preds)*target_sd+target_mu
    y=np.asarray(targets)*target_sd+target_mu
    return float(np.mean(np.abs(p-y))), float(np.sqrt(np.mean((p-y)**2)))

def score_zero(model, query, target_mu, target_sd):
    preds=[]; targets=[]
    with torch.no_grad():
        for ex in query:
            x=torch.tensor(ex["x"],dtype=torch.float32).unsqueeze(0)
            preds.append(model(x).numpy()[0]); targets.append(ex["y"])
    p=np.asarray(preds)*target_sd+target_mu
    y=np.asarray(targets)*target_sd+target_mu
    return float(np.mean(np.abs(p-y))), float(np.sqrt(np.mean((p-y)**2)))

def main():
    token=os.environ.get("HF_TOKEN")
    if not token: raise RuntimeError("HF_TOKEN secret missing")
    OUT.mkdir(parents=True,exist_ok=True)
    by_shard=defaultdict(list)
    for row in EPISODES: by_shard[row["shard"]].append(row)
    records=[]; audit=[]; start=time.time()
    for shard, rows in sorted(by_shard.items()):
        url=hf_hub_url(repo_id=REPO,filename=shard,repo_type="dataset")
        # Reuse the exact range-read path that succeeded in the earlier
        # six-episode and nine-episode TouchScale workflows.
        fs=fsspec.filesystem("http",block_size=8*1024*1024,
                             headers={"Authorization":f"Bearer {token}"})
        with fs.open(url,"rb") as remote:
            with tarfile.open(fileobj=remote,mode="r:") as tar:
                wanted={r["id"]+".episode.h5" for r in rows}; found=set()
                for member in tar:
                    base=Path(member.name).name
                    if not member.isfile() or base not in wanted: continue
                    f=tar.extractfile(member)
                    if f is None: continue
                    blob=f.read()
                    meta=next(r for r in rows if r["id"]+".episode.h5"==base)
                    try:
                        with h5py.File(io.BytesIO(blob),"r") as h5:
                            x,y,a=episode_xy(h5,meta)
                        records.append({"label":meta["label"],"id":meta["id"],"x":x,"y":y})
                        audit.append(a)
                    except Exception as e:
                        audit.append({**meta,"status":"excluded","reason":str(e)})
                    found.add(base)
                    if found==wanted: break
                missing=sorted(wanted-found)
                if missing: raise RuntimeError(f"Missing HDF5 members in {shard}: {missing}")
    pd.DataFrame(audit).to_csv(OUT/"episode_signal_audit.csv",index=False)
    counts=pd.Series([r["label"] for r in records]).value_counts().to_dict()
    eligible={k:[r for r in records if r["label"]==k] for k,v in counts.items() if v>=2}
    if len(eligible)<4: raise RuntimeError(f"Too few usable categories after signal QC: {counts}")
    # Save the exact task pool; category is the task unit, complete Episode is sample unit.
    pd.DataFrame([{"episode_id":r["id"],"task_category":r["label"]} for r in records]).to_csv(OUT/"usable_task_pool.csv",index=False)
    rows_out=[]
    labels=sorted(eligible)
    for heldout in labels:
        train_labels=[x for x in labels if x!=heldout]
        train_raw={k:eligible[k] for k in train_labels}
        # Fit both scalers only on the outer-training categories.
        x_train=np.stack([e["x"] for arr in train_raw.values() for e in arr])
        y_train=np.stack([e["y"] for arr in train_raw.values() for e in arr])
        xs=StandardScaler().fit(x_train)
        ym=y_train.mean(axis=0); ys=y_train.std(axis=0); ys=np.where(ys<1e-5,1.0,ys)
        train={k:[{"x":xs.transform(e["x"][None,:])[0].astype(np.float32),
                   "y":((e["y"]-ym)/ys).astype(np.float32)} for e in arr]
               for k,arr in train_raw.items()}
        test_raw=eligible[heldout]
        test=[{"x":xs.transform(e["x"][None,:])[0].astype(np.float32),
               "y":((e["y"]-ym)/ys).astype(np.float32),"id":e["id"]} for e in test_raw]
        for seed in SEED_LIST:
            total_steps=META_ITERATIONS*min(META_BATCH,len(train_labels))*INNER_STEPS
            reptile=reptile_fit(train,45,105,seed+sum(map(ord,heldout)))
            pooled=pooled_fit(train,45,105,seed+sum(map(ord,heldout)),total_steps)
            rng=np.random.default_rng(seed+sum(map(ord,heldout))+991)
            for shots in (1,2,5):
                if len(test)<=shots: continue
                perm=rng.permutation(len(test))
                support=[test[i] for i in perm[:shots]]
                query=[test[i] for i in perm[shots:]]
                for name,model in (("pooled_mlp",pooled),("reptile",reptile)):
                    zero=score_zero(model,query,ym,ys)
                    adapted=adapt_and_score(model,support,query,seed,ym,ys)
                    rows_out.append({
                        "heldout_category":heldout,"seed":seed,"shots":shots,
                        "support_episode_ids":";".join(x["id"] for x in support),
                        "query_episode_ids":";".join(x["id"] for x in query),
                        "query_episodes":len(query),"model":name,
                        "zero_shot_mae":zero[0],"zero_shot_rmse":zero[1],
                        "adapted_mae":adapted[0],"adapted_rmse":adapted[1],
                        "meta_iterations":META_ITERATIONS if name=="reptile" else 0,
                        "gradient_updates":total_steps
                    })
    metrics=pd.DataFrame(rows_out)
    metrics.to_csv(OUT/"loco_fewshot_metrics.csv",index=False)
    if metrics.empty: raise RuntimeError("No held-out Support/Query evaluations were produced")
    aggregate=metrics.groupby(["shots","model"]).agg(
        heldout_categories=("heldout_category","nunique"),
        evaluation_runs=("heldout_category","count"),
        mean_query_episodes=("query_episodes","mean"),
        mean_zero_shot_mae=("zero_shot_mae","mean"),
        mean_adapted_mae=("adapted_mae","mean"),
        sd_adapted_mae=("adapted_mae","std"),
        mean_adapted_rmse=("adapted_rmse","mean")
    ).reset_index()
    aggregate.to_csv(OUT/"aggregate_metrics.csv",index=False)
    summary={
        "status":"success","candidate_episodes":len(EPISODES),
        "usable_episodes":len(records),"excluded_episodes":len(EPISODES)-len(records),
        "usable_categories":len(eligible),"episode_counts_by_category":{k:len(v) for k,v in eligible.items()},
        "shots_evaluated":sorted(metrics.shots.unique().tolist()),
        "heldout_category_counts_by_shot":metrics.groupby("shots").heldout_category.nunique().to_dict(),
        "metrics_rows":len(metrics),"elapsed_seconds":round(time.time()-start,2),
        "input":"early 40% continuous 15-pad pressure summaries (45 features)",
        "target":"late 40% 15-pad Pearson association upper triangle (105 targets)",
        "split":"leave-one-normalized-category-out; Support and Query are whole independent Episodes",
        "comparison":"Reptile vs pooled MLP, same outer-training categories and same nominal gradient-update count",
        "limits":[
            "Exploratory pilot based on 150-row stratified metadata sample, not full-population task census.",
            "Normalized category is the task unit; physical object instance identity is not established.",
            "Only seven candidate categories and few Episodes per category; 5-shot has very limited eligible held-out categories.",
            "Category/participant/session overlap and task wording heterogeneity may remain; interpret as pilot only.",
            "Target is within-Episode future sensor association, not a causal contact graph."
        ]
    }
    (OUT/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
