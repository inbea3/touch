#!/usr/bin/env python3
"""Diagnose Reptile adaptation dynamics and task-update compatibility on STAG pressure-only data."""
from __future__ import annotations
import argparse, csv, itertools, json, hashlib
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.preprocessing import StandardScaler

from run_reptile_meta_learning import (
    EdgeMLP, build_tasks, enumerate_object_episodes, tensors_for,
    full_support_loss, support_batch_loss, regression_metrics, model_state_sha256,
)
from run_object_coactivation_baselines import extract

METHODS = {
    "reptile": "reptile_seed_{}",
    "support_matched_mlp": "support_matched_mlp_seed_{}",
    "ordinary_mlp": "ordinary_mlp_seed_{}",
}

def load_model(path: Path, n_features: int, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = EdgeMLP(n_features).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model

def adapt(model, support, steps, lr, weight_decay, mode, rng, batch_size):
    adapted = EdgeMLP(next(iter(model.parameters())).shape[1]).to(next(iter(model.parameters())).device)
    adapted.load_state_dict(model.state_dict())
    if steps == 0:
        adapted.eval()
        return adapted
    optimizer = torch.optim.Adam(adapted.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if mode == "full_batch":
            loss = full_support_loss(adapted, support)
        elif mode == "train_style_minibatch":
            loss = support_batch_loss(adapted, support, batch_size, rng)
        else:
            raise ValueError(mode)
        loss.backward()
        optimizer.step()
    adapted.eval()
    return adapted

def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def vector_from_state(model, base_state):
    parts=[]
    for name, p in model.named_parameters():
        parts.append((p.detach() - base_state[name].detach()).reshape(-1).cpu().numpy())
    return np.concatenate(parts)

def task_direction_diagnostic(data, train_tasks, reptile_model, seed, inner_steps, inner_lr,
                              weight_decay, batch_size, device):
    rows=[]
    directions={}
    base_state={name:p.detach().clone() for name,p in reptile_model.named_parameters()}
    for shots in (1,2):
        per_object={}
        for oid, rids in sorted(train_tasks.items()):
            episode_vectors=[]
            for combo_idx, support_ids in enumerate(itertools.combinations(rids, shots)):
                support=tensors_for(data, list(support_ids), device)
                clone=EdgeMLP(next(iter(reptile_model.parameters())).shape[1]).to(device)
                clone.load_state_dict(reptile_model.state_dict())
                optimizer=torch.optim.Adam(clone.parameters(), lr=inner_lr, weight_decay=weight_decay)
                rng=np.random.default_rng(seed + oid*1009 + shots*101 + combo_idx)
                for _ in range(inner_steps):
                    optimizer.zero_grad(set_to_none=True)
                    loss=support_batch_loss(clone, support, batch_size, rng)
                    loss.backward()
                    optimizer.step()
                episode_vectors.append(vector_from_state(clone, base_state))
            mean_vec=np.mean(np.stack(episode_vectors), axis=0)
            norm=float(np.linalg.norm(mean_vec))
            if norm > 0:
                per_object[oid]=mean_vec/norm
            rows.append({
                "seed":seed,"shots":shots,"object_id":oid,
                "n_support_combinations":len(episode_vectors),
                "mean_task_displacement_norm":norm,
            })
        directions[shots]=per_object
    pair_rows=[]
    summary={}
    for shots, vecs in directions.items():
        ids=sorted(vecs)
        cos=[]
        for i,a in enumerate(ids):
            for b in ids[i+1:]:
                value=float(np.dot(vecs[a],vecs[b]))
                cos.append(value)
                pair_rows.append({"seed":seed,"shots":shots,"object_id_a":a,
                                  "object_id_b":b,"cosine_similarity":value})
        summary[str(shots)]={
            "n_objects":len(ids),"n_pairs":len(cos),
            "mean_pairwise_cosine":float(np.mean(cos)),
            "median_pairwise_cosine":float(np.median(cos)),
            "negative_pair_fraction":float(np.mean(np.asarray(cos)<0)),
            "positive_pair_fraction":float(np.mean(np.asarray(cos)>0)),
            "min_pairwise_cosine":float(np.min(cos)),
            "max_pairwise_cosine":float(np.max(cos)),
        }
    return rows,pair_rows,summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--metadata",type=Path,required=True)
    ap.add_argument("--split-manifest",type=Path,required=True)
    ap.add_argument("--checkpoint-dir",type=Path,required=True)
    ap.add_argument("--scaler-npz",type=Path,required=True)
    ap.add_argument("--output-dir",type=Path,required=True)
    ap.add_argument("--seeds",default="20261116,20261117,20261118")
    ap.add_argument("--nodes",type=int,default=96)
    ap.add_argument("--sensor-seed",type=int,default=20261116)
    ap.add_argument("--inner-steps",type=int,default=10)
    ap.add_argument("--inner-lr",type=float,default=0.001)
    ap.add_argument("--weight-decay",type=float,default=0.001)
    ap.add_argument("--batch-size",type=int,default=1024)
    ap.add_argument("--device",default="cpu")
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    seeds=[int(x) for x in args.seeds.split(",")]
    device=torch.device(args.device)
    manifest=json.loads(args.split_manifest.read_text())
    data,sensors=extract(loadmat(args.metadata,simplify_cells=True),manifest,n_nodes=args.nodes,
                         seed=args.sensor_seed,k=5)
    scaler_file=np.load(args.scaler_npz)
    scaler=StandardScaler()
    scaler.mean_=scaler_file["mean"]
    scaler.scale_=scaler_file["scale"]
    scaler.var_=scaler_file["var"]
    scaler.n_features_in_=int(scaler_file["n_features_in_"])
    for d in data.values():
        d["X"]=scaler.transform(d["X"]).astype(np.float32)
    train_tasks=build_tasks(data,"meta_train")
    val_tasks=build_tasks(data,"meta_validation")
    val_episodes=enumerate_object_episodes(val_tasks,np.random.default_rng(1))
    n_features=int(next(iter(data.values()))["X"].shape[1])
    all_metrics=[]
    direction_object_rows=[]
    direction_pair_rows=[]
    direction_summaries={}
    adaptation_steps=[0,1,2,5,10,20]
    for seed in seeds:
        models={}
        for method, pattern in METHODS.items():
            path=args.checkpoint_dir/(pattern.format(seed)+".pt")
            models[method]=load_model(path,n_features,device)
        hashes={m:model_state_sha256(model) for m,model in models.items()}
        print(json.dumps({"seed":seed,"checkpoint_hashes":hashes},sort_keys=True),flush=True)
        # A: fixed episodes, full-batch adaptation curve for 0/1/2/5/10/20 steps.
        # B: exact Reptile-style minibatch adaptation for 10 steps, compared with full batch.
        for ep_index,ep in enumerate(val_episodes,1):
            oid=int(ep["object_id"]); shots=int(ep["shots"])
            lookup={rid:(rid,x,y) for rid,x,y in tensors_for(data,val_tasks[oid],device)}
            support=[lookup[r] for r in ep["support_recording_ids"]]
            for method,base_model in models.items():
                for steps in adaptation_steps:
                    rng=np.random.default_rng(seed + oid*100003 + shots*1009 + ep_index*31 + steps)
                    fitted=adapt(base_model,support,steps,args.inner_lr,args.weight_decay,
                                 "full_batch",rng,args.batch_size)
                    for rid in ep["query_recording_ids"]:
                        _rid,x,y=lookup[rid]
                        with torch.no_grad():
                            pred=fitted(x).cpu().numpy()
                        met=regression_metrics(y.cpu().numpy(),pred)
                        all_metrics.append({
                            "seed":seed,"object_id":oid,"shots":shots,
                            "support_recording_ids":";".join(map(str,ep["support_recording_ids"])),
                            "query_recording_id":rid,"method":method,
                            "adaptation_mode":"full_batch","adapt_steps":steps,**met
                        })
                rng=np.random.default_rng(seed + oid*100003 + shots*1009 + ep_index*31 + 1000000)
                fitted=adapt(base_model,support,args.inner_steps,args.inner_lr,args.weight_decay,
                             "train_style_minibatch",rng,args.batch_size)
                for rid in ep["query_recording_ids"]:
                    _rid,x,y=lookup[rid]
                    with torch.no_grad():
                        pred=fitted(x).cpu().numpy()
                    met=regression_metrics(y.cpu().numpy(),pred)
                    all_metrics.append({
                        "seed":seed,"object_id":oid,"shots":shots,
                        "support_recording_ids":";".join(map(str,ep["support_recording_ids"])),
                        "query_recording_id":rid,"method":method,
                        "adaptation_mode":"train_style_minibatch","adapt_steps":args.inner_steps,**met
                    })
        # C: pairwise cosine compatibility of per-object mean inner-loop parameter displacements.
        reptile=models["reptile"]
        obj_rows,pair_rows,summary=task_direction_diagnostic(
            data,train_tasks,reptile,seed,args.inner_steps,args.inner_lr,args.weight_decay,
            args.batch_size,device)
        direction_object_rows.extend(obj_rows);direction_pair_rows.extend(pair_rows)
        direction_summaries[str(seed)]=summary
        print(json.dumps({"seed":seed,"task_direction_compatibility":summary},sort_keys=True),flush=True)
    write_csv(args.output_dir/"adaptation_episode_metrics.csv",all_metrics)
    # Aggregate recordings/episodes within each object and seed before across-seed/object summaries.
    buckets={}
    for row in all_metrics:
        key=(row["seed"],row["object_id"],row["shots"],row["method"],row["adaptation_mode"],row["adapt_steps"])
        buckets.setdefault(key,[]).append(row)
    object_seed=[]
    for key,vals in sorted(buckets.items()):
        seed,oid,shots,method,mode,steps=key
        item={"seed":seed,"object_id":oid,"shots":shots,"method":method,
              "adaptation_mode":mode,"adapt_steps":steps,"n_query_recording_evaluations":len(vals)}
        for metric in ("mae","rmse","spearman"):
            v=[x[metric] for x in vals if x[metric] is not None and np.isfinite(x[metric])]
            item[metric]=float(np.mean(v)) if v else None
        object_seed.append(item)
    write_csv(args.output_dir/"adaptation_object_seed_metrics.csv",object_seed)
    buckets={}
    for row in object_seed:
        key=(row["object_id"],row["shots"],row["method"],row["adaptation_mode"],row["adapt_steps"])
        buckets.setdefault(key,[]).append(row)
    object_rows=[]
    for key,vals in sorted(buckets.items()):
        oid,shots,method,mode,steps=key
        item={"object_id":oid,"shots":shots,"method":method,"adaptation_mode":mode,
              "adapt_steps":steps,"n_seeds":len(vals)}
        for metric in ("mae","rmse","spearman"):
            v=[x[metric] for x in vals if x[metric] is not None and np.isfinite(x[metric])]
            item[metric]=float(np.mean(v)) if v else None
        object_rows.append(item)
    write_csv(args.output_dir/"adaptation_object_metrics.csv",object_rows)
    write_csv(args.output_dir/"task_direction_object_metrics.csv",direction_object_rows)
    write_csv(args.output_dir/"task_direction_pairwise_cosines.csv",direction_pair_rows)
    summary={
        "scope":"STAG/HumanGrasp pressure-only; no video or multimodal input",
        "source_checkpoints":"200-iteration x 3-seed selected validation run",
        "validation_objects":sorted(val_tasks),
        "meta_train_objects":sorted(train_tasks),
        "independent_inferential_unit":"object",
        "adaptation_steps_full_batch":adaptation_steps,
        "train_style_minibatch_steps":args.inner_steps,
        "training_inner_batch_size":args.batch_size,
        "training_inner_steps":args.inner_steps,
        "task_direction_cosine_by_seed":direction_summaries,
        "notes":[
            "A/B use the same fixed exhaustive Support/Query recording permutations for all methods.",
            "Full-batch adaptation evaluates 0,1,2,5,10,20 steps; train-style adaptation uses the training inner-loop per-Recording minibatch construction for 10 steps.",
            "C averages parameter displacement vectors over all Support combinations per training object, separately for 1-shot and 2-shot, then computes pairwise cosine similarities between 18 object-level vectors.",
            "Pairwise object cosines are descriptive and statistically dependent; they are not treated as independent replicates.",
            "All summaries aggregate query recordings within object before reporting object-level metrics."
        ]
    }
    (args.output_dir/"diagnostic_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("DIAGNOSTIC_COMPLETE",flush=True)
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__":
    main()
