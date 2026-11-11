"""Leakage-aware early tactile classification windows and recording splits."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import numpy as np

def _vec(mat: dict[str, Any], key: str, dtype) -> np.ndarray:
    if key not in mat:
        raise KeyError(f"metadata.mat 缺少字段: {key}")
    return np.asarray(mat[key]).reshape(-1).astype(dtype)

def _segments(ids: np.ndarray) -> list[np.ndarray]:
    if not ids.size: return []
    cuts = np.where(np.diff(ids) != 1)[0] + 1
    return [x for x in np.split(ids, cuts) if x.size]

def build_window_dataset(mat: dict[str, Any], *, k_frames: int = 5, stride: int = 10,
                         use_balanced: bool = False, exclude_empty_hand: bool = True):
    """Build windows that never cross recording, object, split, or invalid-frame gaps."""
    if k_frames < 1 or stride < 1: raise ValueError("k_frames 和 stride 必须为正整数")
    pressure = np.asarray(mat["pressure"])
    if pressure.ndim != 3 or pressure.shape[1:] != (32, 32):
        raise ValueError(f"pressure 应为 (N,32,32)，实际为 {pressure.shape}")
    n = pressure.shape[0]
    valid = _vec(mat, "hasValidLabel", bool)
    rec = _vec(mat, "recordingId", np.int64)
    obj = _vec(mat, "objectId", np.int64)
    split = _vec(mat, "splitId", np.int64)
    if any(v.size != n for v in (valid, rec, obj, split)):
        raise ValueError("pressure 与元数据字段长度不一致")
    mask = valid.copy()
    if use_balanced: mask &= _vec(mat, "isBalanced", bool)
    if exclude_empty_hand: mask &= obj != 0
    # Geometry-only mask; not used as a predictive label or model feature.
    sensor_mask = np.nanstd(pressure.astype(np.float32), axis=0) > 0
    if not sensor_mask.any(): raise ValueError("未识别到有效传感器像素")
    force = np.clip((1024.0 - pressure.astype(np.float32)) / 524.0, 0.0, 1.0)
    force[:, ~sensor_mask] = 0.0
    xs, maps, ys, recs, splits, starts = [], [], [], [], [], []
    for rid in np.unique(rec):
        ids = np.flatnonzero(rec == rid)
        for seg in _segments(ids[mask[ids]]):
            for pos in range(0, len(seg) - k_frames + 1, stride):
                win = seg[pos:pos+k_frames]
                if np.unique(obj[win]).size != 1 or np.unique(split[win]).size != 1: continue
                xs.append(force[win].reshape(k_frames, -1)[:, sensor_mask.ravel()])
                maps.append(force[win]); ys.append(int(obj[win[0]]))
                recs.append(int(rid)); splits.append(int(split[win[0]])); starts.append(int(win[0]))
    if not xs: raise ValueError("没有生成窗口；检查有效帧、k_frames 和 stride")
    arrays = {
        "force_vectors": np.stack(xs).astype(np.float32),
        "pressure_maps": np.stack(maps).astype(np.float32),
        "object_id": np.asarray(ys, np.int64),
        "recording_id": np.asarray(recs, np.int64),
        "split_id": np.asarray(splits, np.int64),
        "start_frame": np.asarray(starts, np.int64),
        "sensor_mask": sensor_mask.astype(bool),
    }
    raw_names = np.asarray(mat.get("objects", []), dtype=object).reshape(-1)
    names = []
    for x in raw_names:
        if isinstance(x, bytes): names.append(x.decode("utf-8", errors="replace"))
        elif isinstance(x, str): names.append(x)
        else: names.append(str(np.asarray(x).reshape(-1)[0]))
    meta = {"schema_version": 1, "task": "early_tactile_object_classification",
            "k_frames": k_frames, "stride": stride, "use_balanced": use_balanced,
            "n_windows": len(xs), "n_sensors": int(sensor_mask.sum()),
            "object_names": names}
    return arrays, meta

def save_window_dataset(arrays: dict[str, np.ndarray], meta: dict[str, Any], output: Path):
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    output.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

def split_by_recording(arrays: dict[str, np.ndarray], *, seed: int = 42,
                       train_ratio: float = .70, val_ratio: float = .15):
    """Class-stratified split; each recording belongs to exactly one partition."""
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("划分比例必须满足 train>0, val>=0, train+val<1")
    y, rec = arrays["object_id"], arrays["recording_id"]
    rng = np.random.default_rng(seed); assignment = {}; counts = {}
    for cls in np.unique(y):
        ids = rng.permutation(np.unique(rec[y == cls])); counts[int(cls)] = int(ids.size)
        if ids.size < 3:
            for rid in ids: assignment[int(rid)] = "train"
            continue
        nt = max(1, round(ids.size * (1-train_ratio-val_ratio)))
        nv = max(1, round(ids.size * val_ratio)) if val_ratio else 0
        while nt + nv >= ids.size:
            if nv: nv -= 1
            else: nt -= 1
        cut1, cut2 = ids.size-nt-nv, ids.size-nt
        for rid in ids[:cut1]: assignment[int(rid)] = "train"
        for rid in ids[cut1:cut2]: assignment[int(rid)] = "val"
        for rid in ids[cut2:]: assignment[int(rid)] = "test"
    labels = np.asarray([assignment[int(r)] for r in rec])
    out = {s: np.flatnonzero(labels == s) for s in ("train", "val", "test")}
    sets = {s: set(rec[idx].tolist()) for s, idx in out.items()}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]), "Recording 泄漏"
    out["class_recording_counts"] = counts
    return out
