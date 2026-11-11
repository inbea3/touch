#!/usr/bin/env python3
"""Build early tactile object-classification windows from metadata.mat."""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scipy.io import loadmat
from touch_research.classification.data import build_window_dataset, save_window_dataset

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, default=Path("data/interim/metadata.mat"))
    p.add_argument("--output", type=Path, default=Path("data/processed/early_classification.npz"))
    p.add_argument("--k-frames", type=int, default=5); p.add_argument("--stride", type=int, default=10)
    p.add_argument("--balanced", action="store_true"); a = p.parse_args()
    if not a.metadata.is_file(): p.error(f"找不到 {a.metadata}；先按 docs/data_audit.md 获取并解压数据")
    arrays, meta = build_window_dataset(loadmat(a.metadata, simplify_cells=True),
        k_frames=a.k_frames, stride=a.stride, use_balanced=a.balanced)
    save_window_dataset(arrays, meta, a.output)
    print(f"窗口数={meta['n_windows']}，有效传感器={meta['n_sensors']}，输出={a.output}")
    print("仅完成数据构建；不代表分类实验已经取得结果。")
if __name__ == "__main__": main()
