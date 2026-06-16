# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from TSB_AD.evaluation.basic_metrics import generate_curve
from TSB_AD.model_wrapper import (
    Semisupervise_AD_Pool,
    Unsupervise_AD_Pool,
    run_Semisupervise_AD,
    run_Unsupervise_AD,
)
from TSB_AD.utils.slidingWindows import find_length_rank


SOURCE_HP_CONFIG = {'CATSV2': {'win_size': [64], 'fusion': ['mul'], 'weight': [0.9]},
    'DAPHNET': {'win_size': [512], 'fusion': ['mul'], 'weight': [0.7]},
    'EXATHLON': {'win_size': [512], 'fusion': ['mul'], 'weight': [0.8]},
    'IOPS': {'win_size': [64], 'fusion': ['mul'], 'weight': [0.5]},
    'LTDB': {'win_size': [512], 'fusion': ['mul'], 'weight': [0.9]},
    'MGAB': {'win_size': [512], 'fusion': ['mul'], 'weight': [0.2]},
    'MITDB': {'win_size': [64], 'fusion': ['mul'], 'weight': [0.3]},
    'MSL': {'win_size': [64], 'fusion': ['mul'], 'weight': [0.5]},
    'NAB': {'win_size': [512], 'fusion': ['mul'], 'weight': [0.4]},
    'NEK': {'win_size': [64], 'fusion': ['add'], 'weight': [0.99]},
    'OPPORTUNITY': {'win_size': [64], 'fusion': ['mul'], 'weight': [0.5]},
    'POWER': {'win_size': [512], 'fusion': ['mul'], 'weight': [0.3]},
    'SED': {'win_size': [64], 'fusion': ['mul'], 'weight': [0.8]},
    'SMAP': {'win_size': [64], 'fusion': ['mul'], 'weight': [0.6]},
    'SMD': {'win_size': [512], 'fusion': ['add'], 'weight': [0.99]},
    'STOCK': {'win_size': [64], 'fusion': ['add'], 'weight': [0.99]},
    'SVDB': {'win_size': [64], 'fusion': ['mul'], 'weight': [0.8]},
    'SWAT': {'win_size': [512], 'fusion': ['mul'], 'weight': [0.7]},
    'TAO': {'win_size': [512], 'fusion': ['add'], 'weight': [0.99]},
    'TODS': {'win_size': [64], 'fusion': ['add'], 'weight': [0.1]},
    'UCR': {'win_size': [512], 'fusion': ['add'], 'weight': [0.1]},
    'WSD': {'win_size': [512], 'fusion': ['mul'], 'weight': [0.99]},
    'YAHOO': {'win_size': [64], 'fusion': ['mul'], 'weight': [0.01]}}



seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



def resolve_path(path):
    path = Path(path)
    candidates = [path, REPO_ROOT / path]
    if not path.is_absolute():
        candidates.append(Path(__file__).resolve().parent / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1] if not path.is_absolute() else path


def parse_source(filename):
    source = Path(filename).stem.split("_")[1].upper()
    if source == "SWAT":
        return "SWAT"
    if source.startswith("OPPORT"):
        return "OPPORTUNITY"
    return source


def hp_scalar(config, key):
    value = config[key]
    return value[0] if isinstance(value, list) else value


def train_prefix_length(filename, data_len):
    parts = Path(filename).stem.split("_")
    try:
        return int(parts[-3])
    except (IndexError, ValueError):
        return data_len


def align_score_length(score, target_len):
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if len(score) == target_len:
        return score
    if len(score) == 0:
        return np.zeros(target_len, dtype=np.float64)
    if len(score) < target_len:
        return np.pad(score, (target_len - len(score), 0), mode="edge")
    return score[:target_len]


def vus_pr(label, score, sliding_window, version="opt", thre=250):
    label = np.asarray(label, dtype=int).reshape(-1)
    score = align_score_length(score, len(label))
    if len(np.unique(label)) < 2:
        return float("nan")
    _, _, _, _, _, _, _, value = generate_curve(label, score, sliding_window, version, thre)
    return float(value)


def evaluate_one(args, filename):
    source = parse_source(filename)
    if source not in SOURCE_HP_CONFIG:
        raise KeyError(f"No source config for {source}: {filename}")

    file_path = Path(args.dataset_dir) / filename
    df = pd.read_csv(file_path).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df.iloc[:, -1].to_numpy(dtype=int)
    config = SOURCE_HP_CONFIG[source]
    train_index = train_prefix_length(filename, len(data))
    data_train = data[:train_index, :]

    params = dict(
        win_size=hp_scalar(config, "win_size"),
        weight=hp_scalar(config, "weight"),
        fusion=hp_scalar(config, "fusion"),
        lr_adapter=args.lr_adapter,
        epochs_adapter=args.epochs_adapter,
        adapter_mode=args.adapter_mode,
        device=args.device,
        adapter_checkpoint_dir=str(args.adapter_checkpoint_dir),
        timercd_win_size=args.timercd_win_size,
        timercd_checkpoint=str(args.timercd_checkpoint),
        filename=filename,
    )
    if args.AD_Name in Semisupervise_AD_Pool:
        score = run_Semisupervise_AD(args.AD_Name, data_train, data, **params)
    elif args.AD_Name in Unsupervise_AD_Pool:
        score = run_Unsupervise_AD(args.AD_Name, data, **params)
    else:
        raise ValueError(f"{args.AD_Name} is not defined")

    if isinstance(score, str):
        raise RuntimeError(score)

    sliding_window = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
    metric = vus_pr(label, score, sliding_window)

    return {
        "file": filename,
        "datasource": source,
        "HP": json.dumps({k: v for k, v in params.items() if k != "filename"}, sort_keys=True),
        "adapter_mode": args.adapter_mode,
        "timercd_mode": "model_suffix_prefix_zero",
        "VUS-PR": metric,
    }


def write_outputs(save_dir, ad_name, rows, elapsed):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    result_csv = save_dir / f"{ad_name}.csv"
    pd.DataFrame(rows).to_csv(result_csv, index=False)

    values = np.asarray([row["VUS-PR"] for row in rows], dtype=np.float64)
    summary = {
        "num_datasets": int(len(rows)),
        "num_sources": int(len({row["datasource"] for row in rows})),
        "mean_vus_pr": float(np.nanmean(values)) if len(values) else float("nan"),
        "median_vus_pr": float(np.nanmedian(values)) if len(values) else float("nan"),
        "elapsed_sec": float(elapsed),
        "config": "addmul_aucpr_0616_weight_fusion_source_level",
    }
    (save_dir / f"{ad_name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return result_csv, summary

def main():
    start_t = time.time()
    parser = argparse.ArgumentParser(description="HP Tuning for TimeRCD-MAFT add/mul VUS-PR fusion")
    parser.add_argument("--dataset_dir", type=str, default="Datasets/TSB-AD-U")
    parser.add_argument("--file_lsit", "--file_list", dest="file_list", type=str, default="Datasets/File_List/TSB-AD-U-Eva.csv")
    parser.add_argument("--save_dir", type=str, default="benchmark_exp/eval/HP_tuning/uni_ma_eps01")
    parser.add_argument("--AD_Name", type=str, default="TimeRCD_MAFT")
    parser.add_argument(
        "--adapter_mode",
        choices=["train", "checkpoint", "score", "auto"],
        default="train",
        help=(
            "train: fit MultiAdapter on the prefix; checkpoint: infer from .pt; "
            "score: use cached .npy; auto: checkpoint then score fallback."
        ),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--lr_adapter", type=float, default=0.001)
    parser.add_argument("--epochs_adapter", type=int, default=5)
    parser.add_argument(
        "--adapter_checkpoint_dir",
        type=str,
        default="checkpoints/MAFT",
    )
    parser.add_argument("--timercd_win_size", type=int, default=15000)
    parser.add_argument("--timercd_checkpoint", type=str, default="checkpoints/time-rcd/pretrain_checkpoint_best_uni.pth")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.dataset_dir = resolve_path(args.dataset_dir)
    args.file_list = resolve_path(args.file_list)
    args.adapter_checkpoint_dir = resolve_path(args.adapter_checkpoint_dir)
    args.timercd_checkpoint = resolve_path(args.timercd_checkpoint)
    args.save_dir = resolve_path(args.save_dir)

    print("CUDA available: ", torch.cuda.is_available())
    print("cuDNN version: ", torch.backends.cudnn.version())
    print(f"Adapter mode: {args.adapter_mode}")
    print(f"TimeRCD win_size: {args.timercd_win_size}")
    print(f"Metric: VUS-PR")

    file_list = pd.read_csv(args.file_list)["file_name"].tolist()
    if args.limit > 0:
        file_list = file_list[: args.limit]

    rows = []
    for index, filename in enumerate(file_list, start=1):
        print(f"[{index}/{len(file_list)}] Processing:{filename} by {args.AD_Name}", flush=True)
        row = evaluate_one(args, filename)
        print("VUS-PR: ", row["VUS-PR"], flush=True)
        rows.append(row)

        result_csv, _ = write_outputs(args.save_dir, args.AD_Name, rows, time.time() - start_t)
        print(f"Temp saved: {result_csv}", flush=True)

    result_csv, summary = write_outputs(args.save_dir, args.AD_Name, rows, time.time() - start_t)
    print(json.dumps(summary, indent=2))
    print(f"Wrote results to {result_csv}")
    print("Total time cost: {:.3f}s".format(time.time() - start_t))


if __name__ == "__main__":
    main()
