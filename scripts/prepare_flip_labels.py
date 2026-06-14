#!/usr/bin/env python3
"""Create fixed 5-fold controlled-noise label-flip protocols."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import KFold, StratifiedKFold


def stable_offset(text: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(text))


def torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def atomic_json_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=True) + "\n")
    os.replace(tmp, path)


def stratification_labels(clean_labels: torch.Tensor, masks: torch.Tensor, dataset: str):
    labels = clean_labels.numpy().astype(int)
    mask = masks.numpy().astype(int)
    if labels.shape[1] == 1:
        return labels[:, 0], f"{dataset}:task0"

    combo = np.array(["".join(str(int(v)) if int(m) else "x" for v, m in zip(row, row_mask)) for row, row_mask in zip(labels, mask)])
    _, counts = np.unique(combo, return_counts=True)
    if counts.min() >= 5:
        return combo, f"{dataset}:combo"

    best_j = None
    best_min = -1
    for j in range(labels.shape[1]):
        valid = mask[:, j] > 0
        vals, cls_counts = np.unique(labels[valid, j], return_counts=True)
        if len(vals) == 2 and cls_counts.min() > best_min:
            best_min = int(cls_counts.min())
            best_j = j
    if best_j is not None and best_min >= 5:
        return labels[:, best_j], f"{dataset}:task{best_j}"
    return None, f"{dataset}:kfold"


def make_base_folds(clean_labels: torch.Tensor, masks: torch.Tensor, dataset: str, seed: int, n_splits: int):
    n = clean_labels.size(0)
    indices = np.arange(n)
    stratify, strat_name = stratification_labels(clean_labels, masks, dataset)
    rng_seed = seed + stable_offset(dataset)
    if stratify is not None:
        _, counts = np.unique(stratify, return_counts=True)
        if counts.min() >= n_splits:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng_seed)
            folds = [np.sort(test_idx) for _, test_idx in splitter.split(indices, stratify)]
            return folds, strat_name
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=rng_seed)
    folds = [np.sort(test_idx) for _, test_idx in splitter.split(indices)]
    return folds, strat_name


def flip_labels(
    labels: torch.Tensor,
    masks: torch.Tensor,
    split_indices: torch.Tensor,
    noise_ratio: float,
    rng: np.random.Generator,
):
    split_labels = labels[split_indices].clone()
    n_flip = int(math.floor(noise_ratio * len(split_indices)))
    chosen = np.sort(rng.choice(len(split_indices), size=n_flip, replace=False)) if n_flip > 0 else np.array([], dtype=np.int64)
    if n_flip > 0:
        chosen_t = torch.tensor(chosen, dtype=torch.long)
        split_labels[chosen_t] = torch.where(
            masks[split_indices][chosen_t] > 0,
            1.0 - split_labels[chosen_t],
            split_labels[chosen_t],
        )
    return split_labels, torch.tensor(chosen, dtype=torch.long)


def write_samples_csv(path: Path, graphs, metadata: dict[str, Any], protocol: dict[str, Any], base_fold_of: np.ndarray):
    task_cols = metadata["task_cols"]
    clean_labels = protocol["clean_labels"]
    masks = protocol["masks"]
    with path.open("w", newline="") as f:
        fieldnames = ["graph_idx", "base_fold", "split", "split_pos", "row_idx", "is_flipped", "smiles"]
        for task in task_cols:
            fieldnames.extend([f"{task}_mask", f"{task}_clean", f"{task}_noisy"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for split_name in ["train", "val", "test"]:
            idx = protocol["split_indices"][split_name].long()
            noisy = protocol["noisy_labels"][split_name]
            flip_set = set(int(x) for x in protocol["flip_local_indices"][split_name].tolist())
            for pos, graph_idx in enumerate(idx.tolist()):
                graph = graphs[graph_idx]
                row = {
                    "graph_idx": graph_idx,
                    "base_fold": int(base_fold_of[graph_idx]),
                    "split": split_name,
                    "split_pos": pos,
                    "row_idx": int(getattr(graph, "row_idx", -1)),
                    "is_flipped": int(pos in flip_set),
                    "smiles": getattr(graph, "smiles", ""),
                }
                for j, task in enumerate(task_cols):
                    row[f"{task}_mask"] = int(masks[graph_idx, j].item())
                    row[f"{task}_clean"] = float(clean_labels[graph_idx, j].item())
                    row[f"{task}_noisy"] = float(noisy[pos, j].item())
                writer.writerow(row)


def prepare_dataset(args, dataset: str):
    data_dir = args.data_root / "processed" / dataset
    metadata = json.loads((data_dir / "metadata.json").read_text())
    graphs = torch_load(data_dir / "graphs.pt")
    clean_labels = torch.stack([g.y.view(-1).float() for g in graphs], dim=0)
    masks = torch.stack([g.mask.view(-1).float() for g in graphs], dim=0)
    folds, strat_name = make_base_folds(clean_labels, masks, dataset, args.seed, args.num_folds)
    base_fold_of = np.full(len(graphs), -1, dtype=np.int64)
    for fold_id, fold_idx in enumerate(folds):
        base_fold_of[fold_idx] = fold_id

    out_root = args.data_root / "flip_protocols" / f"kfold{args.num_folds}_3_1_1_flip{int(args.noise_ratio * 100)}_seed{args.seed}"
    fold_sizes = [int(len(x)) for x in folds]
    for fold_id in range(args.num_folds):
        test_idx = torch.tensor(folds[fold_id], dtype=torch.long)
        val_idx = torch.tensor(folds[(fold_id + 1) % args.num_folds], dtype=torch.long)
        train_parts = [folds[i] for i in range(args.num_folds) if i not in {fold_id, (fold_id + 1) % args.num_folds}]
        train_idx = torch.tensor(np.sort(np.concatenate(train_parts)), dtype=torch.long)

        rng = np.random.default_rng(args.seed + stable_offset(dataset) + fold_id * 1009)
        train_noisy, train_flip = flip_labels(clean_labels, masks, train_idx, args.noise_ratio, rng)
        val_noisy, val_flip = flip_labels(clean_labels, masks, val_idx, args.noise_ratio, rng)
        test_noisy = clean_labels[test_idx].clone()
        test_flip = torch.empty(0, dtype=torch.long)

        protocol = {
            "dataset": dataset,
            "seed": args.seed,
            "fold_id": fold_id,
            "num_folds": args.num_folds,
            "noise_ratio": args.noise_ratio,
            "split_method": f"kfold{args.num_folds}_3_1_1",
            "stratification": strat_name,
            "task_cols": metadata["task_cols"],
            "base_fold_sizes": fold_sizes,
            "base_fold_indices": {str(i): torch.tensor(folds[i], dtype=torch.long) for i in range(args.num_folds)},
            "split_indices": {"train": train_idx, "val": val_idx, "test": test_idx},
            "flip_local_indices": {"train": train_flip, "val": val_flip, "test": test_flip},
            "flip_global_indices": {"train": train_idx[train_flip], "val": val_idx[val_flip], "test": test_idx[test_flip]},
            "clean_labels": clean_labels,
            "masks": masks,
            "noisy_labels": {"train": train_noisy, "val": val_noisy, "test": test_noisy},
        }
        fold_dir = out_root / f"fold_{fold_id}"
        protocol_pt = fold_dir / f"{dataset}_protocol.pt"
        protocol_json = fold_dir / f"{dataset}_protocol.json"
        samples_csv = fold_dir / f"{dataset}_samples.csv"
        if protocol_pt.exists() and not args.force:
            print(f"exists {protocol_pt}", flush=True)
            continue
        atomic_torch_save(protocol, protocol_pt)
        atomic_json_save(
            {
                "dataset": dataset,
                "seed": args.seed,
                "fold_id": fold_id,
                "num_folds": args.num_folds,
                "noise_ratio": args.noise_ratio,
                "split_method": f"kfold{args.num_folds}_3_1_1",
                "stratification": strat_name,
                "task_cols": metadata["task_cols"],
                "base_fold_sizes": fold_sizes,
                "split_sizes": {k: int(len(v)) for k, v in protocol["split_indices"].items()},
                "flip_counts": {k: int(len(v)) for k, v in protocol["flip_local_indices"].items()},
                "protocol_pt": str(protocol_pt),
                "samples_csv": str(samples_csv),
            },
            protocol_json,
        )
        write_samples_csv(samples_csv, graphs, metadata, protocol, base_fold_of)
        print(
            f"{dataset} fold={fold_id} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
            f"flips={len(train_flip)}/{len(val_flip)}/0 strat={strat_name}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("control-noise"))
    parser.add_argument("--datasets", default="hiv,bace,bbbp,clintox")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--noise-ratio", type=float, default=0.30)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    for dataset in [x.strip().lower() for x in args.datasets.split(",") if x.strip()]:
        prepare_dataset(args, dataset)


if __name__ == "__main__":
    main()
