#!/usr/bin/env python3
"""Prepare the controlled-noise datasets for MOLAR.

The script writes graph, text, split, and metadata files under
control-noise/processed/<dataset>/.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.load_molar_data import smiles_to_graph
from data.molecule_description import batch_smiles_to_descriptions

RDLogger.DisableLog("rdApp.*")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    filename: str
    smiles_col: str
    task_cols: tuple[str, ...] | None
    task_count: int
    exclude_cols: tuple[str, ...] = ()


DATASETS: dict[str, DatasetSpec] = {
    "hiv": DatasetSpec(
        "hiv",
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv",
        "HIV.csv",
        "smiles",
        ("HIV_active",),
        1,
        ("activity",),
    ),
    "bace": DatasetSpec(
        "bace",
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
        "bace.csv",
        "mol",
        ("Class",),
        1,
        ("pIC50", "Model", "Set", "CID"),
    ),
    "bbbp": DatasetSpec(
        "bbbp",
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv",
        "BBBP.csv",
        "smiles",
        ("p_np",),
        1,
        ("name", "num"),
    ),
    "clintox": DatasetSpec(
        "clintox",
        "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz",
        "clintox.csv.gz",
        "smiles",
        ("FDA_APPROVED", "CT_TOX"),
        2,
    ),
}


def atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def atomic_text_save(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    try:
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def download_file(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"  raw exists: {dest}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"  downloading: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "molar-data-prep/1.0"})
    with urllib.request.urlopen(req, timeout=300) as response, open(tmp, "wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)


def read_csv(path: Path) -> pd.DataFrame:
    compression = "gzip" if path.suffix == ".gz" else None
    return pd.read_csv(path, compression=compression)


def is_binary_label_column(series: pd.Series) -> bool:
    vals = pd.to_numeric(series, errors="coerce").dropna().unique()
    if len(vals) == 0:
        return False
    return set(float(v) for v in vals).issubset({0.0, 1.0})


def infer_task_cols(df: pd.DataFrame, spec: DatasetSpec) -> list[str]:
    if spec.task_cols is not None:
        missing = [c for c in spec.task_cols if c not in df.columns]
        if missing:
            raise ValueError(f"{spec.name}: missing task columns: {missing}")
        return list(spec.task_cols)

    excluded = {spec.smiles_col, *spec.exclude_cols}
    excluded |= {"smiles", "mol", "name", "num", "mol_id", "id", "cid", "CID", "Unnamed: 0"}
    task_cols = [c for c in df.columns if c not in excluded and is_binary_label_column(df[c])]
    if len(task_cols) != spec.task_count:
        raise ValueError(
            f"{spec.name}: inferred {len(task_cols)} task columns, "
            f"expected {spec.task_count}. Columns: {task_cols[:10]}..."
        )
    return task_cols


def normalize_labels(df: pd.DataFrame, task_cols: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    labels = torch.zeros((len(df), len(task_cols)), dtype=torch.float)
    masks = torch.zeros((len(df), len(task_cols)), dtype=torch.float)

    for j, col in enumerate(task_cols):
        values = pd.to_numeric(df[col], errors="coerce")
        present = values.notna()
        labels[present.to_numpy(), j] = torch.tensor(
            values[present].astype(float).to_numpy(),
            dtype=torch.float,
        )
        masks[present.to_numpy(), j] = 1.0
    return labels, masks


def scaffold_for_smiles(smiles: str) -> str:
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=True)
    except Exception:
        scaffold = ""
    return scaffold or smiles


def make_scaffold_split(smiles_list: list[str], train_frac: float, val_frac: float) -> dict[str, torch.Tensor]:
    scaffold_to_indices: dict[str, list[int]] = {}
    for idx, smiles in enumerate(smiles_list):
        scaffold_to_indices.setdefault(scaffold_for_smiles(smiles), []).append(idx)

    groups = sorted(scaffold_to_indices.values(), key=lambda x: (-len(x), x[0]))
    n_total = len(smiles_list)
    n_train = int(train_frac * n_total)
    n_val = int(val_frac * n_total)

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for group in groups:
        if len(train) + len(group) <= n_train:
            train.extend(group)
        elif len(val) + len(group) <= n_val:
            val.extend(group)
        else:
            test.extend(group)

    return {
        "train": torch.tensor(sorted(train), dtype=torch.long),
        "val": torch.tensor(sorted(val), dtype=torch.long),
        "test": torch.tensor(sorted(test), dtype=torch.long),
    }


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def embed_texts(
    texts: list[str],
    model_name: str,
    device: torch.device,
    batch_size: int,
    local_files_only: bool,
) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only).to(device)
    model.eval()

    embeddings: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model(**inputs)
        embeddings.append(output.last_hidden_state[:, 0, :].detach().cpu())
        done = min(start + batch_size, len(texts))
        if done == len(texts) or done % (batch_size * 20) == 0:
            print(f"    embedded {done}/{len(texts)}")

    return torch.cat(embeddings, dim=0)


def save_text_jsonl(graphs: Iterable, path: Path) -> None:
    lines = []
    for idx, graph in enumerate(graphs):
        lines.append(json.dumps({
            "idx": idx,
            "smiles": graph.smiles,
            "text": graph.text,
        }, ensure_ascii=True))
    atomic_text_save("\n".join(lines) + "\n", path)


def process_dataset(args: argparse.Namespace, spec: DatasetSpec) -> dict:
    raw_path = Path(args.root) / "raw" / spec.name / spec.filename
    processed_dir = Path(args.root) / "processed" / spec.name
    download_file(spec.url, raw_path, force=args.force_download)

    print(f"[{spec.name}] reading raw csv")
    df = read_csv(raw_path)
    if spec.smiles_col not in df.columns:
        raise ValueError(f"{spec.name}: missing SMILES column {spec.smiles_col!r}")
    task_cols = infer_task_cols(df, spec)
    labels, masks = normalize_labels(df, task_cols)

    graphs = []
    valid_row_indices = []
    invalid_rows = 0
    print(f"[{spec.name}] building graphs")
    smiles_values = df[spec.smiles_col].astype(str).tolist()

    for row_idx, smiles in enumerate(smiles_values):
        graph = smiles_to_graph(smiles)
        if graph is None:
            invalid_rows += 1
            continue
        graph.y = labels[row_idx].view(1, -1)
        graph.mask = masks[row_idx].view(1, -1)
        graph.smiles = smiles
        graph.dataset = spec.name
        graph.row_idx = int(row_idx)
        if "mol_id" in df.columns and pd.notna(df.iloc[row_idx]["mol_id"]):
            graph.mol_id = str(df.iloc[row_idx]["mol_id"])
        if "CID" in df.columns and pd.notna(df.iloc[row_idx]["CID"]):
            graph.cid = str(df.iloc[row_idx]["CID"])
        graphs.append(graph)
        valid_row_indices.append(row_idx)

    if not graphs:
        raise RuntimeError(f"{spec.name}: no valid molecular graphs were produced")

    print(f"[{spec.name}] generating molecule texts")
    descriptions = batch_smiles_to_descriptions([g.smiles for g in graphs])
    for graph, text in zip(graphs, descriptions):
        graph.text = text

    valid_texts = [g.text for g in graphs]
    if args.skip_embeddings:
        text_dim = int(args.zero_text_dim)
        text_embeddings = torch.zeros((len(graphs), text_dim), dtype=torch.float)
        text_backend = "zeros"
    else:
        print(f"[{spec.name}] generating text embeddings on {args.device_resolved}")
        text_embeddings = embed_texts(
            valid_texts,
            args.text_model,
            args.device_resolved,
            args.embedding_batch_size,
            args.local_files_only,
        )
        text_dim = int(text_embeddings.shape[1])
        text_backend = args.text_model

    for graph, emb in zip(graphs, text_embeddings):
        graph.text_emb = emb

    split = make_scaffold_split([g.smiles for g in graphs], args.train_frac, args.val_frac)
    train_graphs = [graphs[i] for i in split["train"].tolist()]
    val_graphs = [graphs[i] for i in split["val"].tolist()]
    test_graphs = [graphs[i] for i in split["test"].tolist()]

    processed_dir.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(graphs, processed_dir / "graphs.pt")
    atomic_torch_save(text_embeddings, processed_dir / "text_embeddings_biomedbert.pt")
    atomic_torch_save(split, processed_dir / "splits" / "scaffold.pt")
    atomic_torch_save(train_graphs, processed_dir / "train_graphs.pt")
    atomic_torch_save(val_graphs, processed_dir / "val_graphs.pt")
    atomic_torch_save(test_graphs, processed_dir / "test_graphs.pt")
    save_text_jsonl(graphs, processed_dir / "texts.jsonl")

    label_mask = torch.stack([g.mask.view(-1) for g in graphs], dim=0)
    label_y = torch.stack([g.y.view(-1) for g in graphs], dim=0)
    nodes = torch.tensor([g.num_nodes for g in graphs], dtype=torch.float)
    pyg_edges = torch.tensor([g.edge_index.size(1) for g in graphs], dtype=torch.float)
    task_stats = []
    for j, col in enumerate(task_cols):
        valid = label_mask[:, j] > 0
        positives = int(label_y[valid, j].sum().item()) if valid.any() else 0
        task_stats.append({
            "task": col,
            "available": int(valid.sum().item()),
            "positive": positives,
            "negative": int(valid.sum().item()) - positives,
        })

    metadata = {
        "dataset": spec.name,
        "source_url": spec.url,
        "raw_file": str(raw_path),
        "raw_rows": int(len(df)),
        "valid_graphs": int(len(graphs)),
        "invalid_smiles": int(invalid_rows),
        "smiles_col": spec.smiles_col,
        "task_cols": task_cols,
        "num_tasks": int(len(task_cols)),
        "graph_feature_dim": int(graphs[0].x.shape[1]),
        "bond_feature_dim": int(graphs[0].edge_attr.shape[1]),
        "text_dim": text_dim,
        "text_backend": text_backend,
        "avg_nodes": float(nodes.mean().item()),
        "avg_pyg_edges": float(pyg_edges.mean().item()),
        "avg_undirected_edges": float((pyg_edges / 2.0).mean().item()),
        "split": {
            "method": "scaffold",
            "train": int(len(train_graphs)),
            "val": int(len(val_graphs)),
            "test": int(len(test_graphs)),
            "train_frac_requested": args.train_frac,
            "val_frac_requested": args.val_frac,
        },
        "task_stats": task_stats,
    }
    atomic_text_save(json.dumps(metadata, indent=2, ensure_ascii=True) + "\n", processed_dir / "metadata.json")

    print(
        f"[{spec.name}] done: graphs={len(graphs)} invalid={invalid_rows} "
        f"tasks={len(task_cols)} text_dim={text_dim} "
        f"split={len(train_graphs)}/{len(val_graphs)}/{len(test_graphs)}"
    )
    return metadata


def parse_dataset_names(value: str) -> list[str]:
    if value.lower() == "all":
        return list(DATASETS)
    names = [x.strip().lower() for x in value.split(",") if x.strip()]
    unknown = [x for x in names if x not in DATASETS]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Available: {sorted(DATASETS)}")
    return names


def collect_processed_manifest(root: str) -> list[dict]:
    processed = Path(root) / "processed"
    if not processed.exists():
        return []
    records = []
    for metadata_path in sorted(processed.glob("*/metadata.json")):
        with open(metadata_path) as f:
            records.append(json.load(f))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", default="hiv,bace,bbbp,clintox", help="Comma-separated names or 'all'.")
    parser.add_argument("--root", default="control-noise")
    parser.add_argument("--text-model", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--zero-text-dim", type=int, default=768)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()

    if not 0 < args.train_frac < 1:
        raise ValueError("--train-frac must be in (0, 1)")
    if not 0 <= args.val_frac < 1:
        raise ValueError("--val-frac must be in [0, 1)")
    if args.train_frac + args.val_frac >= 1:
        raise ValueError("--train-frac + --val-frac must be < 1")

    args.device_resolved = pick_device(args.device)
    if args.manifest_only:
        manifest = collect_processed_manifest(args.root)
        manifest_path = Path(args.root) / "processed" / "manifest.json"
        atomic_text_save(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", manifest_path)
        print(f"Saved manifest: {manifest_path} ({len(manifest)} datasets)")
        return

    names = parse_dataset_names(args.datasets)
    manifest = []
    print(f"Preparing datasets: {', '.join(names)}")
    print(f"Output root: {args.root}")

    for name in names:
        manifest.append(process_dataset(args, DATASETS[name]))

    manifest = collect_processed_manifest(args.root)
    manifest_path = Path(args.root) / "processed" / "manifest.json"
    atomic_text_save(json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", manifest_path)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
