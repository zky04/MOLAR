"""Data loading utilities for MOLAR natural-noise and controlled-noise tasks."""

import os
from typing import Optional

import pandas as pd
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

RDLogger.DisableLog("rdApp.warning")


BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def _atom_features(atom):
    feats = [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetFormalCharge(),
        atom.GetNumRadicalElectrons(),
        int(atom.GetIsAromatic()),
        atom.GetTotalNumHs(),
        float(atom.GetHybridization()),
        atom.GetTotalValence(),
        atom.GetExplicitValence(),
        atom.GetImplicitValence(),
        int(atom.IsInRing()),
        atom.GetMass(),
        float(atom.GetChiralTag()),
    ]
    while len(feats) < 36:
        feats.append(0.0)
    return feats[:36]


def _bond_features(bond):
    btype = bond.GetBondType()
    feats = [float(btype == t) for t in BOND_TYPES]
    feats.extend([
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
        float(bond.GetStereo()),
        float(bond.GetBondDir()),
    ])
    while len(feats) < 12:
        feats.append(0.0)
    return feats[:12]


def smiles_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = [_atom_features(atom) for atom in mol.GetAtoms()]
    if not atom_features:
        return None

    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = _bond_features(bond)
        edge_index.extend([(i, j), (j, i)])
        edge_attr.extend([feat, feat])

    x = torch.tensor(atom_features, dtype=torch.float)
    if edge_index:
        ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        ea = torch.tensor(edge_attr, dtype=torch.float)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long)
        ea = torch.zeros((0, 12), dtype=torch.float)
    return Data(x=x, edge_index=ei, edge_attr=ea)


def _normalize_graph(g, num_tasks: int = 1):
    if getattr(g, "edge_attr", None) is None:
        g.edge_attr = torch.zeros((g.edge_index.size(1), 12), dtype=torch.float)
    if hasattr(g, "y"):
        y = g.y.float()
        if y.numel() == 1:
            fixed_y = torch.zeros((1, num_tasks), dtype=torch.float)
            fixed_y[0, 0] = y.view(-1)[0]
            g.y = fixed_y
        else:
            g.y = y.view(1, num_tasks)
    if not hasattr(g, "mask"):
        g.mask = torch.ones((1, num_tasks), dtype=torch.float)
    else:
        mask = g.mask.float()
        if mask.numel() == 1:
            fixed_mask = torch.zeros((1, num_tasks), dtype=torch.float)
            fixed_mask[0, 0] = mask.view(-1)[0]
            g.mask = fixed_mask
        else:
            g.mask = mask.view(1, num_tasks)
    return g


def _generate_text_embeddings(smiles_list, model, tokenizer, device, batch_size=64):
    from data.molecule_description import batch_smiles_to_descriptions

    descs = batch_smiles_to_descriptions(smiles_list)
    embs = []
    for i in range(0, len(descs), batch_size):
        batch = descs[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embs.append(outputs.last_hidden_state[:, 0, :].cpu())
    return torch.cat(embs, dim=0)


def _cache_is_compatible(cached, train_n, val_n, test_n):
    return (
        "train" in cached and "val" in cached and "test" in cached
        and cached["train"].shape[0] == train_n
        and cached["val"].shape[0] == val_n
        and cached["test"].shape[0] == test_n
    )


def _load_graph_cache(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _cache_file_ready(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def _atomic_torch_save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def _graph_cache_is_compatible(cached, train_n, val_n, test_n):
    return (
        isinstance(cached, dict)
        and "train" in cached and "val" in cached and "test" in cached
        and len(cached["train"]) == train_n
        and len(cached["val"]) == val_n
        and len(cached["test"]) == test_n
    )


def load_molar_task(
    task_name: str,
    data_root: str = "noise-7",
    batch_size: int = 32,
    text_model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    device: torch.device = torch.device("cpu"),
    max_train_samples: Optional[int] = None,
    force_zero_text: bool = False,
    num_tasks: int = 1,
):
    task_dir = os.path.join(data_root, task_name)
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    sd_train_full = pd.read_csv(os.path.join(task_dir, "sd_train.csv"))
    dr_val = pd.read_csv(os.path.join(task_dir, "dr_val.csv"))
    dr_test = pd.read_csv(os.path.join(task_dir, "dr_test.csv"))
    sd_train_full["_orig_idx"] = range(len(sd_train_full))
    sd_train = sd_train_full

    if max_train_samples and len(sd_train) > max_train_samples:
        sd_train = sd_train.sample(n=max_train_samples, random_state=42)

    print(f"  sd_train: {len(sd_train)}  dr_val: {len(dr_val)}  dr_test: {len(dr_test)}")
    cache_dir = os.path.join(task_dir, ".cache")
    full_train_n = len(sd_train_full)
    graph_cache_file = os.path.join(cache_dir, f"graphs_molar_full_n{full_train_n}.pt")

    def df_to_graphs(df, label_col):
        graphs = []
        for _, row in df.iterrows():
            g = smiles_to_graph(row["SMILES"])
            if g is None:
                continue
            y = torch.zeros((1, num_tasks), dtype=torch.float)
            mask = torch.zeros((1, num_tasks), dtype=torch.float)
            y[0, 0] = float(row[label_col])
            mask[0, 0] = 1.0
            g.y = y
            g.mask = mask
            g.smiles = row["SMILES"]
            g.cid = int(row["CID"])
            if "_orig_idx" in row:
                g.orig_idx = int(row["_orig_idx"])
            graphs.append(g)
        return graphs

    train_graphs = val_graphs = test_graphs = None
    if _cache_file_ready(graph_cache_file):
        try:
            cached_graphs = _load_graph_cache(graph_cache_file)
            if _graph_cache_is_compatible(cached_graphs, full_train_n, len(dr_val), len(dr_test)):
                if max_train_samples and len(sd_train) < full_train_n:
                    train_idx = [int(i) for i in sd_train["_orig_idx"].tolist()]
                    train_graphs = [cached_graphs["train"][i] for i in train_idx]
                else:
                    train_graphs = cached_graphs["train"]
                val_graphs = cached_graphs["val"]
                test_graphs = cached_graphs["test"]
                train_graphs = [_normalize_graph(g, num_tasks) for g in train_graphs]
                val_graphs = [_normalize_graph(g, num_tasks) for g in val_graphs]
                test_graphs = [_normalize_graph(g, num_tasks) for g in test_graphs]
                print("  Graphs loaded from cache")
            else:
                print(f"  Graph cache mismatch for {graph_cache_file}; rebuilding.")
        except Exception as e:
            print(f"  Graph cache failed ({e}); rebuilding.")

    if train_graphs is None:
        train_graphs = df_to_graphs(sd_train, "sd_label")
        val_graphs = df_to_graphs(dr_val, "dr_label")
        test_graphs = df_to_graphs(dr_test, "dr_label")
        if not max_train_samples:
            _atomic_torch_save({"train": train_graphs, "val": val_graphs, "test": test_graphs}, graph_cache_file)
            print(f"  Graphs cached: {graph_cache_file}")
    print(f"  Graphs: train={len(train_graphs)}  val={len(val_graphs)}  test={len(test_graphs)}")

    n_train = len(train_graphs)
    cache_candidates = [os.path.join(cache_dir, f"text_embeddings_n{n_train}.pt")]
    full_cache_file = os.path.join(cache_dir, f"text_embeddings_n{full_train_n}.pt")
    if full_cache_file not in cache_candidates:
        cache_candidates.append(full_cache_file)
    text_input_dim = 256

    if force_zero_text:
        print("  Using zero text embeddings (forced).")
        train_text_emb = torch.zeros(len(train_graphs), text_input_dim)
        val_text_emb = torch.zeros(len(val_graphs), text_input_dim)
        test_text_emb = torch.zeros(len(test_graphs), text_input_dim)
    elif any(_cache_file_ready(path) for path in cache_candidates):
        cache_file = next(path for path in cache_candidates if _cache_file_ready(path))
        cached = torch.load(cache_file, map_location="cpu")
        if _cache_is_compatible(cached, len(train_graphs), len(val_graphs), len(test_graphs)):
            train_text_emb = cached["train"]
            val_text_emb = cached["val"]
            test_text_emb = cached["test"]
            text_input_dim = train_text_emb.shape[1]
            print(f"  Text embeddings: {text_input_dim}d (cached)")
        elif (
            cached.get("train") is not None
            and cached["train"].shape[0] == full_train_n
            and cached.get("val") is not None
            and cached.get("test") is not None
            and cached["val"].shape[0] == len(val_graphs)
            and cached["test"].shape[0] == len(test_graphs)
            and all(hasattr(g, "orig_idx") for g in train_graphs)
        ):
            train_idx = torch.tensor([g.orig_idx for g in train_graphs], dtype=torch.long)
            train_text_emb = cached["train"][train_idx]
            val_text_emb = cached["val"]
            test_text_emb = cached["test"]
            text_input_dim = train_text_emb.shape[1]
            print(f"  Text embeddings: {text_input_dim}d (subset from full cache)")
        else:
            print(f"  Cache shape mismatch for {cache_file}; ignoring cache.")
            try:
                from transformers import AutoModel, AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(text_model_name)
                text_model = AutoModel.from_pretrained(text_model_name).to(device)
                text_model.eval()

                train_text_emb = _generate_text_embeddings([g.smiles for g in train_graphs], text_model, tokenizer, device)
                val_text_emb = _generate_text_embeddings([g.smiles for g in val_graphs], text_model, tokenizer, device)
                test_text_emb = _generate_text_embeddings([g.smiles for g in test_graphs], text_model, tokenizer, device)
                text_input_dim = train_text_emb.shape[1]
                print(f"  Text embeddings: {text_input_dim}d (generated, cache left unchanged)")
            except Exception as e:
                print(f"  Warning: text model failed ({e}), using zeros.")
                train_text_emb = torch.zeros(len(train_graphs), text_input_dim)
                val_text_emb = torch.zeros(len(val_graphs), text_input_dim)
                test_text_emb = torch.zeros(len(test_graphs), text_input_dim)
    else:
        try:
            from transformers import AutoModel, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(text_model_name)
            text_model = AutoModel.from_pretrained(text_model_name).to(device)
            text_model.eval()

            train_text_emb = _generate_text_embeddings([g.smiles for g in train_graphs], text_model, tokenizer, device)
            val_text_emb = _generate_text_embeddings([g.smiles for g in val_graphs], text_model, tokenizer, device)
            test_text_emb = _generate_text_embeddings([g.smiles for g in test_graphs], text_model, tokenizer, device)
            text_input_dim = train_text_emb.shape[1]
            _atomic_torch_save({"train": train_text_emb, "val": val_text_emb, "test": test_text_emb}, cache_file)
            print(f"  Text embeddings: {text_input_dim}d (generated and cached)")
        except Exception as e:
            print(f"  Warning: text model failed ({e}), using zeros.")
            train_text_emb = torch.zeros(len(train_graphs), text_input_dim)
            val_text_emb = torch.zeros(len(val_graphs), text_input_dim)
            test_text_emb = torch.zeros(len(test_graphs), text_input_dim)

    for g, t in zip(train_graphs, train_text_emb):
        g.text_emb = t
    for g, t in zip(val_graphs, val_text_emb):
        g.text_emb = t
    for g, t in zip(test_graphs, test_text_emb):
        g.text_emb = t

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    train_labels = torch.tensor([g.y.view(-1)[0].item() for g in train_graphs])
    val_labels = torch.tensor([g.y.view(-1)[0].item() for g in val_graphs])
    test_labels = torch.tensor([g.y.view(-1)[0].item() for g in test_graphs])

    stats = {
        "task": task_name,
        "train_n": len(train_graphs), "train_pos": int(train_labels.sum().item()),
        "val_n": len(val_graphs), "val_pos": int(val_labels.sum().item()),
        "test_n": len(test_graphs), "test_pos": int(test_labels.sum().item()),
        "gnn_dim": train_graphs[0].x.shape[1] if train_graphs else 36,
        "bond_dim": train_graphs[0].edge_attr.shape[1] if train_graphs else 12,
        "text_dim": text_input_dim,
        "num_tasks": num_tasks,
    }

    return (
        train_loader, val_loader, test_loader,
        train_text_emb, val_text_emb, test_text_emb,
        stats["gnn_dim"], stats["bond_dim"], text_input_dim, stats,
    )
