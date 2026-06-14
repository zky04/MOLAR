"""Train MOLAR on one MF-PCBA noisy-label task."""

import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader as PyGDataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def pick_device(requested=None):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_task_tensor(x, num_tasks):
    return x.view(-1, num_tasks).float()


def compute_metrics(y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "bacc": balanced_accuracy_score(y_true, y_pred),
        "auprc": average_precision_score(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        metrics["auc"] = float("nan")
    return metrics


def validate(model, loader, device, num_tasks):
    model.eval()
    labels, probs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch, batch.text_emb.to(device))
            y = as_task_tensor(batch.y, num_tasks)
            labels.append(y[:, 0].cpu())
            probs.append(out["p_clean"][:, 0].cpu())
    y_true = torch.cat(labels).numpy()
    y_prob = torch.cat(probs).numpy()
    return compute_metrics(y_true, y_prob)


def make_balanced_loader(graphs, batch_size):
    labels = np.array([int(g.y.view(-1)[0].item()) for g in graphs])
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return PyGDataLoader(graphs, batch_size=batch_size, shuffle=True)
    repeat_factor = max(1, len(neg_idx) // len(pos_idx))
    pos_os = np.tile(pos_idx, repeat_factor)
    if len(pos_os) < len(neg_idx):
        pos_os = np.concatenate([pos_os, np.random.choice(pos_idx, len(neg_idx) - len(pos_os))])
    all_idx = np.concatenate([neg_idx, pos_os])
    np.random.shuffle(all_idx)
    return PyGDataLoader([graphs[i] for i in all_idx], batch_size=batch_size, shuffle=False)


def augment_batch(batch, dropout):
    strong = batch.clone()
    atom_mask_rate = dropout / 2.0
    text_drop_rate = dropout
    mod_drop_rate = dropout / 2.0

    if atom_mask_rate > 0 and hasattr(strong, "x"):
        keep_nodes = (torch.rand(strong.x.size(0), 1, device=strong.x.device) > atom_mask_rate).float()
        strong.x = strong.x * keep_nodes

    if text_drop_rate > 0 and hasattr(strong, "text_emb"):
        keep_text = (torch.rand_like(strong.text_emb) > text_drop_rate).float()
        strong.text_emb = strong.text_emb * keep_text

    if mod_drop_rate > 0 and hasattr(strong, "batch"):
        num_graphs = int(strong.batch.max().item()) + 1 if strong.batch.numel() else 0
        if num_graphs > 0:
            drop = torch.rand(num_graphs, device=strong.x.device) < mod_drop_rate
            drop_graph = drop & (torch.rand(num_graphs, device=strong.x.device) < 0.5)
            drop_text = drop & ~drop_graph
            if drop_graph.any():
                strong.x = strong.x * (~drop_graph[strong.batch]).float().unsqueeze(-1)
            if drop_text.any() and hasattr(strong, "text_emb"):
                text = strong.text_emb.view(num_graphs, -1).clone()
                text[drop_text] = 0.0
                strong.text_emb = text.reshape(-1)
    return strong


def build_optimizer(model, config, model_cfg):
    lr = float(config.get("lr", 3e-4))
    noise_lr_scale = float(model_cfg.get("noise_channel_lr_scale", 0.1))
    noise_names = {"a_plus", "a_minus"}
    noise_params, base_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in noise_names:
            noise_params.append(param)
        else:
            base_params.append(param)
    return optim.Adam(
        [
            {"params": base_params, "lr": lr},
            {"params": noise_params, "lr": lr * noise_lr_scale},
        ],
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )


def main(config):
    from data.load_molar_data import load_molar_task
    from models.molar import MOLAR
    from models.molar_loss import MOLARLoss

    num_tasks = int(config.get("num_tasks", 1))
    device = pick_device(config.get("device"))
    set_seed(int(config.get("seed", 42)))

    task_name = config["task_name"]
    save_dir = config.get(
        "save_dir",
        f"runs/molar_{task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(save_dir, exist_ok=True)

    print("=" * 72)
    print(f"MOLAR on {task_name}")
    print(f"Device: {device} | Seed: {config.get('seed', 42)} | Save: {save_dir}")
    print("=" * 72)

    print("\n[1/4] Loading data...")
    (
        train_loader, val_loader, test_loader,
        _, _, _, gnn_dim, bond_dim, text_dim, stats,
    ) = load_molar_task(
        task_name=task_name,
        data_root=config.get("data_root", "noise-7"),
        batch_size=int(config["batch_size"]),
        device=device,
        max_train_samples=config.get("max_train_samples"),
        force_zero_text=bool(config.get("force_zero_text", False)),
        num_tasks=num_tasks,
    )
    print(
        f"  Train: {stats['train_n']} (pos={stats['train_pos']}) | "
        f"Val: {stats['val_n']} (pos={stats['val_pos']}) | "
        f"Test: {stats['test_n']} (pos={stats['test_pos']})"
    )

    model_cfg = dict(config.get("model_args", {}))
    model_cfg.update({
        "gnn_input_dim": gnn_dim,
        "bond_input_dim": bond_dim,
        "text_input_dim": text_dim,
        "num_tasks": num_tasks,
    })

    print("\n[2/4] Building model...")
    model = MOLAR(model_cfg).to(device)
    if bool(model_cfg.get("initialize_endpoint_bias_from_observed_prevalence", True)):
        train_rate = max(min(stats["train_pos"] / max(stats["train_n"], 1), 1.0 - 1e-4), 1e-4)
        model.initialize_endpoint_base([train_rate])
        print(f"  Endpoint base initialized from observed train prevalence: {train_rate:.6f}")
    teacher = model.create_teacher().to(device)

    loss_cfg = dict(model_cfg)
    loss_cfg.update({
        "num_tasks": num_tasks,
        "beta_conflict": model_cfg.get("beta_conflict", 0.1),
        "gamma_temporal": model_cfg.get("gamma_temporal", 0.3),
        "supervised_target": model_cfg.get("supervised_target", "observed"),
        "unimodal_weight": model_cfg.get("unimodal_weight", 0.0),
    })
    if bool(model_cfg.get("use_class_weights", False)):
        train_pos = max(stats["train_pos"], 1)
        train_neg = max(stats["train_n"] - stats["train_pos"], 1)
        total = train_pos + train_neg
        weight_power = float(model_cfg.get("class_weight_power", 1.0))
        loss_cfg["pos_weight"] = [(train_neg / total) ** weight_power]
        loss_cfg["neg_weight"] = [(train_pos / total) ** weight_power]
        loss_cfg["class_weight_power"] = weight_power
    loss_fn = MOLARLoss(loss_cfg)
    optimizer = build_optimizer(model, config, model_cfg)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(config["num_epochs"]))

    sampler_mode = str(config.get("sampler_mode", model_cfg.get("sampler_mode", "shuffle" if model_cfg.get("method_strict", True) else "balanced")))
    if sampler_mode == "balanced":
        train_epoch_loader = make_balanced_loader(list(train_loader.dataset), int(config["batch_size"]))
    elif sampler_mode == "shuffle":
        train_epoch_loader = PyGDataLoader(list(train_loader.dataset), batch_size=int(config["batch_size"]), shuffle=True)
    else:
        raise ValueError("sampler_mode must be one of {'shuffle', 'balanced'}")
    print(f"  Sampler mode: {sampler_mode} ({len(train_epoch_loader)} batches/epoch)")
    print(f"  Noise channel lr scale: {model_cfg.get('noise_channel_lr_scale', 0.1)}")
    print(f"  Evidence dropout: {model_cfg.get('evidence_dropout', 0.0)}")
    print(f"  Evidence L2: {model_cfg.get('evidence_l2', 0.0)}")
    print(f"  Supervised target: {loss_cfg['supervised_target']}")
    print(f"  Unimodal weight: {loss_cfg['unimodal_weight']}")
    if "pos_weight" in loss_cfg:
        print(
            f"  Class weights: pos={loss_cfg['pos_weight'][0]:.4f} "
            f"neg={loss_cfg['neg_weight'][0]:.4f} "
            f"power={loss_cfg.get('class_weight_power', 1.0):.2f}"
        )

    warmup_epochs = int(config.get("warmup_epochs", 5))
    num_epochs = int(config["num_epochs"])
    dropout = float(model_cfg.get("dropout", 0.2))
    use_temporal_aug = float(model_cfg.get("gamma_temporal", 0.3)) > 0.0
    best_auc, best_epoch, patience = -1.0, 0, 0

    print(f"\n[3/4] Training for {num_epochs} epochs (warmup={warmup_epochs})...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        teacher.eval()
        epoch_stats = []
        correct, seen = 0, 0
        skipped_steps = 0
        warmup = epoch <= warmup_epochs

        for batch in train_epoch_loader:
            batch = batch.to(device)
            y = as_task_tensor(batch.y, num_tasks).to(device)
            mask = as_task_tensor(batch.mask, num_tasks).to(device)

            out_w = model(batch, batch.text_emb.to(device))
            with torch.no_grad():
                out_t = teacher(batch, batch.text_emb.to(device))

            if warmup or not use_temporal_aug:
                out_s = None
            else:
                strong_batch = augment_batch(batch, dropout)
                out_s = model(strong_batch, strong_batch.text_emb.to(device))

            loss, stats_loss = loss_fn(
                out_w,
                y,
                mask,
                outputs_s=out_s,
                teacher_outputs_w=out_t,
                warmup=warmup,
                return_stats=True,
            )
            if not torch.isfinite(loss):
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("grad_clip", 5.0)))
            if not torch.isfinite(grad_norm):
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.step()
            model.ema_update(teacher, momentum=float(config.get("ema_momentum", 0.995)))

            epoch_stats.append(stats_loss)
            pred = (out_w["p_clean"][:, 0] >= 0.5).float()
            correct += (pred == y[:, 0]).sum().item()
            seen += y.size(0)

        if not epoch_stats:
            raise RuntimeError(f"All training steps were non-finite at epoch {epoch}")

        scheduler.step()
        val_m = validate(model, val_loader, device, num_tasks)
        mean_stats = {k: float(np.mean([s[k] for s in epoch_stats])) for k in epoch_stats[0]}

        if val_m["auc"] > best_auc:
            best_auc = val_m["auc"]
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))
            patience = 0
            mark = "*"
        else:
            patience += 1
            mark = ""

        if epoch % 5 == 0 or mark or epoch <= warmup_epochs:
            extra_stats = ""
            if "mean_w_supervised" in mean_stats:
                extra_stats = (
                    f" w_sup={mean_stats['mean_w_supervised']:.3f}"
                    f" a_gt={mean_stats.get('mean_a_gt', 0.0):.3f}"
                )
            print(
                f"  Ep {epoch:3d} | loss={mean_stats['loss']:.4f} "
                f"sup={mean_stats['loss_sup']:.4f} conf={mean_stats['loss_conf']:.4f} "
                f"mod={mean_stats.get('loss_mod', 0.0):.4f} "
                f"temp={mean_stats['loss_temp']:.4f} train_acc={correct/max(seen,1):.4f} | "
                f"val AUC={val_m['auc']:.4f} ACC={val_m['accuracy']:.4f} "
                f"r={mean_stats['mean_r']:.3f} d+= {mean_stats['mean_d_plus']:.3f} "
                f"d-= {mean_stats['mean_d_minus']:.3f} |u_g|={mean_stats['mean_abs_u_g']:.3f} "
                f"|u_t|={mean_stats['mean_abs_u_t']:.3f}{extra_stats} skip={skipped_steps} {mark}"
            )

        if patience >= int(config.get("patience_epoch", 20)):
            print(f"  Early stop at epoch {epoch}")
            break

    print(f"\n[4/4] Testing best epoch {best_epoch}...")
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pt"), map_location=device))
    test_m = validate(model, test_loader, device, num_tasks)
    print(
        f"  Test: ACC={test_m['accuracy']:.4f} BACC={test_m['bacc']:.4f} "
        f"AUC={test_m['auc']:.4f} AUPRC={test_m['auprc']:.4f} "
        f"MCC={test_m['mcc']:.4f} F1={test_m['f1']:.4f}"
    )

    results = {
        "method": "MOLAR",
        "task": task_name,
        "best_epoch": best_epoch,
        "seed": int(config.get("seed", 42)),
        "config": config,
        "data_stats": stats,
        **test_m,
    }
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(save_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(config, f)
    print(f"Results saved to {save_dir}/")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--config", default="configs/config_molar.yaml")
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--force_zero_text", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["task_name"] = args.task
    cfg["seed"] = args.seed
    if args.data_root is not None:
        cfg["data_root"] = args.data_root
    if args.device is not None:
        cfg["device"] = args.device
    if args.force_zero_text:
        cfg["force_zero_text"] = True
    if args.max_train_samples is not None:
        cfg["max_train_samples"] = args.max_train_samples
    if args.num_epochs is not None:
        cfg["num_epochs"] = args.num_epochs
    if args.quick:
        cfg["max_train_samples"] = cfg.get("max_train_samples", 20000)
        cfg["num_epochs"] = min(int(cfg.get("num_epochs", 15)), 15)
        cfg["warmup_epochs"] = min(int(cfg.get("warmup_epochs", 3)), 3)
        print("[QUICK MODE]")
    main(cfg)
