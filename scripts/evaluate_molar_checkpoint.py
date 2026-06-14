"""Evaluate a saved MOLAR checkpoint on val/test splits."""

import argparse
import json
import os
import sys

import torch
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.train_molar import pick_device, set_seed, validate              


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out_json", default=None)
    args = parser.parse_args()

    from data.load_molar_data import load_molar_task
    from models.molar import MOLAR

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.data_root is not None:
        config["data_root"] = args.data_root

    num_tasks = int(config.get("num_tasks", 1))
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = pick_device(args.device)

    (
        _train_loader,
        val_loader,
        test_loader,
        _,
        _,
        _,
        gnn_dim,
        bond_dim,
        text_dim,
        stats,
    ) = load_molar_task(
        task_name=config["task_name"],
        data_root=config.get("data_root", "noise-7"),
        batch_size=int(config["batch_size"]),
        device=device,
        max_train_samples=config.get("max_train_samples"),
        force_zero_text=bool(config.get("force_zero_text", False)),
        num_tasks=num_tasks,
    )

    model_cfg = dict(config.get("model_args", {}))
    model_cfg.update(
        {
            "gnn_input_dim": gnn_dim,
            "bond_input_dim": bond_dim,
            "text_input_dim": text_dim,
            "num_tasks": num_tasks,
        }
    )
    model = MOLAR(model_cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state)

    val_metrics = validate(model, val_loader, device, num_tasks)
    test_metrics = validate(model, test_loader, device, num_tasks)
    result = {
        "task": config["task_name"],
        "seed": seed,
        "checkpoint": args.checkpoint,
        "config": args.config,
        "data_stats": stats,
        "val": val_metrics,
        "test": test_metrics,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
