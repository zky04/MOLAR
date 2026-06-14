"""Run the full-data MOLAR benchmark across multiple GPUs."""

import argparse
import csv
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime


TASKS = [
    ("NSD2", "MFPCBA-1053173-743445"),
    ("SKN1", "MFPCBA-624474-624304"),
    ("CASP6", "MFPCBA-720632-686996"),
    ("RAD52", "MFPCBA-652116-651710"),
    ("GSK3A", "MFPCBA-463203-2650"),
    ("GIV", "MFPCBA-1259350-1224905"),
    ("UBC13", "MFPCBA-493155-485273"),
]


def parse_test_line(line):
    vals = {}
    if "Test:" not in line:
        return vals
    for part in line.strip().replace(":", " ").split():
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        try:
            vals[key.lower()] = float(value)
        except ValueError:
            pass
    return vals


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "abbrev", "task", "gpu", "status", "start_time", "end_time",
        "acc", "bacc", "auc", "auprc", "mcc", "f1", "log",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_task(args, abbrev, task, gpu_id, log_path):
    cmd = [
        sys.executable,
        "-u",
        "scripts/train_molar.py",
        "--task", task,
        "--config", args.config,
        "--data_root", args.data_root,
        "--device", "cuda",
        "--seed", str(args.seed),
    ]
    if args.num_epochs is not None:
        cmd.extend(["--num_epochs", str(args.num_epochs)])
    if args.max_train_samples is not None:
        cmd.extend(["--max_train_samples", str(args.max_train_samples)])
    if args.quick:
        cmd.append("--quick")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    vals = {}
    with open(log_path, "w") as log:
        log.write(f"# GPU {gpu_id}\n")
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        for line in proc.stdout:
            print(f"[GPU{gpu_id} {abbrev}] {line}", end="", flush=True)
            log.write(line)
            log.flush()
            parsed = parse_test_line(line)
            if parsed:
                vals.update(parsed)
        code = proc.wait()
    return code, vals


def worker(gpu_id, task_queue, args, rows, rows_lock, csv_path):
    while True:
        try:
            abbrev, task = task_queue.get_nowait()
        except queue.Empty:
            return

        start = datetime.now().isoformat(timespec="seconds")
        log_path = os.path.join(args.out_dir, "logs", f"{abbrev}_{task}_gpu{gpu_id}.log")
        print("=" * 80, flush=True)
        print(f"[{start}] GPU{gpu_id} {abbrev} {task}", flush=True)
        print("=" * 80, flush=True)
        code, vals = run_task(args, abbrev, task, gpu_id, log_path)
        end = datetime.now().isoformat(timespec="seconds")
        row = {
            "abbrev": abbrev,
            "task": task,
            "gpu": gpu_id,
            "status": "ok" if code == 0 else f"failed:{code}",
            "start_time": start,
            "end_time": end,
            "log": log_path,
            **vals,
        }
        with rows_lock:
            rows.append(row)
            write_csv(csv_path, sorted(rows, key=lambda r: TASKS.index((r["abbrev"], r["task"]))))
        task_queue.task_done()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config_molar.yaml")
    parser.add_argument("--data_root", default="noise-7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--out_dir", default="runs/molar_gpu_parallel")
    args = parser.parse_args()

    gpu_ids = [g.strip() for g in args.gpus.split(",") if g.strip()]
    if not gpu_ids:
        raise ValueError("--gpus must contain at least one GPU id")

    os.makedirs(os.path.join(args.out_dir, "logs"), exist_ok=True)
    csv_path = os.path.join(args.out_dir, "all_results.csv")
    task_queue = queue.Queue()
    for task in TASKS:
        task_queue.put(task)

    rows = []
    rows_lock = threading.Lock()
    threads = []
    for gpu_id in gpu_ids:
        t = threading.Thread(target=worker, args=(gpu_id, task_queue, args, rows, rows_lock, csv_path))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    failures = [r for r in rows if r["status"] != "ok"]
    print(f"[ALL DONE] {csv_path}", flush=True)
    if failures:
        print(f"[FAILURES] {len(failures)} task(s) failed", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
