"""
Experiment logger for ablation studies.
Saves all experiment configs and results to experiments/results.jsonl (one JSON per line).
Each entry captures: timestamp, method, model, task, hyperparams, metrics, and notes.

Usage:
    from experiment_logger import log_experiment, load_experiments, print_summary

    # Log a result
    log_experiment(
        method="stage1_finetune",
        model="chinese-macbert-large",
        task="promise",
        config={"epochs": 15, "lr": 2e-5, "fgm": True, "rdrop": True, "ema": True},
        metrics={"cv_f1": 0.9247, "fold_scores": [0.915, 0.892, 0.937, 0.956, 0.924]},
        notes="Baseline with all improvements"
    )

    # View all experiments
    print_summary()
"""

import json
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = os.path.join(BASE_DIR, "experiments", "results.jsonl")


def log_experiment(method, model, task, config, metrics, notes=""):
    """Append one experiment record to results.jsonl."""
    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "method": method,
        "model": model,
        "task": task,
        "config": config,
        "metrics": metrics,
        "notes": notes,
    }
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[LOG] {method} | {model} | {task} | {metrics}")
    return entry


def load_experiments(method=None, task=None, model=None):
    """Load experiments, optionally filtered."""
    if not os.path.exists(RESULTS_FILE):
        return []
    entries = []
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if method and entry.get("method") != method:
                continue
            if task and entry.get("task") != task:
                continue
            if model and entry.get("model") != model:
                continue
            entries.append(entry)
    return entries


def print_summary(method=None, task=None):
    """Print a formatted summary table of all experiments."""
    entries = load_experiments(method=method, task=task)
    if not entries:
        print("No experiments found.")
        return

    print(f"\n{'='*100}")
    print(f"{'Timestamp':<22} {'Method':<20} {'Model':<30} {'Task':<10} {'Key Metrics'}")
    print(f"{'='*100}")
    for e in entries:
        metrics_str = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in e["metrics"].items()
                                if k not in ("fold_scores", "fold_thresholds"))
        print(f"{e['timestamp']:<22} {e['method']:<20} {e['model']:<30} {e['task']:<10} {metrics_str}")
        if e.get("notes"):
            print(f"{'':>22} Notes: {e['notes']}")
    print(f"{'='*100}")
    print(f"Total: {len(entries)} experiments\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default=None)
    parser.add_argument("--task", default=None)
    args = parser.parse_args()
    print_summary(method=args.method, task=args.task)
