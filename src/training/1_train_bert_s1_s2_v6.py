"""
Grid Search BERT S1 + S2 — v6
================================
Tương tự S3 v6: grid search oversample x weight để tìm sweet spot Macro F1.

S1 (promise_status):  Clear=0 -> No, 1 -> Yes  (No la minority)
S2 (evidence_status): 0 -> No, 1 -> Yes        (No la minority)

Hard constraint: Yes recall >= YES_RECALL_MIN
  (Yes la majority class va co weight cao trong composite score:
   S1 weight=0.20, S2 weight=0.30 -> khong duoc de Yes recall sap)

Select best: uu tien macro_f1 (vi composite score dung F1 binary,
              nhung macro_f1 phan anh balance tot)

Grid:
  S1: oversample No in [3, 4, 5] x weight No in [1.5, 2.0, 2.5]
  S2: oversample No in [2, 3, 4] x weight No in [1.5, 2.0, 2.5]

Output:
  D:\\LLMs\\BERT-ESG\\macbert-large-s1-v6\\
  D:\\LLMs\\BERT-ESG\\roberta-wwm-large-s2-v6\\
  D:\\LLMs\\BERT-ESG\\macbert-large-s1-v6\\grid_results.json
  D:\\LLMs\\BERT-ESG\\roberta-wwm-large-s2-v6\\grid_results.json
"""

import json
import os
import random
import gc
import shutil
import numpy as np
import torch
import torch.nn as nn
from collections import Counter
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from torch.utils.data import Dataset

# ==========================================
# CONFIG
# ==========================================
TRAIN_JSON  = r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_1000 V1.json"
VAL_JSON    = r"C:\Users\VU\Documents\NLP\AICup26\datasets\validation\vpesg4k_val_1000.json"
OUTPUT_BASE = r"D:\LLMs\BERT-ESG"
SEED        = 42

LABEL2ID = {"No": 0, "Yes": 1}
ID2LABEL = {0: "No", 1: "Yes"}

YES_RECALL_MIN = 0.85   # hard constraint: khong de Yes recall sap qua

TASK_CONFIGS = {
    "s1": {
        "model_id":   "hfl/chinese-macbert-large",
        "out_name":   "macbert-large-s1-v6",
        "oversample_grid": [3, 4, 5],
        "weight_grid":     [1.5, 2.0, 2.5],
        "epochs":     8,
        "patience":   3,
        "batch_size": 8,
    },
    "s2": {
        "model_id":   "hfl/chinese-roberta-wwm-ext-large",
        "out_name":   "roberta-wwm-large-s2-v6",
        "oversample_grid": [2, 3, 4],
        "weight_grid":     [1.5, 2.0, 2.5],
        "epochs":     8,
        "patience":   3,
        "batch_size": 8,
    },
}

TRAIN_ARGS_BASE = dict(
    per_device_eval_batch_size = 8,
    learning_rate           = 2e-5,
    weight_decay            = 0.01,
    warmup_ratio            = 0.1,
    eval_strategy           = "epoch",
    save_strategy           = "epoch",
    save_total_limit        = 1,
    load_best_model_at_end  = True,
    metric_for_best_model   = "macro_f1",
    greater_is_better       = True,
    logging_steps           = 20,
    bf16                    = True,
    seed                    = SEED,
    report_to               = "none",
    dataloader_num_workers  = 0,
)

# ==========================================
# NORMALIZE
# ==========================================
def normalize_timeline(records):
    for r in records:
        if r.get("verification_timeline") == "longer_than_5_years":
            r["verification_timeline"] = "more_than_5_years"
    return records

# ==========================================
# LOAD DATA
# ==========================================
def load_combined_data():
    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_records = json.load(f)
    with open(VAL_JSON, "r", encoding="utf-8") as f:
        val_records = json.load(f)

    train_records = normalize_timeline(train_records)
    val_records   = normalize_timeline(val_records)

    random.seed(SEED)
    random.shuffle(train_records)
    train_800 = train_records[:800]
    test_200  = train_records[800:]
    all_train = train_800 + val_records   # 1800 rows

    print(f"\nData loaded: {len(all_train)} train (800+{len(val_records)}) | "
          f"{len(test_200)} test")

    for task in ["s1", "s2"]:
        tr = task_rows(all_train, task)
        te = task_rows(test_200, task)
        dist_tr = Counter(str(r["__label__"]) for r in tr)
        dist_te = Counter(str(r["__label__"]) for r in te)
        ratio = dist_tr.get("Yes", 1) / max(dist_tr.get("No", 1), 1)
        print(f"  {task.upper()}: train={len(tr)} {dict(dist_tr)} | "
              f"test={len(te)} {dict(dist_te)} | imbalance(Yes/No)={ratio:.2f}x")

    return all_train, test_200

def task_rows(records, task):
    """Lay rows + gan __label__ cho task tuong ung"""
    out = []
    for r in records:
        promise = str(r.get("promise_status", "")).strip()
        if task == "s1":
            if promise not in LABEL2ID:
                continue
            r2 = dict(r)
            r2["__label__"] = promise
            out.append(r2)
        elif task == "s2":
            if promise != "Yes":
                continue
            evidence = str(r.get("evidence_status", "")).strip()
            if evidence not in LABEL2ID:
                continue
            r2 = dict(r)
            r2["__label__"] = evidence
            out.append(r2)
    return out

# ==========================================
# DATASET
# ==========================================
class BinDataset(Dataset):
    def __init__(self, records, task, tokenizer, oversample_no=1, oversample_mode=True):
        self.tokenizer = tokenizer
        self.samples   = []
        rows = task_rows(records, task)

        oversample_map = {"No": oversample_no, "Yes": 1}

        for r in rows:
            label  = LABEL2ID[r["__label__"]]
            repeat = oversample_map[r["__label__"]] if oversample_mode else 1
            data   = str(r.get("data", "")).strip()
            for _ in range(repeat):
                self.samples.append({"text": data, "label": label})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item     = self.samples[idx]
        encoding = self.tokenizer(
            item["text"],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels":         torch.tensor(item["label"], dtype=torch.long),
        }

# ==========================================
# WEIGHTED TRAINER
# ==========================================
class WeightedTrainer(Trainer):
    def __init__(self, weight_no, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # weight[No, Yes] = [weight_no, 1.0]
        self.class_weights = [weight_no, 1.0]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.get("logits")
        weights = torch.tensor(
            self.class_weights, dtype=torch.float, device=logits.device
        )
        loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss

# ==========================================
# METRICS
# ==========================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds    = np.argmax(logits, axis=-1)
    acc      = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}

# ==========================================
# TRAIN ONE COMBO
# ==========================================
def train_one_combo(task, cfg, oversample_no, weight_no,
                    train_records, test_records, tokenizer, run_dir):
    os.makedirs(run_dir, exist_ok=True)

    train_ds = BinDataset(train_records, task, tokenizer, oversample_no, oversample_mode=True)
    eval_ds  = BinDataset(test_records,  task, tokenizer, oversample_no=1, oversample_mode=False)

    dist = Counter(s["label"] for s in train_ds.samples)
    print(f"    Train: {len(train_ds)} samples  "
          f"No={dist[0]} / Yes={dist[1]}")
    print(f"    Eval : {len(eval_ds)} samples")

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_id"],
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=run_dir,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        **TRAIN_ARGS_BASE,
    )

    trainer = WeightedTrainer(
        weight_no=weight_no,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["patience"])],
    )

    trainer.train()

    preds_out = trainer.predict(eval_ds)
    preds     = np.argmax(preds_out.predictions, axis=-1)
    labels    = preds_out.label_ids
    report    = classification_report(
        labels, preds, target_names=["No", "Yes"], output_dict=True
    )

    no_f1      = report["No"]["f1-score"]
    no_recall  = report["No"]["recall"]
    yes_recall = report["Yes"]["recall"]
    macro_f1   = report["macro avg"]["f1-score"]

    print(f"\n    Results:")
    print(classification_report(labels, preds, target_names=["No", "Yes"]))

    trainer.save_model(run_dir)
    tokenizer.save_pretrained(run_dir)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "oversample_no":   oversample_no,
        "weight_no":       weight_no,
        "effective_boost": round(oversample_no * weight_no, 2),
        "no_f1":           round(no_f1,      4),
        "no_recall":       round(no_recall,  4),
        "yes_recall":      round(yes_recall, 4),
        "macro_f1":        round(macro_f1,   4),
        "run_dir":         run_dir,
        "passes_constraint": yes_recall >= YES_RECALL_MIN,
    }

# ==========================================
# SELECT BEST
# ==========================================
def select_best(results):
    """
    Uu tien:
      1. Yes recall >= YES_RECALL_MIN (hard constraint, Yes la majority + weight cao)
      2. Macro F1 cao nhat (composite score dung F1 binary -> can balance)
      3. Tie-break: No F1 cao hon
    """
    valid = [r for r in results if r["passes_constraint"]]
    pool  = valid if valid else results

    best = max(pool, key=lambda r: (r["macro_f1"], r["no_f1"]))
    return best

# ==========================================
# PRINT GRID SUMMARY
# ==========================================
def print_grid_summary(task, results):
    print("\n" + "=" * 80)
    print(f"  GRID SEARCH SUMMARY — {task.upper()}")
    print("=" * 80)
    header = f"{'OS':>3}  {'W':>4}  {'Boost':>6}  {'No_F1':>6}  {'No_Rec':>7}  {'Yes_Rec':>7}  {'MacF1':>6}  {'Pass':>5}"
    print(header)
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["macro_f1"], reverse=True):
        flag = "OK" if r["passes_constraint"] else "X"
        print(
            f"  x{r['oversample_no']}  "
            f"w{r['weight_no']:.1f}  "
            f"{r['effective_boost']:>6.1f}x  "
            f"{r['no_f1']:>6.4f}  "
            f"{r['no_recall']:>7.4f}  "
            f"{r['yes_recall']:>7.4f}  "
            f"{r['macro_f1']:>6.4f}  "
            f"{flag:>5}"
        )
    print("=" * 80)

# ==========================================
# RUN GRID FOR ONE TASK
# ==========================================
def run_grid_for_task(task, train_records, test_records):
    cfg = TASK_CONFIGS[task]
    print(f"\n{'#'*70}")
    print(f"#  TASK: {task.upper()}  ({cfg['model_id']})")
    print(f"{'#'*70}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])

    grid = [(o, w) for o in cfg["oversample_grid"] for w in cfg["weight_grid"]]
    print(f"\nGrid: {len(cfg['oversample_grid'])} oversample x "
          f"{len(cfg['weight_grid'])} weights = {len(grid)} combos")
    print(f"Hard constraint: Yes recall >= {YES_RECALL_MIN}\n")

    results = []
    for idx, (oversample_no, weight_no) in enumerate(grid, 1):
        effective = oversample_no * weight_no
        run_name  = f"{task}-v6-os{oversample_no}-w{weight_no:.1f}"
        run_dir   = os.path.join(OUTPUT_BASE, f"grid_{task}_v6", run_name)

        print(f"\n[{idx}/{len(grid)}] oversample=x{oversample_no}  "
              f"weight={weight_no}  effective={effective:.1f}x")
        print(f"  Dir: {run_dir}")

        result = train_one_combo(
            task, cfg, oversample_no, weight_no,
            train_records, test_records, tokenizer, run_dir,
        )
        results.append(result)

    print_grid_summary(task, results)

    # Save grid results
    final_dir = os.path.join(OUTPUT_BASE, cfg["out_name"])
    os.makedirs(final_dir, exist_ok=True)
    with open(os.path.join(final_dir, "grid_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Select & copy best
    best = select_best(results)
    print(f"\n{'='*60}")
    print(f"  Best combo for {task.upper()}:")
    print(f"    oversample  = x{best['oversample_no']}")
    print(f"    weight      = {best['weight_no']}")
    print(f"    effective   = {best['effective_boost']}x")
    print(f"    No F1       = {best['no_f1']}")
    print(f"    No Recall   = {best['no_recall']}")
    print(f"    Yes Recall  = {best['yes_recall']}  "
          f"({'OK pass' if best['passes_constraint'] else 'X fail'})")
    print(f"    Macro F1    = {best['macro_f1']}")
    print(f"    Source dir  = {best['run_dir']}")

    for fname in os.listdir(best["run_dir"]):
        src = os.path.join(best["run_dir"], fname)
        dst = os.path.join(final_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    best_cfg = {**best, "final_dir": final_dir}
    with open(os.path.join(final_dir, "train_metrics_v6.json"), "w", encoding="utf-8") as f:
        json.dump(best_cfg, f, indent=2, ensure_ascii=False)

    print(f"\n  Best model copied to: {final_dir}")
    print(f"{'='*60}")

    return best

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CPU mode")

    train_records, test_records = load_combined_data()

    bests = {}
    for task in ["s1", "s2"]:
        bests[task] = run_grid_for_task(task, train_records, test_records)

    print("\n" + "=" * 60)
    print("  FINAL SUMMARY")
    print("=" * 60)
    for task, b in bests.items():
        cfg = TASK_CONFIGS[task]
        print(f"  {task.upper()}: {cfg['out_name']}")
        print(f"    oversample=x{b['oversample_no']} weight={b['weight_no']} "
              f"-> Macro F1={b['macro_f1']}  No F1={b['no_f1']}  "
              f"Yes Recall={b['yes_recall']}")

    print("\nNext: update main_pipeline_v4.py:")
    for task, b in bests.items():
        cfg = TASK_CONFIGS[task]
        print(f'  "{task}": r"D:\\LLMs\\BERT-ESG\\{cfg["out_name"]}"')