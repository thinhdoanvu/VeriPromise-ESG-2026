"""
Retrain BERT S3-only — v6
==========================
Mục tiêu: cải thiện Not Clear recall (v5: 0.38 quá thấp)

Phân tích:
  S3 imbalance ratio = 4.97x
  v4: oversample x2 × weight 4.0 = effective 8x  → Not Clear recall=0.81, precision=0.38 (over)
  v5: oversample x2 × weight 2.2 = effective 4.4x → Not Clear recall=0.38 (under)
  v6: grid search 3×3 = 9 combos → tìm sweet spot

Grid:
  oversample : [2, 3, 4]
  weight     : [3.0, 3.5, 4.0]
  effective  : 6x / 7x / 8x / 9x / 10.5x / 12x / 8x / 10.5x / 16x

Metric tối ưu: Not Clear F1 (nhưng Clear recall >= 0.80 là hard constraint)

Output:
  D:\\LLMs\\BERT-ESG\\macbert-large-s3-v6\\   ← best combo
  D:\\LLMs\\BERT-ESG\\macbert-large-s3-v6\\grid_results.json
"""

import json
import os
import random
import gc
import copy
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
MODEL_ID    = "hfl/chinese-macbert-large"
SEED        = 42

S3_LABEL2ID = {"Clear": 0, "Not Clear": 1}
S3_ID2LABEL = {0: "Clear", 1: "Not Clear"}

# Hard constraint: Clear recall phải >= ngưỡng này
CLEAR_RECALL_MIN = 0.80

# Grid search
OVERSAMPLE_GRID = [2, 3, 4]
WEIGHT_GRID     = [3.0, 3.5, 4.0]

TRAIN_ARGS_BASE = dict(
    num_train_epochs        = 8,      # tăng từ 6 (v5) lên 8
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
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
PATIENCE = 3   # tăng từ 2 (v5) lên 3


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

    # S3 subset stats
    def s3_rows(records):
        return [
            r for r in records
            if str(r.get("promise_status","")).strip()  == "Yes"
            and str(r.get("evidence_status","")).strip() == "Yes"
            and str(r.get("evidence_quality","")).strip() in S3_LABEL2ID
        ]

    train_s3 = s3_rows(all_train)
    test_s3  = s3_rows(test_200)
    dist_train = Counter(str(r.get("evidence_quality","")).strip() for r in train_s3)
    dist_test  = Counter(str(r.get("evidence_quality","")).strip() for r in test_s3)

    print(f"\nData loaded:")
    print(f"  Train (S3 eligible): {len(train_s3)}  {dict(dist_train)}")
    print(f"  Test  (S3 eligible): {len(test_s3)}   {dict(dist_test)}")

    ratio = dist_train.get("Clear", 1) / max(dist_train.get("Not Clear", 1), 1)
    print(f"  Imbalance ratio: {ratio:.2f}x")

    return all_train, test_200


# ==========================================
# DATASET
# ==========================================
class S3Dataset(Dataset):
    def __init__(self, records, tokenizer, oversample_nc=2, oversample_mode=True):
        self.tokenizer = tokenizer
        self.samples   = []

        oversample_map = {"Clear": 1, "Not Clear": oversample_nc}

        for row in records:
            promise  = str(row.get("promise_status",  "")).strip()
            evidence = str(row.get("evidence_status", "")).strip()
            quality  = str(row.get("evidence_quality","")).strip()

            if promise  != "Yes": continue
            if evidence != "Yes": continue
            if quality not in S3_LABEL2ID: continue   # drop Misleading

            label  = S3_LABEL2ID[quality]
            repeat = oversample_map[quality] if oversample_mode else 1
            data   = str(row.get("data", "")).strip()
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
    def __init__(self, weight_nc, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # weight[Clear, Not Clear] = [1.0, weight_nc]
        self.class_weights = [1.0, weight_nc]

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
def train_one_combo(oversample_nc, weight_nc, train_records, test_records,
                    tokenizer, run_dir):
    os.makedirs(run_dir, exist_ok=True)

    train_ds = S3Dataset(train_records, tokenizer, oversample_nc, oversample_mode=True)
    eval_ds  = S3Dataset(test_records,  tokenizer, oversample_nc=1, oversample_mode=False)

    dist = Counter(s["label"] for s in train_ds.samples)
    print(f"    Train: {len(train_ds)} samples  "
          f"Clear={dist[0]} / Not Clear={dist[1]}")
    print(f"    Eval : {len(eval_ds)} samples")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        id2label=S3_ID2LABEL,
        label2id=S3_LABEL2ID,
    )

    args = TrainingArguments(output_dir=run_dir, **TRAIN_ARGS_BASE)

    trainer = WeightedTrainer(
        weight_nc=weight_nc,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    trainer.train()

    preds_out    = trainer.predict(eval_ds)
    preds        = np.argmax(preds_out.predictions, axis=-1)
    labels       = preds_out.label_ids
    report       = classification_report(
        labels, preds,
        target_names=list(S3_ID2LABEL.values()),
        output_dict=True,
    )

    not_clear_f1      = report["Not Clear"]["f1-score"]
    not_clear_recall  = report["Not Clear"]["recall"]
    clear_recall      = report["Clear"]["recall"]
    macro_f1          = report["macro avg"]["f1-score"]

    print(f"\n    Results:")
    print(classification_report(labels, preds, target_names=list(S3_ID2LABEL.values())))

    # Save model cho combo này
    trainer.save_model(run_dir)
    tokenizer.save_pretrained(run_dir)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "oversample_nc":   oversample_nc,
        "weight_nc":       weight_nc,
        "effective_boost": round(oversample_nc * weight_nc, 2),
        "not_clear_f1":    round(not_clear_f1,     4),
        "not_clear_recall":round(not_clear_recall, 4),
        "clear_recall":    round(clear_recall,     4),
        "macro_f1":        round(macro_f1,         4),
        "run_dir":         run_dir,
        "passes_constraint": clear_recall >= CLEAR_RECALL_MIN,
    }


# ==========================================
# SELECT BEST
# ==========================================
def select_best(results):
    """
    Ưu tiên:
      1. Clear recall >= CLEAR_RECALL_MIN (hard constraint)
      2. Not Clear F1 cao nhất
      3. Nếu tie: macro_f1 cao hơn
    """
    valid = [r for r in results if r["passes_constraint"]]
    pool  = valid if valid else results   # nếu không ai pass → fallback toàn bộ

    best = max(pool, key=lambda r: (r["not_clear_f1"], r["macro_f1"]))
    return best


# ==========================================
# PRINT GRID SUMMARY
# ==========================================
def print_grid_summary(results):
    print("\n" + "=" * 80)
    print("  GRID SEARCH SUMMARY")
    print("=" * 80)
    header = f"{'OS':>3}  {'W':>4}  {'Boost':>6}  {'NC_F1':>6}  {'NC_Rec':>7}  {'Cl_Rec':>7}  {'MacF1':>6}  {'Pass':>5}"
    print(header)
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["not_clear_f1"], reverse=True):
        flag = "✅" if r["passes_constraint"] else "❌"
        print(
            f"  x{r['oversample_nc']}  "
            f"w{r['weight_nc']:.1f}  "
            f"{r['effective_boost']:>6.1f}x  "
            f"{r['not_clear_f1']:>6.4f}  "
            f"{r['not_clear_recall']:>7.4f}  "
            f"{r['clear_recall']:>7.4f}  "
            f"{r['macro_f1']:>6.4f}  "
            f"{flag}"
        )
    print("=" * 80)


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

    print(f"\nLoading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    total_combos = len(OVERSAMPLE_GRID) * len(WEIGHT_GRID)
    print(f"\nGrid search: {len(OVERSAMPLE_GRID)} oversample × "
          f"{len(WEIGHT_GRID)} weights = {total_combos} combos")
    print(f"Hard constraint: Clear recall >= {CLEAR_RECALL_MIN}\n")

    results = []
    combo_idx = 0

    for oversample_nc in OVERSAMPLE_GRID:
        for weight_nc in WEIGHT_GRID:
            combo_idx += 1
            effective = oversample_nc * weight_nc
            run_name  = f"s3-v6-os{oversample_nc}-w{weight_nc:.1f}"
            run_dir   = os.path.join(OUTPUT_BASE, "grid_s3_v6", run_name)

            print(f"\n[{combo_idx}/{total_combos}] oversample=x{oversample_nc}  "
                  f"weight={weight_nc}  effective={effective:.1f}x")
            print(f"  Dir: {run_dir}")

            result = train_one_combo(
                oversample_nc, weight_nc,
                train_records, test_records,
                tokenizer, run_dir,
            )
            results.append(result)

    # ---- Summary ----
    print_grid_summary(results)

    # ---- Save grid results ----
    grid_json = os.path.join(OUTPUT_BASE, "macbert-large-s3-v6", "grid_results.json")
    os.makedirs(os.path.dirname(grid_json), exist_ok=True)
    with open(grid_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nGrid results saved: {grid_json}")

    # ---- Copy best to final output ----
    best = select_best(results)
    print(f"\n{'='*60}")
    print(f"  Best combo:")
    print(f"    oversample    = x{best['oversample_nc']}")
    print(f"    weight        = {best['weight_nc']}")
    print(f"    effective     = {best['effective_boost']}x")
    print(f"    Not Clear F1  = {best['not_clear_f1']}")
    print(f"    Not Clear Rec = {best['not_clear_recall']}")
    print(f"    Clear Rec     = {best['clear_recall']}  "
          f"({'✅ pass' if best['passes_constraint'] else '❌ fail'})")
    print(f"    Macro F1      = {best['macro_f1']}")
    print(f"    Source dir    = {best['run_dir']}")

    # Copy best model files to final v6 dir
    import shutil
    final_dir = os.path.join(OUTPUT_BASE, "macbert-large-s3-v6")
    os.makedirs(final_dir, exist_ok=True)

    for fname in os.listdir(best["run_dir"]):
        src = os.path.join(best["run_dir"], fname)
        dst = os.path.join(final_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    # Save best config
    best_cfg = {**best, "final_dir": final_dir}
    with open(os.path.join(final_dir, "train_metrics_v6.json"), "w", encoding="utf-8") as f:
        json.dump(best_cfg, f, indent=2, ensure_ascii=False)

    print(f"\n  ✅ Best model copied to: {final_dir}")
    print(f"{'='*60}")
    print(f"\nNext: update main_pipeline_v5.py (hoặc v6):")
    print(f'  "s3": r"{final_dir}"')