"""
Train BERT S4 — 4-class verification_timeline (v1)
======================================================
Thay the LLM Council cho S4. BERT hieu ngu canh tot hon LLM cho
task nay (xem phan tich: LLM ceiling ~0.487 do "vague" cases
khong co pattern ro rang trong text, nhung BERT co the hoc
decision boundary tu data ma khong can "giai thich duoc ly do").

4 classes (sau normalize longer_than_5_years -> more_than_5_years):
  already                : 644/1459 = 44.1%
  between_2_and_5_years  : 437/1459 = 30.0%
  more_than_5_years      : 347/1459 = 23.8%
  within_2_years         :  31/1459 =  2.1%  <- imbalance 20.77x

Grid search: 4 configs, varying oversample cho within_2_years
(class kho nhat) + more_than_5_years/between_2_and_5_years/already.
Class weights = balanced (tu dong tinh tu effective count sau oversample).

Hard constraints (tranh sup do majority classes):
  already_recall            >= 0.60
  between_2_and_5_years_recall >= 0.40

Metric chinh: Macro F1 (tren 4 classes)

Output:
  D:\\LLMs\\BERT-ESG\\macbert-large-s4-v1\\        <- best combo
  D:\\LLMs\\BERT-ESG\\macbert-large-s4-v1\\grid_results.json

Cach dung:
  python train_bert_s4_v1.py
"""

import json
import os
import random
import gc
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

S4_LABEL2ID = {
    "already":               0,
    "within_2_years":        1,
    "between_2_and_5_years": 2,
    "more_than_5_years":     3,
}
S4_ID2LABEL = {v: k for k, v in S4_LABEL2ID.items()}
VALID_S4    = list(S4_LABEL2ID.keys())

# Hard constraints
ALREADY_RECALL_MIN = 0.60
BETWEEN_RECALL_MIN = 0.40

TRAIN_ARGS_BASE = dict(
    num_train_epochs        = 8,
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
PATIENCE = 3

# ==========================================
# GRID — 4 configs: oversample multiplier per class
# (already, within_2_years, between_2_and_5_years, more_than_5_years)
# ==========================================
GRID = [
    {"name": "s4-v1-a", "oversample": {"already": 1.0, "within_2_years": 6,  "between_2_and_5_years": 1.0, "more_than_5_years": 1.0}},
    {"name": "s4-v1-b", "oversample": {"already": 1.0, "within_2_years": 10, "between_2_and_5_years": 1.0, "more_than_5_years": 1.0}},
    {"name": "s4-v1-c", "oversample": {"already": 0.8, "within_2_years": 10, "between_2_and_5_years": 1.2, "more_than_5_years": 1.5}},
    {"name": "s4-v1-d", "oversample": {"already": 0.8, "within_2_years": 15, "between_2_and_5_years": 1.2, "more_than_5_years": 1.5}},
]


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

    def s4_rows(records):
        return [
            r for r in records
            if str(r.get("promise_status","")).strip()  == "Yes"
            and str(r.get("verification_timeline","")).strip() in VALID_S4
        ]

    train_s4 = s4_rows(all_train)
    test_s4  = s4_rows(test_200)
    dist_train = Counter(str(r.get("verification_timeline","")).strip() for r in train_s4)
    dist_test  = Counter(str(r.get("verification_timeline","")).strip() for r in test_s4)

    print(f"\nData loaded:")
    print(f"  Train (S4 eligible): {len(train_s4)}  {dict(dist_train)}")
    print(f"  Test  (S4 eligible): {len(test_s4)}   {dict(dist_test)}")

    return train_s4, test_s4


# ==========================================
# DATASET
# ==========================================
class S4Dataset(Dataset):
    def __init__(self, records, tokenizer, oversample_map=None, apply_oversample=True):
        self.tokenizer = tokenizer
        self.samples   = []
        oversample_map = oversample_map or {}

        # Group by label first (de ho tro fractional oversample, vd 0.8x)
        by_label = {label: [] for label in VALID_S4}
        for row in records:
            label = str(row.get("verification_timeline","")).strip()
            if label not in VALID_S4:
                continue
            data = str(row.get("data","")).strip()
            by_label[label].append({"text": data, "label": S4_LABEL2ID[label]})

        for label, items in by_label.items():
            if not apply_oversample:
                self.samples.extend(items)
                continue
            ratio = oversample_map.get(label, 1.0)
            if ratio >= 1.0:
                full_repeats = int(ratio)
                frac = ratio - full_repeats
                for _ in range(full_repeats):
                    self.samples.extend(items)
                if frac > 0:
                    n_extra = int(len(items) * frac)
                    self.samples.extend(items[:n_extra])
            else:
                # undersample: giu lai ratio% (deterministic, da shuffle tu load_combined_data)
                n_keep = max(1, int(len(items) * ratio))
                self.samples.extend(items[:n_keep])

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
# WEIGHTED TRAINER — class weights = balanced (tu effective counts)
# ==========================================
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.get("logits")
        weights = torch.tensor(
            self.class_weights, dtype=torch.float, device=logits.device
        )
        loss = nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_balanced_weights(samples):
    """weight_c = total / (n_classes * count_c), normalized so min(weight)=1.0"""
    counts = Counter(s["label"] for s in samples)
    total  = len(samples)
    n_cls  = len(VALID_S4)
    raw = [total / (n_cls * counts.get(i, 1)) for i in range(n_cls)]
    min_w = min(raw)
    return [w / min_w for w in raw]


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
def train_one_combo(cfg, train_records, test_records, tokenizer, run_dir):
    os.makedirs(run_dir, exist_ok=True)

    oversample_map = cfg["oversample"]
    train_ds = S4Dataset(train_records, tokenizer, oversample_map, apply_oversample=True)
    eval_ds  = S4Dataset(test_records,  tokenizer, oversample_map=None, apply_oversample=False)

    dist = Counter(s["label"] for s in train_ds.samples)
    dist_named = {S4_ID2LABEL[k]: v for k, v in dist.items()}
    print(f"    Train: {len(train_ds)} samples  {dist_named}")
    print(f"    Eval : {len(eval_ds)} samples")

    class_weights = compute_balanced_weights(train_ds.samples)
    print(f"    Class weights (balanced): "
          f"{dict(zip(VALID_S4, [round(w,2) for w in class_weights]))}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=len(VALID_S4),
        id2label=S4_ID2LABEL,
        label2id=S4_LABEL2ID,
    )

    args = TrainingArguments(output_dir=run_dir, **TRAIN_ARGS_BASE)

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    trainer.train()

    preds_out = trainer.predict(eval_ds)
    preds     = np.argmax(preds_out.predictions, axis=-1)
    labels    = preds_out.label_ids
    report    = classification_report(
        labels, preds,
        target_names=VALID_S4,
        labels=list(range(len(VALID_S4))),
        output_dict=True,
        zero_division=0,
    )

    macro_f1 = report["macro avg"]["f1-score"]
    per_class = {
        label: {
            "f1":     round(report[label]["f1-score"], 4),
            "recall": round(report[label]["recall"], 4),
            "precision": round(report[label]["precision"], 4),
            "support": int(report[label]["support"]),
        }
        for label in VALID_S4
    }

    print(f"\n    Results:")
    print(classification_report(labels, preds, target_names=VALID_S4,
                                  labels=list(range(len(VALID_S4))), zero_division=0))

    trainer.save_model(run_dir)
    tokenizer.save_pretrained(run_dir)

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    already_recall = per_class["already"]["recall"]
    between_recall = per_class["between_2_and_5_years"]["recall"]
    passes = (already_recall >= ALREADY_RECALL_MIN
              and between_recall >= BETWEEN_RECALL_MIN)

    return {
        "name":            cfg["name"],
        "oversample":      oversample_map,
        "class_weights":   [round(w, 3) for w in class_weights],
        "macro_f1":        round(macro_f1, 4),
        "accuracy":        round(report["accuracy"], 4) if "accuracy" in report else None,
        "per_class":       per_class,
        "already_recall":  already_recall,
        "between_recall":  between_recall,
        "passes_constraint": passes,
        "run_dir":         run_dir,
    }


# ==========================================
# SELECT BEST
# ==========================================
def select_best(results):
    valid = [r for r in results if r["passes_constraint"]]
    pool  = valid if valid else results
    return max(pool, key=lambda r: r["macro_f1"])


def print_grid_summary(results):
    print("\n" + "=" * 100)
    print("  GRID SEARCH SUMMARY")
    print("=" * 100)
    header = (f"{'Name':<10} {'MacroF1':>8} {'Acc':>6}  "
              f"{'already_R':>10} {'within2_R':>10} {'btw25_R':>9} {'more5_R':>9}  {'Pass'}")
    print(header)
    print("-" * 100)
    for r in sorted(results, key=lambda x: x["macro_f1"], reverse=True):
        pc = r["per_class"]
        flag = "OK" if r["passes_constraint"] else "X"
        print(f"{r['name']:<10} {r['macro_f1']:>8.4f} {r['accuracy']:>6.4f}  "
              f"{pc['already']['recall']:>10.4f} "
              f"{pc['within_2_years']['recall']:>10.4f} "
              f"{pc['between_2_and_5_years']['recall']:>9.4f} "
              f"{pc['more_than_5_years']['recall']:>9.4f}  {flag}")
    print("=" * 100)


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

    print(f"\nGrid search: {len(GRID)} configs")
    print(f"Hard constraints: already_recall>={ALREADY_RECALL_MIN}, "
          f"between_2_and_5_years_recall>={BETWEEN_RECALL_MIN}\n")

    results = []
    for i, cfg in enumerate(GRID, 1):
        run_dir = os.path.join(OUTPUT_BASE, "grid_s4_v1", cfg["name"])
        print(f"\n[{i}/{len(GRID)}] {cfg['name']}  oversample={cfg['oversample']}")
        print(f"  Dir: {run_dir}")
        result = train_one_combo(cfg, train_records, test_records, tokenizer, run_dir)
        results.append(result)

    print_grid_summary(results)

    grid_json = os.path.join(OUTPUT_BASE, "macbert-large-s4-v1", "grid_results.json")
    os.makedirs(os.path.dirname(grid_json), exist_ok=True)
    with open(grid_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nGrid results saved: {grid_json}")

    best = select_best(results)
    print(f"\n{'='*60}")
    print(f"  Best combo: {best['name']}")
    print(f"    oversample = {best['oversample']}")
    print(f"    Macro F1   = {best['macro_f1']}")
    print(f"    already_recall = {best['already_recall']}  "
          f"({'OK' if best['already_recall']>=ALREADY_RECALL_MIN else 'FAIL'})")
    print(f"    between_recall = {best['between_recall']}  "
          f"({'OK' if best['between_recall']>=BETWEEN_RECALL_MIN else 'FAIL'})")
    print(f"    Source dir = {best['run_dir']}")

    import shutil
    final_dir = os.path.join(OUTPUT_BASE, "macbert-large-s4-v1")
    os.makedirs(final_dir, exist_ok=True)
    for fname in os.listdir(best["run_dir"]):
        src = os.path.join(best["run_dir"], fname)
        dst = os.path.join(final_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    with open(os.path.join(final_dir, "train_metrics_s4_v1.json"), "w", encoding="utf-8") as f:
        json.dump({**best, "final_dir": final_dir}, f, indent=2, ensure_ascii=False)

    print(f"\n  Best model copied to: {final_dir}")
    print(f"{'='*60}")
    print(f"\nNext: update main_pipeline_v4.py")
    print(f'  - Them BERT_MODELS["s4"] = r"{final_dir}"')
    print(f'  - Thay predict_s4()/build_prompt_s4()/LLM Council bang')
    print(f'    bert_predict(bert["s4"], data) -> verification_timeline')
    print(f'  - VAN giu rule_based_s4() nhu safety net (override "already"')
    print(f'    -> nam tuong lai + commit-verb, ap dung CHO CA BERT result)')