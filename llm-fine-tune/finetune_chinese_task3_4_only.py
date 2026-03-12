import json
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, classification_report
import random
import os
import gc

# ==========================================
# CONFIG
# ==========================================
JSON_FILE   = r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_1000 V1.json"
OUTPUT_BASE = r"D:\LLMs\BERT-ESG"

BERT_MODELS = [
    {
        "name":       "macbert-large",
        "model_id":   "hfl/chinese-macbert-large",
        "max_length": 512,
        "batch_size": 8,
        "epochs":     10,
    },
    {
        "name":       "roberta-wwm-large",
        "model_id":   "hfl/chinese-roberta-wwm-ext-large",
        "max_length": 512,
        "batch_size": 8,
        "epochs":     10,
    },
]

# ==========================================
# LABEL MAPS
# ==========================================
S3_LABELS   = {"Clear": 0, "Not Clear": 1, "Misleading": 2, "N/A": 3}
S3_ID2LABEL = {0: "Clear", 1: "Not Clear", 2: "Misleading", 3: "N/A"}

S4_LABELS   = {
    "already":                  0,
    "within_2_years":           1,
    "between_2_and_5_years":    2,
    "longer_than_5_years":      3,
}
S4_ID2LABEL = {v: k for k, v in S4_LABELS.items()}

# ==========================================
# LOAD & SPLIT
# ==========================================
def load_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    random.seed(42)
    random.shuffle(records)
    return records[:800], records[800:]

# ==========================================
# PYTORCH DATASET
# ==========================================
class ESGDataset(Dataset):
    def __init__(self, records, tokenizer, task, max_length=512, oversample=True):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = []

        for row in records:
            data             = str(row["data"]).strip()
            promise_status   = str(row["promise_status"]).strip()
            promise_string   = str(row.get("promise_string") or "").strip()
            evidence_string  = str(row.get("evidence_string") or "").strip()
            evidence_quality = str(row.get("evidence_quality") or "N/A").strip()
            timeline         = str(row.get("verification_timeline") or "").strip()

            # S3 và S4 chỉ áp dụng khi promise = Yes
            if promise_status != "Yes":
                continue

            # Thêm hint từ promise_string
            hint_promise  = f" 關鍵承諾：{promise_string}" if promise_string else ""
            hint_evidence = f" 相關證據：{evidence_string}" if evidence_string else ""

            if task == "s3":
                if evidence_quality not in S3_LABELS:
                    continue
                label = S3_LABELS[evidence_quality]
                text  = data + hint_promise + hint_evidence

                # Oversample minority classes
                if oversample:
                    if evidence_quality == "Not Clear":
                        repeat = 4
                    elif evidence_quality == "Misleading":
                        repeat = 20
                    else:
                        repeat = 1
                else:
                    repeat = 1

                for _ in range(repeat):
                    self.samples.append({"text": text, "label": label})

            elif task == "s4":
                if timeline not in S4_LABELS:
                    continue
                label = S4_LABELS[timeline]
                text  = data + hint_promise

                # Oversample within_2_years
                if oversample:
                    repeat = 15 if timeline == "within_2_years" else 1
                else:
                    repeat = 1

                for _ in range(repeat):
                    self.samples.append({"text": text, "label": label})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item     = self.samples[idx]
        encoding = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels":         torch.tensor(item["label"], dtype=torch.long)
        }

# ==========================================
# METRICS
# ==========================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds)}

# ==========================================
# TRAIN 1 BERT MODEL FOR 1 TASK
# ==========================================
def train_bert(bert_cfg, task, train_records, test_records):
    id2label   = S3_ID2LABEL if task == "s3" else S4_ID2LABEL
    label2id   = S3_LABELS   if task == "s3" else S4_LABELS
    num_labels = len(id2label)
    output_dir = os.path.join(OUTPUT_BASE, f"{bert_cfg['name']}-{task}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Training: {bert_cfg['name']} — {task.upper()}")
    print(f"  Model:    {bert_cfg['model_id']}")
    print(f"  Labels:   {list(id2label.values())}")
    print(f"  Output:   {output_dir}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(bert_cfg["model_id"])
    model = AutoModelForSequenceClassification.from_pretrained(
        bert_cfg["model_id"],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    train_dataset = ESGDataset(train_records, tokenizer, task,
                               max_length=bert_cfg["max_length"], oversample=True)
    eval_dataset  = ESGDataset(test_records,  tokenizer, task,
                               max_length=bert_cfg["max_length"], oversample=False)

    # Label distribution
    from collections import Counter
    dist = Counter(s["label"] for s in train_dataset.samples)
    print(f"  Train samples: {len(train_dataset.samples)}")
    for lid, cnt in sorted(dist.items()):
        print(f"    {id2label[lid]:30s}: {cnt}")
    print(f"  Eval samples:  {len(eval_dataset.samples)}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=bert_cfg["epochs"],
        per_device_train_batch_size=bert_cfg["batch_size"],
        per_device_eval_batch_size=bert_cfg["batch_size"],
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_steps=10,
        bf16=True,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Saved: {output_dir}")

    # Final report
    print(f"\n  Final evaluation:")
    preds_output = trainer.predict(eval_dataset)
    preds  = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids
    print(classification_report(
        labels, preds,
        target_names=[id2label[i] for i in range(num_labels)]
    ))

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  VRAM cleared\n")

    return output_dir


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    train_records, test_records = load_data(JSON_FILE)
    print(f"Train: {len(train_records)} | Test: {len(test_records)}")

    results = []

    for bert_cfg in BERT_MODELS:
        for task in ["s3", "s4"]:       # Chỉ train S3, S4
            path = train_bert(bert_cfg, task, train_records, test_records)
            results.append({
                "model": bert_cfg["name"],
                "task":  task,
                "path":  path
            })

    print("\n" + "="*60)
    print("All BERT models trained!")
    print("="*60)
    for r in results:
        print(f"  {r['model']:25s} {r['task'].upper()} → {r['path']}")