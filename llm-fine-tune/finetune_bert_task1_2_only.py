import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, classification_report
import random
import os

# ==========================================
# CONFIG
# ==========================================
JSON_FILE   = r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_1000 V1.json"
OUTPUT_BASE = r"D:\LLMs\BERT-ESG"

# Các models Chinese BERT để thử
BERT_MODELS = [
    {
        "name": "macbert-large",
        "model_id": "hfl/chinese-macbert-large",
        "max_length": 512,
        "batch_size": 8,
        "epochs": 10,
    },
    {
        "name": "roberta-wwm-large",
        "model_id": "hfl/chinese-roberta-wwm-ext-large",
        "max_length": 512,
        "batch_size": 8,
        "epochs": 10,
    },
    # Uncomment để thử thêm
    # {
    #     "name": "macbert-base",
    #     "model_id": "hfl/chinese-macbert-base",
    #     "max_length": 512,
    #     "batch_size": 16,
    #     "epochs": 10,
    # },
]

# ==========================================
# LABEL MAPS
# ==========================================
S1_LABELS = {"No": 0, "Yes": 1}
S2_LABELS = {"No": 0, "Yes": 1}
S1_ID2LABEL = {0: "No", 1: "Yes"}
S2_ID2LABEL = {0: "No", 1: "Yes"}

# ==========================================
# LOAD & SPLIT DATA
# ==========================================
def load_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    random.seed(42)
    random.shuffle(records)
    train = records[:800]
    test  = records[800:]
    return train, test

# ==========================================
# PYTORCH DATASET
# ==========================================
class ESGDataset(Dataset):
    def __init__(self, records, tokenizer, task, max_length=512, oversample=True):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.samples    = []

        for row in records:
            data           = str(row["data"]).strip()
            promise_status = str(row["promise_status"]).strip()

            if task == "s1":
                label = S1_LABELS[promise_status]
                repeat = 3 if promise_status == "No" else 1  # oversample No
                for _ in range(repeat if oversample else 1):
                    self.samples.append({"text": data, "label": label})

            elif task == "s2":
                if promise_status != "Yes":
                    continue
                evidence_status = str(row.get("evidence_status", "No")).strip()
                if evidence_status not in S2_LABELS:
                    continue
                label = S2_LABELS[evidence_status]
                repeat = 2 if evidence_status == "No" else 1  # oversample No
                for _ in range(repeat if oversample else 1):
                    self.samples.append({"text": data, "label": label})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
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
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# ==========================================
# TRAIN 1 BERT MODEL FOR 1 TASK
# ==========================================
def train_bert(bert_cfg, task, train_records, test_records):
    task_upper = task.upper()
    id2label   = S1_ID2LABEL if task == "s1" else S2_ID2LABEL
    label2id   = S1_LABELS   if task == "s1" else S2_LABELS
    num_labels = 2

    output_dir = os.path.join(OUTPUT_BASE, f"{bert_cfg['name']}-{task}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training: {bert_cfg['name']} — {task_upper}")
    print(f"Model:    {bert_cfg['model_id']}")
    print(f"Output:   {output_dir}")
    print(f"{'='*60}")

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(bert_cfg["model_id"])
    model = AutoModelForSequenceClassification.from_pretrained(
        bert_cfg["model_id"],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )

    # Build datasets
    train_dataset = ESGDataset(train_records, tokenizer, task,
                               max_length=bert_cfg["max_length"], oversample=True)
    eval_dataset  = ESGDataset(test_records,  tokenizer, task,
                               max_length=bert_cfg["max_length"], oversample=False)

    # Label distribution
    from collections import Counter
    label_counts = Counter(s["label"] for s in train_dataset.samples)
    print(f"  Train samples: {len(train_dataset.samples)}")
    print(f"  Label dist:    {dict(label_counts)}")
    print(f"  Eval samples:  {len(eval_dataset.samples)}")

    # Training args
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

    # Save best model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved: {output_dir}")

    # Final evaluation
    print(f"\nFinal evaluation on test set:")
    preds_output = trainer.predict(eval_dataset)
    preds = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids
    print(classification_report(
        labels, preds,
        target_names=list(id2label.values())
    ))

    # Cleanup
    del model
    del trainer
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"VRAM cleared\n")

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
        for task in ["s1", "s2"]:
            output_dir = train_bert(bert_cfg, task, train_records, test_records)
            results.append({
                "model": bert_cfg["name"],
                "task":  task,
                "path":  output_dir
            })

    print("\n" + "="*60)
    print("All BERT models trained!")
    print("="*60)
    for r in results:
        print(f"  {r['model']:25s} {r['task'].upper()} → {r['path']}")

    print("\nNext: load these models in council pipeline")
    print("  from transformers import pipeline")
    print("  s1_classifier = pipeline('text-classification', model=path_s1, device=0)")
    print("  s2_classifier = pipeline('text-classification', model=path_s2, device=0)")