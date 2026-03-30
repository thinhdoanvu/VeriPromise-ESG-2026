"""
Train BERT S3 — Binary Classification: Clear vs Not Clear
==========================================================
- Input : cột `data` (chỉ rows promise=Yes AND evidence=Yes)
- Output: Clear / Not Clear
- N/A   → pipeline tự xử lý (S2=No), không cần predict
- Misleading → chấp nhận bỏ qua (chỉ 1 case)
- Oversample Not Clear ×4 để cân bằng class
"""

import json
import os
import random
import gc
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from torch.utils.data import Dataset

# ==========================================
# CONFIG
# ==========================================
JSON_FILE   = r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_1000 V1.json"
OUTPUT_DIR  = r"D:\LLMs\BERT-ESG\macbert-large-s3"

MODEL_ID    = "hfl/chinese-macbert-large"
MAX_LENGTH  = 512
BATCH_SIZE  = 8
EPOCHS      = 10
LR          = 2e-5

# Binary labels — Misleading và N/A bỏ qua
LABEL2ID    = {"Clear": 0, "Not Clear": 1}
ID2LABEL    = {0: "Clear", 1: "Not Clear"}
OVERSAMPLE_NOT_CLEAR = 8   # v2: best result
SEED = 42

# ==========================================
# FIX RANDOM SEED — đảm bảo reproduce được
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)

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
class S3Dataset(Dataset):
    def __init__(self, records, tokenizer, oversample=True):
        self.tokenizer = tokenizer
        self.samples   = []

        for row in records:
            promise  = str(row.get("promise_status", "")).strip()
            evidence = str(row.get("evidence_status", "")).strip()
            quality  = str(row.get("evidence_quality", "")).strip()

            # Chỉ lấy rows: promise=Yes AND evidence=Yes AND label là Clear/Not Clear
            if promise  != "Yes":      continue
            if evidence != "Yes":      continue
            if quality  not in LABEL2ID: continue   # bỏ Misleading, N/A

            data   = str(row["data"]).strip()
            label  = LABEL2ID[quality]
            repeat = OVERSAMPLE_NOT_CLEAR if (oversample and quality == "Not Clear") else 1

            for _ in range(repeat):
                self.samples.append({"text": data, "label": label})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item     = self.samples[idx]
        encoding = self.tokenizer(
            item["text"],
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels":         torch.tensor(item["label"], dtype=torch.long)
        }

import torch.nn as nn

# ==========================================
# WEIGHTED TRAINER — phạt nặng khi miss Not Clear
# ==========================================
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.get("labels")
        outputs = model(**inputs)
        logits  = outputs.get("logits")

        # Clear=0 weight=1.0 / Not Clear=1 weight=3.0
        weights = torch.tensor([1.0, 3.0], device=logits.device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        loss    = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss
def compute_metrics(eval_pred):
    from sklearn.metrics import f1_score
    logits, labels = eval_pred
    preds   = np.argmax(logits, axis=-1)
    acc     = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "macro_f1": macro_f1}

# ==========================================
# TRAIN
# ==========================================
def train():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training: MacBERT-large — S3 Binary (Clear / Not Clear)")
    print(f"Model:    {MODEL_ID}")
    print(f"Output:   {OUTPUT_DIR}")
    print(f"{'='*60}")

    # Load data
    train_records, test_records = load_data(JSON_FILE)
    print(f"Train records: {len(train_records)} | Test records: {len(test_records)}")

    # Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Datasets
    train_dataset = S3Dataset(train_records, tokenizer, oversample=True)
    eval_dataset  = S3Dataset(test_records,  tokenizer, oversample=False)

    # Label distribution
    train_dist = Counter(s["label"] for s in train_dataset.samples)
    eval_dist  = Counter(s["label"] for s in eval_dataset.samples)
    print(f"\n  Train samples: {len(train_dataset)}")
    print(f"  Label dist:    { {ID2LABEL[k]: v for k, v in train_dist.items()} }")
    print(f"  Eval samples:  {len(eval_dataset)}")
    print(f"  Label dist:    { {ID2LABEL[k]: v for k, v in eval_dist.items()} }")

    # Training args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=10,
        bf16=True,
        seed=SEED,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()

    # Save best model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nSaved → {OUTPUT_DIR}")

    # Final evaluation
    print(f"\nFinal evaluation on test set:")
    preds_output = trainer.predict(eval_dataset)
    preds  = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids
    print(classification_report(labels, preds, target_names=["Clear", "Not Clear"]))

    # Cleanup
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    print("VRAM cleared.")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    train()
    print("\nNext step: cập nhật main_pipeline.py dùng BERT S3 thay LLM Council")
    print(f"  model path: {OUTPUT_DIR}")