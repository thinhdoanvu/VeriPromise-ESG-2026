import pandas as pd
import numpy as np
import os
from datetime import datetime
from transformers import pipeline
import torch

# ==========================================
# CONFIG
# ==========================================
TEST_CSV   = r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_1000 V1_test200.csv"
OUTPUT_DIR = r"C:\Users\VU\Documents\NLP\AICup26\results"

DEVICE = 0 if torch.cuda.is_available() else -1

BERT_MODELS = [
    {
        "name":    "macbert-large-s1",
        "task":    "s1",
        "path":    r"D:\LLMs\BERT-ESG\macbert-large-s1",
    },
    {
        "name":    "macbert-large-s2",
        "task":    "s2",
        "path":    r"D:\LLMs\BERT-ESG\macbert-large-s2",
    },
    {
        "name":    "roberta-wwm-large-s1",
        "task":    "s1",
        "path":    r"D:\LLMs\BERT-ESG\roberta-wwm-large-s1",
    },
    {
        "name":    "roberta-wwm-large-s2",
        "task":    "s2",
        "path":    r"D:\LLMs\BERT-ESG\roberta-wwm-large-s2",
    },
]

# ==========================================
# EVALUATE 1 BERT MODEL
# ==========================================
def evaluate_bert(bert_cfg, df):
    print(f"\n{'='*60}")
    print(f"  Evaluating: {bert_cfg['name']}")
    print(f"  Task:       {bert_cfg['task'].upper()}")
    print(f"  Path:       {bert_cfg['path']}")
    print(f"{'='*60}")

    # Load model
    classifier = pipeline(
        "text-classification",
        model=bert_cfg["path"],
        tokenizer=bert_cfg["path"],
        device=DEVICE,
        truncation=True,
        max_length=512,
    )

    task      = bert_cfg["task"]
    results   = []
    total     = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{total}] Processing...")

        data        = str(row["data"]).strip()
        promise_gt  = str(row["promise_status"]).strip()
        evidence_gt = str(row["evidence_status"]).strip() if pd.notna(row["evidence_status"]) else "No"

        if task == "s1":
            # Predict
            out    = classifier(data)[0]
            pred   = out["label"]
            score  = out["score"]
            gt     = promise_gt
            correct = (pred == gt)

            results.append({
                "id":       row["id"],
                "model":    bert_cfg["name"],
                "task":     "s1",
                "gt":       gt,
                "pred":     pred,
                "score":    round(score, 4),
                "correct":  correct,
            })

        elif task == "s2":
            # Chỉ evaluate khi promise = Yes
            if promise_gt != "Yes":
                results.append({
                    "id":       row["id"],
                    "model":    bert_cfg["name"],
                    "task":     "s2",
                    "gt":       "SKIP",
                    "pred":     "SKIP",
                    "score":    None,
                    "correct":  None,
                })
                continue

            out    = classifier(data)[0]
            pred   = out["label"]
            score  = out["score"]
            gt     = evidence_gt
            correct = (pred == gt)

            results.append({
                "id":       row["id"],
                "model":    bert_cfg["name"],
                "task":     "s2",
                "gt":       gt,
                "pred":     pred,
                "score":    round(score, 4),
                "correct":  correct,
            })

    # Cleanup
    del classifier
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    return pd.DataFrame(results)


# ==========================================
# PRINT ACCURACY REPORT
# ==========================================
def print_accuracy(df_results, model_name, task):
    print(f"\n{'='*60}")
    print(f"  ACCURACY REPORT: {model_name} ({task.upper()})")
    print(f"{'='*60}")

    valid = df_results[df_results["correct"].notna()]
    if len(valid) == 0:
        print("  No valid predictions!")
        return 0

    correct = int(valid["correct"].sum())
    count   = len(valid)
    acc     = correct / count * 100
    print(f"  Overall: {correct}/{count} = {acc:.1f}%")

    # Per-label breakdown
    for gt_label in sorted(valid["gt"].unique()):
        if gt_label in ["SKIP", ""]:
            continue
        mask = valid["gt"] == gt_label
        lc   = int(valid[mask]["correct"].sum())
        lct  = int(mask.sum())
        print(f"    L {gt_label:10s}: {lc:3d}/{lct} = {lc/lct*100:.1f}%")

    # Confusion
    print(f"\n  Confusion breakdown:")
    for gt_label in sorted(valid["gt"].unique()):
        if gt_label in ["SKIP", ""]:
            continue
        for pred_label in sorted(valid["pred"].unique()):
            if pred_label in ["SKIP", ""]:
                continue
            mask = (valid["gt"] == gt_label) & (valid["pred"] == pred_label)
            count_m = int(mask.sum())
            if count_m > 0:
                marker = "OK" if gt_label == pred_label else "X "
                print(f"    [{marker}] GT={gt_label:10s} PRED={pred_label:10s}: {count_m}")

    print(f"{'='*60}")
    return acc


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.read_csv(TEST_CSV)
    print(f"Test set: {len(df)} rows")
    print(f"Device:   {'GPU' if DEVICE == 0 else 'CPU'}")

    summary = []

    for bert_cfg in BERT_MODELS:
        df_results = evaluate_bert(bert_cfg, df)

        # Save CSV
        out_csv = os.path.join(OUTPUT_DIR, f"{bert_cfg['name']}_{timestamp}.csv")
        df_results.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"  CSV saved: {out_csv}")

        # Print accuracy
        acc = print_accuracy(df_results, bert_cfg["name"], bert_cfg["task"])
        summary.append({
            "model": bert_cfg["name"],
            "task":  bert_cfg["task"],
            "acc":   acc
        })

    # Final summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':30s} {'Task':5s} {'Accuracy':>10s}")
    print(f"  {'-'*50}")
    for s in sorted(summary, key=lambda x: (x["task"], -x["acc"])):
        print(f"  {s['model']:30s} {s['task'].upper():5s} {s['acc']:>9.1f}%")

    # Best per task
    print(f"\n  Best models:")
    for task in ["s1", "s2"]:
        best = max([s for s in summary if s["task"] == task], key=lambda x: x["acc"])
        print(f"  {task.upper()}: {best['model']:30s} → {best['acc']:.1f}%")
    print(f"{'='*60}")

# Train xong thì xóa cache C:\Users\VU\.cache\huggingface