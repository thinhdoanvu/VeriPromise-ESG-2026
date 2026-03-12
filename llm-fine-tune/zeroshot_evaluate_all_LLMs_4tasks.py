import pandas as pd
import requests
import json
import re
import os
from datetime import datetime

# ==========================================
# CONFIG
# ==========================================
TEST_CSV = r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_1000 V1_test200.csv"
OUTPUT_DIR = r"C:\Users\VU\Documents\NLP\AICup26\results"
OLLAMA_URL = "http://localhost:11434/api/generate"

MODELS = [
    {"name": "qwen2.5-14b-esg",     "role": "voter"},
    {"name": "qwen2.5-72b-esg",     "role": "voter"},
    {"name": "deepseek-r1-70b-esg", "role": "voter"},
    {"name": "qwen2.5-32b-esg",     "role": "chairman"},
]

# Valid labels cho từng subtask
VALID_LABELS = {
    "s1": ["Yes", "No"],
    "s2": ["Yes", "No"],
    "s3": ["Clear", "Not Clear", "Misleading", "N/A"],
    "s4": ["already", "within_2_years", "between_2_and_5_years", "longer_than_5_years"],
}

# ==========================================
# OLLAMA INFERENCE
# ==========================================
def ollama_predict(model_name, prompt, valid_labels):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 20,
            "top_p": 1,
        }
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()

        # Extract valid label từ response
        for label in valid_labels:
            if label.lower() in raw.lower():
                return label, raw
        return "UNKNOWN", raw

    except Exception as e:
        return "ERROR", str(e)


# ==========================================
# BUILD PROMPTS — 4 Subtasks
# ==========================================
def build_prompts(row):
    data = str(row["data"]).strip()
    prompts = {}

    # Subtask 1
    prompts["s1"] = (
        "You are an ESG analyst. "
        "Does the following ESG statement express a concrete corporate commitment or promise toward future actions?\n"
        "Answer only: Yes or No\n\n"
        f"Statement: {data}\n"
        "Answer:"
    )

    # Subtask 2 — chỉ khi promise = Yes
    prompts["s2"] = (
        "You are an ESG analyst. "
        "Is the following ESG commitment supported by concrete evidence such as action plans, data, methodologies, or implementation records?\n"
        "Answer only: Yes or No\n\n"
        f"Statement: {data}\n"
        "Answer:"
    )

    # Subtask 3 — chỉ khi promise = Yes
    prompts["s3"] = (
        "You are an ESG analyst. "
        "Evaluate whether the following ESG statement contains semantically clear evidence.\n"
        "Choose exactly one:\n"
        "- Clear (specific, measurable, verifiable, no vague wording)\n"
        "- Not Clear (vague phrases like 'continuously improving', 'striving to achieve')\n"
        "- Misleading (potentially misleading or greenwashing language)\n"
        "- N/A (no evidence present)\n\n"
        f"Statement: {data}\n"
        "Answer:"
    )

    # Subtask 4 — chỉ khi promise = Yes
    prompts["s4"] = (
        "You are an ESG analyst. The ESG report was published in 2024.\n"
        "Based on the statement, classify the expected completion timeframe of the commitment.\n"
        "Choose exactly one:\n"
        "- already (action already implemented or ongoing in 2024, no future deadline mentioned)\n"
        "- within_2_years (specific target year 2025 or 2026 mentioned)\n"
        "- between_2_and_5_years (target year 2027, 2028, or 2029 mentioned)\n"
        "- longer_than_5_years (target year 2030 or beyond mentioned, e.g. 2030, 2050)\n\n"
        f"Statement: {data}\n"
        "Answer:"
    )

    return prompts


# ==========================================
# EVALUATE 1 MODEL
# ==========================================
def evaluate_model(model_cfg, df):
    model_name = model_cfg["name"]
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name}")
    print(f"{'='*60}")

    results = []
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{total}] Processing...")

        promise_gt  = str(row["promise_status"]).strip()
        evidence_gt = str(row["evidence_status"]).strip() if pd.notna(row["evidence_status"]) else "No"
        quality_gt  = str(row["evidence_quality"]).strip() if pd.notna(row["evidence_quality"]) else "N/A"
        timeline_gt = str(row["verification_timeline"]).strip() if pd.notna(row["verification_timeline"]) else ""

        prompts = build_prompts(row)

        # --- Subtask 1 ---
        s1_pred, s1_raw = ollama_predict(model_name, prompts["s1"], VALID_LABELS["s1"])

        # --- Subtask 2, 3, 4 — chỉ khi promise = Yes ---
        if promise_gt == "Yes":
            s2_pred, s2_raw = ollama_predict(model_name, prompts["s2"], VALID_LABELS["s2"])
            s3_pred, s3_raw = ollama_predict(model_name, prompts["s3"], VALID_LABELS["s3"])
            s4_pred, s4_raw = ollama_predict(model_name, prompts["s4"], VALID_LABELS["s4"]) if timeline_gt else ("N/A", "")
        else:
            s2_pred = s2_raw = "SKIP"
            s3_pred = s3_raw = "SKIP"
            s4_pred = s4_raw = "SKIP"

        results.append({
            "id":           row["id"],
            "model":        model_name,
            # Ground truth
            "s1_gt":        promise_gt,
            "s2_gt":        evidence_gt if promise_gt == "Yes" else "SKIP",
            "s3_gt":        quality_gt  if promise_gt == "Yes" else "SKIP",
            "s4_gt":        timeline_gt if promise_gt == "Yes" and timeline_gt else "SKIP",
            # Predictions
            "s1_pred":      s1_pred,
            "s2_pred":      s2_pred,
            "s3_pred":      s3_pred,
            "s4_pred":      s4_pred,
            # Raw responses
            "s1_raw":       s1_raw,
            "s2_raw":       s2_raw,
            "s3_raw":       s3_raw,
            "s4_raw":       s4_raw,
            # Correct?
            "s1_correct":   s1_pred == promise_gt,
            "s2_correct":   s2_pred == evidence_gt if promise_gt == "Yes" else None,
            "s3_correct":   s3_pred == quality_gt  if promise_gt == "Yes" else None,
            "s4_correct":   s4_pred == timeline_gt if promise_gt == "Yes" and timeline_gt else None,
        })

    return pd.DataFrame(results)


# ==========================================
# PRINT ACCURACY REPORT
# ==========================================
def print_accuracy(df_results, model_name):
    print(f"\n{'='*60}")
    print(f"  ACCURACY REPORT: {model_name}")
    print(f"{'='*60}")

    subtasks = {
        "S1 Commitment":  "s1",
        "S2 Evidence":    "s2",
        "S3 Clarity":     "s3",
        "S4 Timeline":    "s4",
    }

    total_correct = 0
    total_count = 0

    for label, key in subtasks.items():
        col = f"{key}_correct"
        valid = df_results[col].dropna()
        if len(valid) == 0:
            print(f"  {label:20s}: N/A")
            continue
        correct = valid.sum()
        count   = len(valid)
        acc     = correct / count * 100
        total_correct += correct
        total_count   += count
        print(f"  {label:20s}: {correct:3.0f}/{count} = {acc:.1f}%")

        # Per-label breakdown
        sub_df = df_results[df_results[col].notna()]
        for gt_label in sub_df[f"{key}_gt"].unique():
            if gt_label in ["SKIP", ""]:
                continue
            mask = sub_df[f"{key}_gt"] == gt_label
            sub_correct = (sub_df[mask][col]).sum()
            sub_count   = mask.sum()
            print(f"    └ {gt_label:25s}: {sub_correct:3.0f}/{sub_count} = {sub_correct/sub_count*100:.1f}%")

    overall = total_correct / total_count * 100 if total_count > 0 else 0
    print(f"\n  {'Overall':20s}: {total_correct:.0f}/{total_count} = {overall:.1f}%")
    print(f"{'='*60}")

    return overall


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load test set
    df = pd.read_csv(TEST_CSV)
    print(f"Test set loaded: {len(df)} rows")

    summary = []

    for model_cfg in MODELS:
        # Evaluate
        df_results = evaluate_model(model_cfg, df)

        # Save CSV
        out_csv = os.path.join(OUTPUT_DIR, f"{model_cfg['name']}_{timestamp}.csv")
        df_results.to_csv(out_csv, index=False, encoding="utf-8-sig")
        print(f"\n  CSV saved: {out_csv}")

        # Print accuracy
        overall = print_accuracy(df_results, model_cfg["name"])
        summary.append({"model": model_cfg["name"], "overall_acc": overall})

    # Summary
    print(f"\n{'='*60}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*60}")
    for s in sorted(summary, key=lambda x: x["overall_acc"], reverse=True):
        print(f"  {s['model']:30s}: {s['overall_acc']:.1f}%")
    print(f"{'='*60}")