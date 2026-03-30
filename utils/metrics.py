"""
Evaluation script for VeriPromise ESG 2026 competition.
Computes the official competition score:
  Total = Commitment_F1 * 0.20 + Evidence_F1 * 0.30 + Clarity_MacroF1 * 0.35 + Timeline_MacroF1 * 0.15

Ground Truth : vpesg4k_train_1000 V1.json  (test200 subset — last 200 rows after shuffle seed=42)
Predictions  : output TSV từ main_pipeline.py
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

# ==========================================
# PATHS
# ==========================================
BASE_DIR     = r"C:\Users\VU\Documents\NLP\AICup26"
GT_JSON      = os.path.join(BASE_DIR, "datasets", "vpesg4k_train_1000 V1.json")
GT_CSV       = os.path.join(BASE_DIR, "datasets", "vpesg4k_train_1000 V1_test200.csv")
DEFAULT_PRED = os.path.join(BASE_DIR, "results", "predictions_v4_20260325_230010.tsv")

# ==========================================
# LOAD GROUND TRUTH — dùng test200.csv
# ==========================================
def load_ground_truth(path):
    df = pd.read_csv(path, dtype=str)
    df["id"] = df["id"].astype(str)
    cols = ["id", "promise_status", "verification_timeline",
            "evidence_status", "evidence_quality"]
    return df[cols]

# ==========================================
# LOAD PREDICTIONS — TSV từ main_pipeline.py
# ==========================================
def load_predictions(path):
    # Thử TSV trước, fallback sang CSV
    try:
        df = pd.read_csv(path, sep="\t", dtype=str)
        if len(df.columns) == 1:   # nếu chỉ 1 cột → thử CSV
            raise ValueError
    except Exception:
        df = pd.read_csv(path, sep=",", dtype=str)

    df["id"] = df["id"].astype(str)

    # Kiểm tra columns
    required = ["id", "promise_status", "verification_timeline",
                "evidence_status", "evidence_quality"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ Missing columns in predictions: {missing}")
        sys.exit(1)

    return df[required]

# ==========================================
# NORMALIZE
# ==========================================
def normalize(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    val = str(val).strip()
    if val in ("", "nan", "NaN", "None", "none", "null"):
        return "N/A"
    return val

# ==========================================
# METRICS
# ==========================================
def f1_binary(y_true, y_pred, positive="Yes"):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p == positive)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p == positive)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p != positive)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

def macro_f1(y_true, y_pred, classes):
    per_class = {cls: f1_binary(y_true, y_pred, positive=cls) for cls in classes}
    macro     = np.mean([per_class[c]["f1"] for c in classes])
    return macro, per_class

def confusion_matrix_str(y_true, y_pred, classes):
    from collections import Counter
    pairs  = Counter(zip(y_true, y_pred))
    header = f"{'True/Pred':<25}" + "".join(f"{c:<25}" for c in classes)
    lines  = [header, "-" * len(header)]
    for tc in classes:
        row = f"{tc:<25}" + "".join(f"{pairs.get((tc, pc), 0):<25}" for pc in classes)
        lines.append(row)
    return "\n".join(lines)

# ==========================================
# EVALUATE
# ==========================================
def evaluate(gt_path, pred_path):
    df_gt   = load_ground_truth(gt_path)
    df_pred = load_predictions(pred_path)

    # Merge on id
    merged = pd.merge(df_gt, df_pred, on="id", suffixes=("_gt", "_pred"))
    print(f"Matched samples: {len(merged)} (GT: {len(df_gt)}, Pred: {len(df_pred)})")

    if len(merged) == 0:
        print("❌ No matching IDs between ground truth and predictions.")
        sys.exit(1)

    if len(merged) < len(df_gt):
        missing_ids = set(df_gt["id"]) - set(df_pred["id"])
        print(f"⚠️  {len(missing_ids)} GT rows not found in predictions: {list(missing_ids)[:5]}...")

    # Normalize
    for col in ["promise_status", "verification_timeline", "evidence_status", "evidence_quality"]:
        for suffix in ["_gt", "_pred"]:
            merged[col + suffix] = merged[col + suffix].apply(normalize)

    errors = []

    # ==========================================
    # 1. COMMITMENT F1 (20%)
    # ==========================================
    res_s1     = f1_binary(merged["promise_status_gt"].tolist(),
                           merged["promise_status_pred"].tolist(), positive="Yes")
    commitment_f1 = res_s1["f1"]

    print("\n" + "=" * 60)
    print("1. COMMITMENT (promise_status) — Weight: 0.20")
    print("=" * 60)
    print(f"   F1={commitment_f1:.4f}  P={res_s1['precision']:.4f}  R={res_s1['recall']:.4f}")
    print(f"   TP={res_s1['tp']}  FP={res_s1['fp']}  FN={res_s1['fn']}")
    print(f"\n   GT:   {dict(merged['promise_status_gt'].value_counts())}")
    print(f"   Pred: {dict(merged['promise_status_pred'].value_counts())}")
    print(f"\n{confusion_matrix_str(merged['promise_status_gt'].tolist(), merged['promise_status_pred'].tolist(), ['Yes', 'No'])}")

    for _, row in merged.iterrows():
        if row["promise_status_gt"] != row["promise_status_pred"]:
            errors.append({"id": row["id"], "field": "promise_status",
                           "gt": row["promise_status_gt"], "pred": row["promise_status_pred"]})

    # ==========================================
    # 2. EVIDENCE F1 (30%) — GT promise = Yes
    # ==========================================
    ev_subset  = merged[merged["promise_status_gt"] == "Yes"]
    res_s2     = f1_binary(ev_subset["evidence_status_gt"].tolist(),
                           ev_subset["evidence_status_pred"].tolist(), positive="Yes")
    evidence_f1 = res_s2["f1"]

    print("\n" + "=" * 60)
    print("2. EVIDENCE (evidence_status, GT promise=Yes) — Weight: 0.30")
    print("=" * 60)
    print(f"   Evaluated on {len(ev_subset)} samples")
    print(f"   F1={evidence_f1:.4f}  P={res_s2['precision']:.4f}  R={res_s2['recall']:.4f}")
    print(f"   TP={res_s2['tp']}  FP={res_s2['fp']}  FN={res_s2['fn']}")
    print(f"\n   GT:   {dict(ev_subset['evidence_status_gt'].value_counts())}")
    print(f"   Pred: {dict(ev_subset['evidence_status_pred'].value_counts())}")
    print(f"\n{confusion_matrix_str(ev_subset['evidence_status_gt'].tolist(), ev_subset['evidence_status_pred'].tolist(), ['Yes', 'No'])}")

    for _, row in ev_subset.iterrows():
        if row["evidence_status_gt"] != row["evidence_status_pred"]:
            errors.append({"id": row["id"], "field": "evidence_status",
                           "gt": row["evidence_status_gt"], "pred": row["evidence_status_pred"]})

    # ==========================================
    # 3. CLARITY MACRO-F1 (35%) — GT evidence = Yes
    # ==========================================
    clarity_subset  = merged[merged["evidence_status_gt"] == "Yes"]
    clarity_classes = ["Clear", "Not Clear", "Misleading"]
    clarity_macro, clarity_per = macro_f1(
        clarity_subset["evidence_quality_gt"].tolist(),
        clarity_subset["evidence_quality_pred"].tolist(),
        clarity_classes
    )

    print("\n" + "=" * 60)
    print("3. CLARITY (evidence_quality, GT evidence=Yes) — Weight: 0.35")
    print("=" * 60)
    print(f"   Evaluated on {len(clarity_subset)} samples")
    print(f"   Macro-F1 = {clarity_macro:.4f}")
    for cls, m in clarity_per.items():
        print(f"   {cls:<15} F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})")
    print(f"\n   GT:   {dict(clarity_subset['evidence_quality_gt'].value_counts())}")
    print(f"   Pred: {dict(clarity_subset['evidence_quality_pred'].value_counts())}")
    print(f"\n{confusion_matrix_str(clarity_subset['evidence_quality_gt'].tolist(), clarity_subset['evidence_quality_pred'].tolist(), clarity_classes)}")

    for _, row in clarity_subset.iterrows():
        if row["evidence_quality_gt"] != row["evidence_quality_pred"]:
            errors.append({"id": row["id"], "field": "evidence_quality",
                           "gt": row["evidence_quality_gt"], "pred": row["evidence_quality_pred"]})

    # ==========================================
    # 4. TIMELINE MACRO-F1 (15%) — GT promise = Yes
    # ==========================================
    timeline_classes = ["already", "within_2_years", "between_2_and_5_years", "longer_than_5_years"]
    timeline_macro, timeline_per = macro_f1(
        ev_subset["verification_timeline_gt"].tolist(),
        ev_subset["verification_timeline_pred"].tolist(),
        timeline_classes
    )

    print("\n" + "=" * 60)
    print("4. TIMELINE (verification_timeline, GT promise=Yes) — Weight: 0.15")
    print("=" * 60)
    print(f"   Evaluated on {len(ev_subset)} samples")
    print(f"   Macro-F1 = {timeline_macro:.4f}")
    for cls, m in timeline_per.items():
        print(f"   {cls:<30} F1={m['f1']:.4f}  P={m['precision']:.4f}  R={m['recall']:.4f}  "
              f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})")
    print(f"\n   GT:   {dict(ev_subset['verification_timeline_gt'].value_counts())}")
    print(f"   Pred: {dict(ev_subset['verification_timeline_pred'].value_counts())}")
    print(f"\n{confusion_matrix_str(ev_subset['verification_timeline_gt'].tolist(), ev_subset['verification_timeline_pred'].tolist(), timeline_classes)}")

    for _, row in ev_subset.iterrows():
        if row["verification_timeline_gt"] != row["verification_timeline_pred"]:
            errors.append({"id": row["id"], "field": "verification_timeline",
                           "gt": row["verification_timeline_gt"], "pred": row["verification_timeline_pred"]})

    # ==========================================
    # COMPOSITE SCORE
    # ==========================================
    total = (
        commitment_f1  * 0.20 +
        evidence_f1    * 0.30 +
        clarity_macro  * 0.35 +
        timeline_macro * 0.15
    )

    print("\n" + "=" * 60)
    print("COMPOSITE SCORE")
    print("=" * 60)
    print(f"  Commitment F1:     {commitment_f1:.4f} x 0.20 = {commitment_f1 * 0.20:.4f}")
    print(f"  Evidence F1:       {evidence_f1:.4f} x 0.30 = {evidence_f1 * 0.30:.4f}")
    print(f"  Clarity Macro-F1:  {clarity_macro:.4f} x 0.35 = {clarity_macro * 0.35:.4f}")
    print(f"  Timeline Macro-F1: {timeline_macro:.4f} x 0.15 = {timeline_macro * 0.15:.4f}")
    print(f"  {'─' * 40}")
    print(f"  ✅ TOTAL SCORE:    {total:.4f}")
    print("=" * 60)

    # Save errors
    if errors:
        errors_df   = pd.DataFrame(errors)
        errors_path = os.path.join(BASE_DIR, "results", "evaluation_errors.csv")
        os.makedirs(os.path.dirname(errors_path), exist_ok=True)
        errors_df.to_csv(errors_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved {len(errors)} errors → {errors_path}")

    return total


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VeriPromise ESG predictions")
    parser.add_argument("--predictions",  default=DEFAULT_PRED, help="Path to predictions TSV/CSV")
    parser.add_argument("--ground-truth", default=GT_CSV,       help="Path to ground truth test200 CSV")
    args = parser.parse_args()

    evaluate(args.ground_truth, args.predictions)