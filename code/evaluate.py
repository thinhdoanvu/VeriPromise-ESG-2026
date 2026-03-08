"""
Evaluation script for VeriPromise ESG 2026 competition.
Computes the official competition score:
  Total = Commitment_F1 * 0.20 + Evidence_F1 * 0.30 + Clarity_MacroF1 * 0.35 + Timeline_MacroF1 * 0.15
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_JSON = os.path.join(BASE_DIR, "datasets", "vpesg4k_train_1000 V1.json")
DEFAULT_PRED = os.path.join(BASE_DIR, "datasets", "output_esg_1000.csv")


def load_ground_truth(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df["id"] = df["id"].astype(str)
    return df


def load_predictions(path):
    df = pd.read_csv(path, dtype=str)
    df["id"] = df["id"].astype(str)
    return df


def normalize(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    val = str(val).strip()
    if val in ("", "nan", "NaN", "None", "none", "null"):
        return "N/A"
    return val


def f1_binary(y_true, y_pred, positive="Yes"):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p == positive)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive and p == positive)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive and p != positive)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def macro_f1(y_true, y_pred, classes):
    per_class = {}
    for cls in classes:
        per_class[cls] = f1_binary(y_true, y_pred, positive=cls)
    macro = np.mean([per_class[c]["f1"] for c in classes])
    return macro, per_class


def confusion_matrix_str(y_true, y_pred, classes):
    from collections import Counter
    pairs = Counter(zip(y_true, y_pred))
    # Header
    label = 'True\\Pred'
    header = f"{label:<20}" + "".join(f"{c:<20}" for c in classes)
    lines = [header, "-" * len(header)]
    for tc in classes:
        row = f"{tc:<20}" + "".join(f"{pairs.get((tc, pc), 0):<20}" for pc in classes)
        lines.append(row)
    return "\n".join(lines)


def evaluate(gt_path, pred_path):
    df_gt = load_ground_truth(gt_path)
    df_pred = load_predictions(pred_path)

    # Inner join on id
    merged = pd.merge(df_gt, df_pred, on="id", suffixes=("_gt", "_pred"))
    print(f"Matched samples: {len(merged)} (GT: {len(df_gt)}, Pred: {len(df_pred)})")

    if len(merged) == 0:
        print("ERROR: No matching IDs between ground truth and predictions.")
        sys.exit(1)

    # Normalize all relevant fields
    for col in ["promise_status", "verification_timeline", "evidence_status", "evidence_quality"]:
        for suffix in ["_gt", "_pred"]:
            merged[col + suffix] = merged[col + suffix].apply(normalize)

    errors = []

    # === 1. Commitment F1 (20%) ===
    commitment_result = f1_binary(
        merged["promise_status_gt"].tolist(),
        merged["promise_status_pred"].tolist(),
        positive="Yes"
    )
    commitment_f1 = commitment_result["f1"]

    print("\n" + "=" * 60)
    print("1. COMMITMENT (promise_status) — Weight: 0.20")
    print("=" * 60)
    print(f"   F1={commitment_f1:.4f}  P={commitment_result['precision']:.4f}  R={commitment_result['recall']:.4f}")
    print(f"   TP={commitment_result['tp']}  FP={commitment_result['fp']}  FN={commitment_result['fn']}")
    cm_classes = ["Yes", "No"]
    print(f"\n   Distribution GT:   {dict(merged['promise_status_gt'].value_counts())}")
    print(f"   Distribution Pred: {dict(merged['promise_status_pred'].value_counts())}")
    print(f"\n{confusion_matrix_str(merged['promise_status_gt'].tolist(), merged['promise_status_pred'].tolist(), cm_classes)}")

    # Track errors
    for _, row in merged.iterrows():
        if row["promise_status_gt"] != row["promise_status_pred"]:
            errors.append({
                "id": row["id"],
                "field": "promise_status",
                "gt": row["promise_status_gt"],
                "pred": row["promise_status_pred"],
            })

    # === 2. Evidence F1 (30%) — only where GT promise_status = "Yes" ===
    mask_promise_yes = merged["promise_status_gt"] == "Yes"
    ev_subset = merged[mask_promise_yes]

    evidence_result = f1_binary(
        ev_subset["evidence_status_gt"].tolist(),
        ev_subset["evidence_status_pred"].tolist(),
        positive="Yes"
    )
    evidence_f1 = evidence_result["f1"]

    print("\n" + "=" * 60)
    print("2. EVIDENCE (evidence_status, where GT promise=Yes) — Weight: 0.30")
    print("=" * 60)
    print(f"   Evaluated on {len(ev_subset)} samples")
    print(f"   F1={evidence_f1:.4f}  P={evidence_result['precision']:.4f}  R={evidence_result['recall']:.4f}")
    print(f"   TP={evidence_result['tp']}  FP={evidence_result['fp']}  FN={evidence_result['fn']}")
    ev_classes = ["Yes", "No"]
    print(f"\n   Distribution GT:   {dict(ev_subset['evidence_status_gt'].value_counts())}")
    print(f"   Distribution Pred: {dict(ev_subset['evidence_status_pred'].value_counts())}")
    print(f"\n{confusion_matrix_str(ev_subset['evidence_status_gt'].tolist(), ev_subset['evidence_status_pred'].tolist(), ev_classes)}")

    for _, row in ev_subset.iterrows():
        if row["evidence_status_gt"] != row["evidence_status_pred"]:
            errors.append({
                "id": row["id"],
                "field": "evidence_status",
                "gt": row["evidence_status_gt"],
                "pred": row["evidence_status_pred"],
            })

    # === 3. Clarity Macro-F1 (35%) — where GT evidence_status = "Yes" ===
    mask_evidence_yes = merged["evidence_status_gt"] == "Yes"
    clarity_subset = merged[mask_evidence_yes]

    clarity_classes = ["Clear", "Not Clear", "Misleading"]
    clarity_macro_f1, clarity_per_class = macro_f1(
        clarity_subset["evidence_quality_gt"].tolist(),
        clarity_subset["evidence_quality_pred"].tolist(),
        clarity_classes
    )

    print("\n" + "=" * 60)
    print("3. CLARITY (evidence_quality, where GT evidence=Yes) — Weight: 0.35")
    print("=" * 60)
    print(f"   Evaluated on {len(clarity_subset)} samples")
    print(f"   Macro-F1 = {clarity_macro_f1:.4f}")
    for cls, metrics in clarity_per_class.items():
        print(f"   {cls:<15} F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  (TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']})")
    print(f"\n   Distribution GT:   {dict(clarity_subset['evidence_quality_gt'].value_counts())}")
    print(f"   Distribution Pred: {dict(clarity_subset['evidence_quality_pred'].value_counts())}")
    print(f"\n{confusion_matrix_str(clarity_subset['evidence_quality_gt'].tolist(), clarity_subset['evidence_quality_pred'].tolist(), clarity_classes)}")

    for _, row in clarity_subset.iterrows():
        if row["evidence_quality_gt"] != row["evidence_quality_pred"]:
            errors.append({
                "id": row["id"],
                "field": "evidence_quality",
                "gt": row["evidence_quality_gt"],
                "pred": row["evidence_quality_pred"],
            })

    # === 4. Timeline Macro-F1 (15%) — where GT promise_status = "Yes" ===
    timeline_classes = ["already", "within_2_years", "between_2_and_5_years", "longer_than_5_years"]
    timeline_macro_f1, timeline_per_class = macro_f1(
        ev_subset["verification_timeline_gt"].tolist(),
        ev_subset["verification_timeline_pred"].tolist(),
        timeline_classes
    )

    print("\n" + "=" * 60)
    print("4. TIMELINE (verification_timeline, where GT promise=Yes) — Weight: 0.15")
    print("=" * 60)
    print(f"   Evaluated on {len(ev_subset)} samples")
    print(f"   Macro-F1 = {timeline_macro_f1:.4f}")
    for cls, metrics in timeline_per_class.items():
        print(f"   {cls:<30} F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  R={metrics['recall']:.4f}  (TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']})")
    print(f"\n   Distribution GT:   {dict(ev_subset['verification_timeline_gt'].value_counts())}")
    print(f"   Distribution Pred: {dict(ev_subset['verification_timeline_pred'].value_counts())}")
    print(f"\n{confusion_matrix_str(ev_subset['verification_timeline_gt'].tolist(), ev_subset['verification_timeline_pred'].tolist(), timeline_classes)}")

    for _, row in ev_subset.iterrows():
        if row["verification_timeline_gt"] != row["verification_timeline_pred"]:
            errors.append({
                "id": row["id"],
                "field": "verification_timeline",
                "gt": row["verification_timeline_gt"],
                "pred": row["verification_timeline_pred"],
            })

    # === Composite Score ===
    total = (
        commitment_f1 * 0.20
        + evidence_f1 * 0.30
        + clarity_macro_f1 * 0.35
        + timeline_macro_f1 * 0.15
    )

    print("\n" + "=" * 60)
    print("COMPOSITE SCORE")
    print("=" * 60)
    print(f"  Commitment F1:     {commitment_f1:.4f} x 0.20 = {commitment_f1 * 0.20:.4f}")
    print(f"  Evidence F1:       {evidence_f1:.4f} x 0.30 = {evidence_f1 * 0.30:.4f}")
    print(f"  Clarity Macro-F1:  {clarity_macro_f1:.4f} x 0.35 = {clarity_macro_f1 * 0.35:.4f}")
    print(f"  Timeline Macro-F1: {timeline_macro_f1:.4f} x 0.15 = {timeline_macro_f1 * 0.15:.4f}")
    print(f"  ─────────────────────────────────────")
    print(f"  TOTAL SCORE:       {total:.4f}")
    print("=" * 60)

    # Save errors
    if errors:
        errors_df = pd.DataFrame(errors)
        errors_path = os.path.join(BASE_DIR, "datasets", "evaluation_errors.csv")
        errors_df.to_csv(errors_path, index=False, encoding="utf-8-sig")
        print(f"\nSaved {len(errors)} errors to {errors_path}")

    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VeriPromise ESG predictions")
    parser.add_argument("--predictions", default=DEFAULT_PRED, help="Path to predictions CSV")
    parser.add_argument("--ground-truth", default=GT_JSON, help="Path to ground truth JSON")
    args = parser.parse_args()

    evaluate(args.ground_truth, args.predictions)
