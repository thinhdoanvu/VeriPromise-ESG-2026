"""
Generate readable results tables from experiments/results.jsonl.
Outputs both terminal tables and a CSV file.

Usage:
    python experiments/results_table.py
    python experiments/results_table.py --csv   # also save CSV
"""

import json
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = os.path.join(BASE_DIR, "experiments", "results.jsonl")
CSV_FILE = os.path.join(BASE_DIR, "experiments", "results_summary.csv")


def load_all():
    entries = []
    with open(RESULTS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def print_divider(title, width=90):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def table_stage1(entries):
    """Stage 1: single-task BERT (promise and evidence separately)."""
    rows = [e for e in entries if e["method"] == "stage1_finetune"]
    ensembles = [e for e in entries if e["method"] == "stage1_ensemble"]
    if not rows:
        return

    print_divider("STAGE 1: Single-Task BERT (separate promise & evidence models)")

    # Group by task
    for task in ["promise", "evidence"]:
        task_rows = [r for r in rows if r["task"] == task]
        task_ens = [r for r in ensembles if r["task"] == task]
        if not task_rows:
            continue

        print(f"\n  Task: {task}")
        print(f"  {'Model':<35} {'Argmax F1':>10} {'Thr-Tuned F1':>13} {'Threshold':>10}")
        print(f"  {'-'*35} {'-'*10} {'-'*13} {'-'*10}")

        for r in task_rows:
            m = r["metrics"]
            print(f"  {r['model']:<35} {m['cv_f1_argmax']:>10.4f} {m['cv_f1_threshold_tuned']:>13.4f} {m['best_threshold']:>10.2f}")

        for r in task_ens:
            m = r["metrics"]
            print(f"  {'** ENSEMBLE **':<35} {m['cv_f1_argmax']:>10.4f} {m['cv_f1_threshold_tuned']:>13.4f} {m['best_threshold']:>10.2f}")


def table_stage2(entries):
    """Stage 2: multi-task BERT (promise + evidence + quality jointly)."""
    rows = [e for e in entries if e["method"] == "stage2_multitask"]
    if not rows:
        return

    print_divider("STAGE 2: Multi-Task BERT (joint promise + evidence + quality)")

    print(f"\n  {'Model':<35} {'Promise F1':>11} {'Evidence F1':>12} {'Quality MF1':>12} {'Combined':>9}")
    print(f"  {'-'*35} {'-'*11} {'-'*12} {'-'*12} {'-'*9}")

    for r in rows:
        m = r["metrics"]
        print(f"  {r['model']:<35} {m['cv_promise_f1']:>11.4f} {m['cv_evidence_f1']:>12.4f} {m['cv_quality_macro_f1']:>12.4f} {m['cv_combined']:>9.4f}")

    # Per-fold detail
    print(f"\n  Per-fold breakdown (Combined):")
    print(f"  {'Model':<35} {'Fold1':>7} {'Fold2':>7} {'Fold3':>7} {'Fold4':>7} {'Fold5':>7} {'Std':>7}")
    print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for r in rows:
        folds = r["metrics"]["fold_combined"]
        import numpy as np
        std = np.std(folds)
        fold_str = " ".join(f"{f:>7.4f}" for f in folds)
        print(f"  {r['model']:<35} {fold_str} {std:>7.4f}")


def table_stage3(entries):
    """Stage 3: LLM timeline results."""
    rows = [e for e in entries if e["method"] == "stage3_eval" and e["metrics"].get("n_samples", 0) > 0]
    if not rows:
        return

    print_divider("STAGE 3: Timeline Classification (LLM + PDF context)")

    print(f"\n  {'Model':<30} {'N':>5} {'Macro-F1':>9} {'already':>9} {'within2y':>9} {'2-5y':>9} {'5y+':>9}")
    print(f"  {'-'*30} {'-'*5} {'-'*9} {'-'*9} {'-'*9} {'-'*9} {'-'*9}")

    for r in rows:
        m = r["metrics"]
        # Find corresponding LLM run to get model name
        model = r.get("model", "unknown")
        print(f"  {model:<30} {m['n_samples']:>5} {m['timeline_macro_f1']:>9.4f} "
              f"{m.get('f1_already',0):>9.4f} {m.get('f1_within_2_years',0):>9.4f} "
              f"{m.get('f1_between_2_and_5_years',0):>9.4f} {m.get('f1_longer_than_5_years',0):>9.4f}")


def table_baseline(entries):
    """LLM baseline."""
    rows = [e for e in entries if e["method"] == "llm_zeroshot"]
    if not rows:
        return

    print_divider("BASELINE: LLM Zero-Shot (full pipeline)")

    for r in rows:
        m = r["metrics"]
        print(f"\n  Model: {r['model']}")
        print(f"  Total Score:       {m['total_score']}")
        print(f"  Clarity Macro-F1:  {m['clarity_macro_f1']}")
        print(f"  Timeline Macro-F1: {m['timeline_macro_f1']}")
        print(f"  Note: {r.get('notes', '')}")


def table_comparison(entries):
    """Side-by-side comparison: baseline vs Stage 1 vs Stage 2."""
    baseline = [e for e in entries if e["method"] == "llm_zeroshot"]
    stage1_ens = [e for e in entries if e["method"] == "stage1_ensemble"]
    stage2 = [e for e in entries if e["method"] == "stage2_multitask"]

    if not stage2:
        return

    print_divider("COMPARISON: Baseline vs Stage 1 vs Stage 2 (best per approach)")

    # Best stage2 by combined
    best_s2 = max(stage2, key=lambda x: x["metrics"]["cv_combined"])

    # Stage 1 ensemble
    s1_promise = next((e for e in stage1_ens if e["task"] == "promise"), None)
    s1_evidence = next((e for e in stage1_ens if e["task"] == "evidence"), None)

    print(f"\n  {'Task (weight)':<25} {'LLM Baseline':>13} {'Stage1 Ensemble':>16} {'Stage2 Best':>12} {'Improvement':>12}")
    print(f"  {'-'*25} {'-'*13} {'-'*16} {'-'*12} {'-'*12}")

    # Promise
    bl_p = "~0.84"
    s1_p = f"{s1_promise['metrics']['cv_f1_threshold_tuned']:.4f}" if s1_promise else "—"
    s2_p = f"{best_s2['metrics']['cv_promise_f1']:.4f}"
    print(f"  {'Promise (0.20)':<25} {bl_p:>13} {s1_p:>16} {s2_p:>12}")

    # Evidence
    bl_e = "~0.65"
    s1_e = f"{s1_evidence['metrics']['cv_f1_threshold_tuned']:.4f}" if s1_evidence else "—"
    s2_e = f"{best_s2['metrics']['cv_evidence_f1']:.4f}"
    print(f"  {'Evidence (0.30)':<25} {bl_e:>13} {s1_e:>16} {s2_e:>12}")

    # Quality
    bl_q = "0.3181"
    s2_q = f"{best_s2['metrics']['cv_quality_macro_f1']:.4f}"
    print(f"  {'Quality (0.35)':<25} {bl_q:>13} {'— (not trained)':>16} {s2_q:>12} {'*** +0.45':>12}")

    # Timeline
    print(f"  {'Timeline (0.15)':<25} {'0.5127':>13} {'— (not trained)':>16} {'(pending)':>12}")

    # Estimated total
    s2_total = (best_s2['metrics']['cv_promise_f1'] * 0.20 +
                best_s2['metrics']['cv_evidence_f1'] * 0.30 +
                best_s2['metrics']['cv_quality_macro_f1'] * 0.35 +
                0.5127 * 0.15)
    print(f"\n  Estimated Total Score:")
    print(f"    Baseline:            0.6333")
    print(f"    Stage2 + LLM timeline: {s2_total:.4f} (using baseline timeline 0.5127)")
    print(f"    Improvement:         +{s2_total - 0.6333:.4f}")


def save_csv(entries):
    """Save results as CSV for easy viewing in spreadsheets."""
    import csv

    # Stage 2 results
    stage2 = [e for e in entries if e["method"] == "stage2_multitask"]
    stage1 = [e for e in entries if e["method"] in ("stage1_finetune", "stage1_ensemble")]

    with open(CSV_FILE, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)

        w.writerow(["STAGE 2: Multi-Task BERT Results"])
        w.writerow(["Model", "Promise F1", "Evidence F1", "Quality Macro-F1", "Combined",
                     "Fold1", "Fold2", "Fold3", "Fold4", "Fold5"])
        for r in stage2:
            m = r["metrics"]
            folds = m["fold_combined"]
            w.writerow([r["model"], f"{m['cv_promise_f1']:.4f}", f"{m['cv_evidence_f1']:.4f}",
                        f"{m['cv_quality_macro_f1']:.4f}", f"{m['cv_combined']:.4f}"] +
                       [f"{fc:.4f}" for fc in folds])

        w.writerow([])
        w.writerow(["STAGE 1: Single-Task BERT Results"])
        w.writerow(["Model", "Task", "Argmax F1", "Threshold-Tuned F1", "Best Threshold"])
        for r in stage1:
            m = r["metrics"]
            if "cv_f1_argmax" in m:
                w.writerow([r["model"], r["task"], f"{m['cv_f1_argmax']:.4f}",
                            f"{m['cv_f1_threshold_tuned']:.4f}", f"{m['best_threshold']:.2f}"])
            elif "cv_f1_threshold_tuned" in m:
                w.writerow([r["model"], r["task"], f"{m['cv_f1_argmax']:.4f}",
                            f"{m['cv_f1_threshold_tuned']:.4f}", f"{m['best_threshold']:.2f}"])

    print(f"\nSaved CSV to: {CSV_FILE}")


def main():
    entries = load_all()
    print(f"Loaded {len(entries)} experiment records\n")

    table_baseline(entries)
    table_stage1(entries)
    table_stage2(entries)
    table_stage3(entries)
    table_comparison(entries)

    if "--csv" in sys.argv:
        save_csv(entries)


if __name__ == "__main__":
    main()
