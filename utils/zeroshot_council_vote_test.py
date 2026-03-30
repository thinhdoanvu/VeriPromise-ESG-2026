import os
import sys
import time
import json
import pandas as pd
from collections import Counter
from openai import OpenAI

# ==========================================
# 1. CONFIG
# ==========================================
BASE_URL = "http://localhost:11434/v1"
API_KEY  = "not-needed"
client   = OpenAI(base_url=BASE_URL, api_key=API_KEY)

TEST_CSV   = r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_1000 V1_test200.csv"
OUTPUT_CSV = r"C:\Users\VU\Documents\NLP\AICup26\results\council_test200_results.csv"
OUTPUT_DIR = r"C:\Users\VU\Documents\NLP\AICup26\results"

VOTERS_S1 = ["qwen2.5-14b-esg", "deepseek-r1-70b-esg", "qwen2.5-32b-esg"]
VOTERS    = ["qwen2.5-14b-esg", "qwen2.5-72b-esg", "deepseek-r1-70b-esg"]
CHAIRMAN  = "qwen2.5-32b-esg"

TIMEOUT_LONG  = 600
TIMEOUT_SHORT = 300

VALID_S1 = ["Yes", "No"]
VALID_S2 = ["Yes", "No"]
VALID_S3 = ["Clear", "Not Clear", "Misleading", "N/A"]
VALID_S4 = ["already", "within_2_years", "between_2_and_5_years", "longer_than_5_years"]

# ==========================================
# 2. CALL MODEL
# ==========================================
def call_model(model_name, prompt):
    timeout = TIMEOUT_LONG if any(x in model_name for x in ["72b", "70b"]) else TIMEOUT_SHORT
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
            top_p=1.0,
            timeout=timeout
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [WARN] {model_name} failed: {e}")
        return None

# ==========================================
# 3. EXTRACT VALID LABEL
# ==========================================
def extract_label(raw, valid_labels):
    if not raw:
        return "UNKNOWN"
    for label in valid_labels:
        if label.lower() in raw.lower():
            return label
    return "UNKNOWN"

# ==========================================
# 4. COUNCIL VOTE — 3 voters + chairman tie-break
# ==========================================
def council_vote(prompt, valid_labels, voters=None):
    votes = {}
    raw_responses = {}

    # 3 voters vote
    active_voters = voters if voters else VOTERS
    for voter in active_voters:
        raw = call_model(voter, prompt)
        label = extract_label(raw, valid_labels)
        votes[voter] = label
        raw_responses[voter] = raw or ""

    vote_values = list(votes.values())
    count = Counter(vote_values)
    most_common = count.most_common()

    # Majority 2/3 → done
    if most_common[0][1] >= 2:
        winner = most_common[0][0]
        method = "majority"
    else:
        # 1-1-1 tie → chairman decides
        print(f"  [TIE] Calling chairman...")
        raw_chair = call_model(CHAIRMAN, prompt)
        winner = extract_label(raw_chair, valid_labels)
        raw_responses["chairman"] = raw_chair or ""
        method = "chairman"

    return winner, votes, method

# ==========================================
# 5. PROMPTS
# ==========================================
def prompt_s1(data):
    return (
        "You are an ESG analyst. "
        "Does the following ESG statement express a concrete corporate commitment or promise toward future actions?\n"
        "Answer only: Yes or No\n\n"
        f"Statement: {data}\n"
        "Answer:"
    )

def prompt_s2(data):
    return (
        "You are an ESG analyst. "
        "Is the following ESG commitment supported by concrete evidence such as action plans, data, methodologies, or implementation records?\n"
        "Answer only: Yes or No\n\n"
        f"Statement: {data}\n"
        "Answer:"
    )

def prompt_s3(data):
    return (
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

def prompt_s4(data):
    return (
        "You are an ESG analyst. The ESG report was published in 2024.\n"
        "Based on the statement, classify the expected completion timeframe of the commitment.\n"
        "Choose exactly one:\n"
        "- already (action already implemented or ongoing in 2024)\n"
        "- within_2_years (specific target year 2025 or 2026 mentioned)\n"
        "- between_2_and_5_years (target year 2027, 2028, or 2029 mentioned)\n"
        "- longer_than_5_years (target year 2030 or beyond, e.g. 2030, 2050)\n\n"
        f"Statement: {data}\n"
        "Answer:"
    )

# ==========================================
# 6. PROCESS 1 ROW
# ==========================================
def process_row(row):
    data        = str(row["data"]).strip()
    promise_gt  = str(row["promise_status"]).strip()
    evidence_gt = str(row["evidence_status"]).strip() if pd.notna(row["evidence_status"]) else "No"
    quality_gt  = str(row["evidence_quality"]).strip() if pd.notna(row["evidence_quality"]) else "N/A"
    timeline_gt = str(row["verification_timeline"]).strip() if pd.notna(row["verification_timeline"]) else ""

    result = {
        "id":      row["id"],
        # Ground truth
        "s1_gt":   promise_gt,
        "s2_gt":   evidence_gt if promise_gt == "Yes" else "SKIP",
        "s3_gt":   quality_gt  if promise_gt == "Yes" else "SKIP",
        "s4_gt":   timeline_gt if promise_gt == "Yes" and timeline_gt else "SKIP",
    }

    # --- S1: Commitment ---
    s1_pred, s1_votes, s1_method = council_vote(prompt_s1(data), VALID_S1, voters=VOTERS_S1)
    result["s1_pred"]   = s1_pred
    result["s1_votes"]  = str(s1_votes)
    result["s1_method"] = s1_method
    result["s1_correct"] = (s1_pred == promise_gt)

    if promise_gt == "Yes":
        # --- S2: Evidence ---
        s2_pred, s2_votes, s2_method = council_vote(prompt_s2(data), VALID_S2)
        result["s2_pred"]    = s2_pred
        result["s2_votes"]   = str(s2_votes)
        result["s2_method"]  = s2_method
        result["s2_correct"] = (s2_pred == evidence_gt)

        # --- S3: Clarity ---
        s3_pred, s3_votes, s3_method = council_vote(prompt_s3(data), VALID_S3)
        result["s3_pred"]    = s3_pred
        result["s3_votes"]   = str(s3_votes)
        result["s3_method"]  = s3_method
        result["s3_correct"] = (s3_pred == quality_gt)

        # --- S4: Timeline ---
        if timeline_gt:
            s4_pred, s4_votes, s4_method = council_vote(prompt_s4(data), VALID_S4)
            result["s4_pred"]    = s4_pred
            result["s4_votes"]   = str(s4_votes)
            result["s4_method"]  = s4_method
            result["s4_correct"] = (s4_pred == timeline_gt)
        else:
            result["s4_pred"] = result["s4_votes"] = result["s4_method"] = "SKIP"
            result["s4_correct"] = None
    else:
        for key in ["s2_pred","s2_votes","s2_method","s3_pred","s3_votes","s3_method","s4_pred","s4_votes","s4_method"]:
            result[key] = "SKIP"
        result["s2_correct"] = result["s3_correct"] = result["s4_correct"] = None

    return result

# ==========================================
# 7. PRINT ACCURACY
# ==========================================
def print_accuracy(df):
    print(f"\n{'='*60}")
    print(f"  ACCURACY REPORT: Council Voting")
    print(f"{'='*60}")

    subtasks = {
        "S1 Commitment": "s1",
        "S2 Evidence":   "s2",
        "S3 Clarity":    "s3",
        "S4 Timeline":   "s4",
    }

    total_correct = total_count = 0
    for label, key in subtasks.items():
        col = f"{key}_correct"
        valid = df[col].dropna()
        if len(valid) == 0:
            continue
        correct = int(valid.sum())
        count   = len(valid)
        acc     = correct / count * 100
        total_correct += correct
        total_count   += count
        print(f"  {label:20s}: {correct:3d}/{count} = {acc:.1f}%")

        # Per-label breakdown
        sub = df[df[col].notna()]
        for gt in sorted(sub[f"{key}_gt"].unique()):
            if gt in ["SKIP", ""]:
                continue
            mask = sub[f"{key}_gt"] == gt
            sc = int(sub[mask][col].sum())
            ct = int(mask.sum())
            print(f"    L {gt:28s}: {sc:3d}/{ct} = {sc/ct*100:.1f}%")

    overall = total_correct / total_count * 100 if total_count > 0 else 0
    print(f"\n  {'Overall':20s}: {total_correct}/{total_count} = {overall:.1f}%")

    # Chairman usage
    for key in ["s1","s2","s3","s4"]:
        col = f"{key}_method"
        if col in df.columns:
            ties = (df[col] == "chairman").sum()
            total = (df[col] != "SKIP").sum()
            if total > 0:
                print(f"  {key.upper()} chairman called: {ties}/{total} = {ties/total*100:.1f}%")

    print(f"{'='*60}")

# ==========================================
# 8. MAIN
# ==========================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(TEST_CSV)
    print(f"Test set: {len(df)} rows")
    print(f"Voters S1: {VOTERS_S1}")
    print(f"Voters S2/S3/S4: {VOTERS}")
    print(f"Chairman: {CHAIRMAN}")

    results = []
    total = len(df)

    for idx, (_, row) in enumerate(df.iterrows()):
        print(f"\n[{idx+1}/{total}] id={row['id']}")
        result = process_row(row)
        results.append(result)

        # Luu CSV tung row de tranh mat du lieu
        df_row = pd.DataFrame([result])
        if not os.path.exists(OUTPUT_CSV):
            df_row.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        else:
            df_row.to_csv(OUTPUT_CSV, index=False, mode="a", header=False, encoding="utf-8-sig")

        time.sleep(0.2)

    # Final report
    df_results = pd.DataFrame(results)
    print_accuracy(df_results)
    print(f"\nCSV saved: {OUTPUT_CSV}")