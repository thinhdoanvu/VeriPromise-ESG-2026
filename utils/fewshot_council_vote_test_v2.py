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
OUTPUT_CSV = r"C:\Users\VU\Documents\NLP\AICup26\results\council_test200_results2.csv"
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
"""
PROMPT v2 — Few-shot examples lay nguyen van tu data goc (vpesg4k_train_1000_V1.json)
Khong paraphrase, khong chinh sua — dung y chang de model bam sat thuc te
"""

# ==========================================
# S1: Commitment
# ==========================================
def prompt_s1(data):
    return (
        "You are an ESG analyst reviewing Traditional Chinese ESG reports published in 2024.\n"
        "Task: Does this statement contain a CONCRETE FUTURE COMMITMENT or PROMISE?\n\n"
        "Answer Yes: statement includes specific future actions, targets, or pledges\n"
        "Answer No: statement only reports past facts, describes current situation, lists subsidiaries, or summarizes report methodology\n\n"
        "Examples:\n"
        "Statement: 我們總結世芯電子 2024 年永續報告書內容，對於世芯電子之相關運作與績效則提供了一個公平的觀點。基於保證範圍限制事項，世芯電子所提供資訊數據以及經檢視之測試，此報告書並無重大之不實陳述；而報告書中有關水與放流水主題之具體管理方針及績效指標內容則為實質正確之呈現。報告書所揭露之永續績效資訊展現了世芯電子識別利害關係人的努力。\n"
        "Answer: No\n\n"
        "Statement: 本報告書依循全球永續性標準理事會（GSSB）所發布之最新版 GRI 準則（GRI Universal Standards 2021）進行編撰，同時參考臺灣證券交易所發布之「上市公司編製與申報永續報告書作業辦法」及相關國際準則倡議等編製與揭露本報告書。\n"
        "Answer: No\n\n"
        "Statement: 聯發科技除在「工作規則」中依照勞基法明確規定「員工在產假期間公司不得終止勞動契約」外，為支持同仁與其家人度過人生不同階段，自 2024 年起提供女性員工在分娩前後計有 12 週共 84 天的產假；男性員工則可於其配偶懷孕期間陪同產檢或生（流）產日及前後 15 日間請假陪伴，兩者合計共有 10 天陪產（檢）假可運用，陪產（檢）假期間工資照常給付。\n"
        "Answer: Yes\n\n"
        "Statement: 在與供應商攜手邁向永續發展的過程中，台泥致力於建立一套以合作為基礎的價值體體系。核心原則是推動供應鏈的低碳轉型，同時維護人權、保護環境和促進生物多樣性，以建立永續的合作夥伴關係。\n"
        "Answer: Yes\n\n"
        f"Statement: {data}\n"
        "Answer:"
    )

# ==========================================
# S2: Evidence
# ==========================================
def prompt_s2(data):
    return (
        "You are an ESG analyst reviewing Traditional Chinese ESG reports published in 2024.\n"
        "Task: Is this ESG commitment supported by CONCRETE EVIDENCE?\n\n"
        "Answer Yes: has specific data, numbers, percentages, named action plans, standards, third-party verification, or implementation records\n"
        "Answer No: only general statements, vague intentions, or commitments without supporting evidence\n\n"
        "Examples:\n"
        "Statement: III. 產品危害物質減免 (HSF) 短、中、長期目標設定    **1. 2024 年 目標和實績**  *   已達標  *   詳述於 \"產品各階段危害物質減免風險管理目標與成效\"    **2. 短期目標 2025~2026 年**  *   監控 PFAS 中有害物質管制要求  *   減少 RoHS 豁免零件之使用，往 Lead Free 持續前進。    **3. 中長期目標 2026~2027 年**  *   優化綠色管理系統審查作業提升效率。\n"
        "Answer: Yes\n\n"
        "Statement: 在與供應商攜手邁向永續發展的過程中，台泥致力於建立一套以合作為基礎的價值體體系。核心原則是推動供應鏈的低碳轉型，同時維護人權、保護環境和促進生物多樣性，以建立永續的合作夥伴關係。\n"
        "Answer: Yes\n\n"
        "Statement: 聯發科技除在「工作規則」中依照勞基法明確規定「員工在產假期間公司不得終止勞動契約」外，為支持同仁與其家人度過人生不同階段，自 2024 年起提供女性員工在分娩前後計有 12 週共 84 天的產假；男性員工則可於其配偶懷孕期間陪同產檢或生（流）產日及前後 15 日間請假陪伴，兩者合計共有 10 天陪產（檢）假可運用，陪產（檢）假期間工資照常給付。\n"
        "Answer: No\n\n"
        f"Statement: {data}\n"
        "Answer:"
    )

# ==========================================
# S3: Clarity
# ==========================================
def prompt_s3(data):
    return (
        "You are an ESG analyst reviewing Traditional Chinese ESG reports published in 2024.\n"
        "Task: Evaluate the CLARITY of evidence in this statement.\n\n"
        "Definitions:\n"
        "- Clear: evidence contains specific numbers, percentages, named standards, third-party verification, or verifiable facts\n"
        "- Not Clear: evidence uses vague language such as '致力於', '積極推動', '逐步實施', '持續改善' without concrete metrics\n"
        "- Misleading: statement appears credible but contains unverifiable claims or overly broad commitments with no substance\n"
        "- N/A: no evidence present in the statement\n\n"
        "Examples:\n"
        "Statement: 面對這些全球與國內政策的推動，富邦人壽秉持永續經營理念，積極透過永續金融的投資力量，引導被投資公司與企業進行永續轉型，除非資金明確用於綠能轉型計畫，不再新增投資燃煤比重超過 50% 的電廠。同時，針對燃料煤開採、運輸業、燃料煤發電及非典型油氣產業，制定嚴格的准入與撤資標準，積極引導資金流向低碳與可再生能源領域，展現對環境永續的堅定承諾，並連續 4 年榮獲「台灣永續投資典範機構獎 – 機構影響力 (壽險組)」殊榮，在責任投資與推動企業永續發展上的卓越表現備受肯定。\n"
        "Answer: Clear\n\n"
        "Statement: 在與供應商攜手邁向永續發展的過程中，台泥致力於建立一套以合作為基礎的價值體體系。核心原則是推動供應鏈的低碳轉型，同時維護人權、保護環境和促進生物多樣性，以建立永續的合作夥伴關係。\n"
        "Answer: Not Clear\n\n"
        "Statement: 台達透過產品再設計、低碳材料議合及材料循環利用等措施，積極推動供應鏈範疇三的減碳行動，並依據以下流程，集團在各事業單位內逐步實施年度減碳對策。以 2021 年作為減碳目標的基準年，並依照集團的 SBT 目標，預計至 2030 年，範疇三碳排放將降低 25%。\n"
        "Answer: Not Clear\n\n"
        "Statement: 統一企業致力於穩健經營，確保公司財務穩定與持續成長，並兼顧股東權益、員工發展與社會責任。我們透過創新產品與服務、多元化營運布局及強化供應鏈韌性，提升企業競爭力，並承諾遵循財務與稅務法規，維持高標準的財務透明度與公司治理。\n"
        "Answer: Misleading\n\n"
        "Statement: 聯發科技除在「工作規則」中依照勞基法明確規定「員工在產假期間公司不得終止勞動契約」外，為支持同仁與其家人度過人生不同階段，自 2024 年起提供女性員工在分娩前後計有 12 週共 84 天的產假；男性員工則可於其配偶懷孕期間陪同產檢或生（流）產日及前後 15 日間請假陪伴，兩者合計共有 10 天陪產（檢）假可運用，陪產（檢）假期間工資照常給付。\n"
        "Answer: N/A\n\n"
        f"Statement: {data}\n"
        "Answer:"
    )

# ==========================================
# S4: Timeline — chain-of-thought + real examples
# ==========================================
def prompt_s4(data):
    return (
        "You are an ESG analyst reviewing Traditional Chinese ESG reports published in 2024.\n"
        "Task: Classify the TIMEFRAME of this commitment.\n\n"
        "Step 1: Find the specific target year or timeframe mentioned.\n"
        "Step 2: Calculate gap = target_year - 2024.\n"
        "Step 3: Classify:\n"
        "  - already:               no future year, or action starts/started in 2024\n"
        "  - within_2_years:        gap 0-2 years (target year 2024, 2025, or 2026)\n"
        "  - between_2_and_5_years: gap 3-5 years (target year 2027, 2028, or 2029)\n"
        "  - longer_than_5_years:   gap > 5 years (target year 2030 or beyond)\n\n"
        "Examples:\n"
        "Statement: 聯發科技除在「工作規則」中依照勞基法明確規定「員工在產假期間公司不得終止勞動契約」外，為支持同仁與其家人度過人生不同階段，自 2024 年起提供女性員工在分娩前後計有 12 週共 84 天的產假；男性員工則可於其配偶懷孕期間陪同產檢或生（流）產日及前後 15 日間請假陪伴，兩者合計共有 10 天陪產（檢）假可運用，陪產（檢）假期間工資照常給付。\n"
        "Reasoning: starts in 2024, already implemented → already\n"
        "Answer: already\n\n"
        "Statement: 研華積極開發產品碳足跡計算系統，此系統採用生命週期評估 (LCA) 方法，參考 ISO14040、ISO 14067 和 GHG Protocol 等國際標準，透過整合研華內部的原物料、供應商資訊及生產管理系統，串接外部 API，並結合 AI 技術建立各原物料的碳足跡係數庫，使系統能快速計算產品碳足跡並生成碳足跡報告。預期於 2025 年可完成於各銷售中產品的計算，佔整體營收約 100%。\n"
        "Reasoning: year=2025, gap=1 year → within_2_years\n"
        "Answer: within_2_years\n\n"
        "Statement: 2024年台灣大進行年度採購分析，共選出40家包含重大供應商及高風險廠商進行實地審查，識別出0家高風險廠商。台灣大預計2025年將持續進行之重大供應商實地審查，盼持續透過實地審查深化與供應商之連結並推進永續實務。\n"
        "Reasoning: year=2025, gap=1 year → within_2_years\n"
        "Answer: within_2_years\n\n"
        "Statement: III. 產品危害物質減免 (HSF) 短、中、長期目標設定    **1. 2024 年 目標和實績**  *   已達標  *   **2. 短期目標 2025~2026 年**  *   監控 PFAS 中有害物質管制要求  *   減少 RoHS 豁免零件之使用，往 Lead Free 持續前進。    **3. 中長期目標 2026~2027 年**  *   優化綠色管理系統審查作業提升效率。\n"
        "Reasoning: main targets 2026-2027, gap=2-3 years → between_2_and_5_years\n"
        "Answer: between_2_and_5_years\n\n"
        "Statement: 台達透過產品再設計、低碳材料議合及材料循環利用等措施，積極推動供應鏈範疇三的減碳行動，並依據以下流程，集團在各事業單位內逐步實施年度減碳對策。以 2021 年作為減碳目標的基準年，並依照集團的 SBT 目標，預計至 2030 年，範疇三碳排放將降低 25%。\n"
        "Reasoning: year=2030, gap=6 years → longer_than_5_years\n"
        "Answer: longer_than_5_years\n\n"
        f"Statement: {data}\n"
        "Reasoning:"
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