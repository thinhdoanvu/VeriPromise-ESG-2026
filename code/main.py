import os
import argparse
import pandas as pd
import json, re
from tqdm import tqdm

from config import client, DEFAULT_MODEL, MODELS, PARAMS, BATCH_SIZE

# ==========================================
# 1. Paths
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_CSV = os.path.join(BASE_DIR, "datasets", "vpesg4k_train_1000 V1.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "datasets", "output_esg_1000.csv")

# ==========================================
# 2. CLI args
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None, help="Model name or preset key from config.MODELS")
parser.add_argument("--input", default=INPUT_CSV, help="Input CSV path")
parser.add_argument("--output", default=OUTPUT_CSV, help="Output CSV path")
parser.add_argument("--limit", type=int, default=0, help="Max rows to process (0=all)")
args = parser.parse_args()

MODEL = MODELS.get(args.model, args.model) if args.model else DEFAULT_MODEL
INPUT_CSV = args.input
OUTPUT_CSV = args.output

# ==========================================
# 3. Load input and existing output
# ==========================================
df_input = pd.read_csv(INPUT_CSV)
print(f"Input rows: {len(df_input)}, Model: {MODEL}")

if os.path.exists(OUTPUT_CSV):
    df_output = pd.read_csv(OUTPUT_CSV)
    processed_ids = set(df_output["id"].astype(str))
    print("Already processed IDs:", len(processed_ids))
else:
    df_output = pd.DataFrame(columns=[
        "id","data","esg_type",
        "promise_status","promise_string",
        "verification_timeline",
        "evidence_status","evidence_string","evidence_quality"
    ])
    processed_ids = set()

# ==========================================
# 4. Prompt
# ==========================================

SYSTEM_PROMPT = """You are an ESG (Environmental, Social, Governance) analyst and data annotator.
Your task is to carefully analyze corporate ESG disclosures and classify them.

Return ONLY a single valid JSON object with these exact keys:
  promise_status, promise_string, verification_timeline,
  evidence_status, evidence_string, evidence_quality

Rules:

**promise_status** — "Yes" or "No"
- "Yes": The text contains a corporate COMMITMENT — a forward-looking pledge, goal, plan, or ongoing initiative the company commits to.
- "No": The text is purely factual reporting, third-party descriptions, organizational listings, methodology descriptions, historical data without commitments, or general industry observations without a specific corporate pledge.
- Key distinction: A commitment requires the company to take action or achieve a result. Merely reporting facts, describing structures, or summarizing external information is NOT a commitment.

**promise_string** — The excerpt showing the commitment. Empty string if promise_status="No".

**verification_timeline** — When the commitment will be fulfilled:
- "N/A" if promise_status="No"
- "already" if the action is already implemented, ongoing, or completed
- "within_2_years" if the commitment will be achieved within 2 years from 2026
- "between_2_and_5_years" if the target is 2-5 years away from 2026
- "longer_than_5_years" if the target is more than 5 years away from 2026
- If a specific year is mentioned, calculate: target_year - 2026
  - target_year <= 2027 → within_2_years
  - 2028 <= target_year <= 2030 → between_2_and_5_years
  - target_year >= 2031 → longer_than_5_years
- Qualitative terms: short-term → within_2_years, medium-term → between_2_and_5_years, long-term → longer_than_5_years
- If the commitment has NO timeframe and describes an ongoing/indefinite effort → "longer_than_5_years"
- If the commitment describes broad, aspirational goals without specific dates (e.g., "致力於...", "持續推動...") → "longer_than_5_years"

**evidence_status** — "Yes" or "No"
- "N/A" if promise_status="No"
- "Yes" if there is concrete supporting evidence (data, metrics, specific actions taken, third-party certifications, awards)
- "No" if the commitment stands alone without supporting evidence

**evidence_string** — The excerpt providing evidence. Empty string if evidence_status is "No" or "N/A".

**evidence_quality** — Quality of the evidence:
- "N/A" if evidence_status is "No" or "N/A"
- "Clear" — Evidence includes specific numbers, dates, certifications, verifiable data, or concrete completed actions
- "Not Clear" — Evidence is vague, describes plans/processes without concrete data, uses general descriptions without specifics, or merely restates intentions
- "Misleading" — Evidence contradicts the commitment or is deceptive
- Key distinction for "Not Clear": If evidence only describes WHAT will be done (plans, processes, frameworks) without showing concrete results/data, it is "Not Clear". For evidence to be "Clear", it must contain verifiable specifics."""

FEW_SHOT_EXAMPLES = """
Example 1 (promise=No — factual reporting, no commitment):
Text: "我們總結世芯電子 2024 年永續報告書內容，對於世芯電子之相關運作與績效則提供了一個公平的觀點。基於保證範圍限制事項，世芯電子所提供資訊數據以及經檢視之測試，此報告書並無重大之不實陳述；而報告書中有關水與放流水主題之具體管理方針及績效指標內容則為實質正確之呈現。報告書所揭露之永續績效資訊展現了世芯電子識別利害關係人的努力。"
Output:
{{"promise_status":"No","promise_string":"","verification_timeline":"N/A","evidence_status":"N/A","evidence_string":"","evidence_quality":"N/A"}}

Example 2 (promise=No — organizational listing, no commitment):
Text: "其中，廣達上海製造城 QSMC 包含達豐(上海)電腦有限公司、達功(上海)電腦有限公司、達利(上海)電腦有限公司、達人(上海)電腦有限公司、達群(上海)電腦有限公司及達偉(上海)物流倉儲有限公司；廣達重慶製造城 QCMC 則包含達豐(重慶)電腦有限公司、達功(重慶)電腦有限公司、達偉(重慶)物流有限公司及雲達(重慶)科技有限公司。截至2024年底，廣達集團全球員工總數為64,935人，涵蓋臺灣的QRDC總部及全球各主要製造據點。"
Output:
{{"promise_status":"No","promise_string":"","verification_timeline":"N/A","evidence_status":"N/A","evidence_string":"","evidence_quality":"N/A"}}

Example 3 (promise=Yes, timeline=already, evidence=No):
Text: "聯發科技除在「工作規則」中依照勞基法明確規定「員工在產假期間公司不得終止勞動契約」外，為支持同仁與其家人度過人生不同階段，自 2024 年起提供女性員工在分娩前後計有 12 週共 84 天的產假；男性員工則可於其配偶懷孕期間陪同產檢或生（流）產日及前後 15 日間請假陪伴，兩者合計共有 10 天陪產（檢）假可運用，陪產（檢）假期間工資照常給付。"
Output:
{{"promise_status":"Yes","promise_string":"為支持同仁與其家人度過人生不同階段，自 2024 年起提供女性員工在分娩前後計有 12 週共 84 天的產假；男性員工則可於其配偶懷孕期間陪同產檢或生（流）產日及前後 15 日間請假陪伴，兩者合計共有 10 天陪產（檢）假可運用，陪產（檢）假期間工資照常給付。","verification_timeline":"already","evidence_status":"No","evidence_string":"","evidence_quality":"N/A"}}

Example 4 (promise=Yes, timeline=within_2_years, evidence=Clear):
Text: "研華積極開發產品碳足跡計算系統，除協助掌握各銷售中產品的碳足跡外，亦可接軌碳定價及產品碳足跡減量策略的推進。此系統採用生命週期評估 (LCA) 方法，參考 ISO14040、ISO 14067 和 GHG Protocol 等國際標準，透過整合研華內部的原物料、供應商資訊及生產管理系統，串接外部 API，並結合 AI 技術建立各原物料的碳足跡係數庫，使系統能快速計算產品碳足跡並生成碳足跡報告，以應用於評估產品在各生命週期階段的排放量。預期於 2025 年可完成於各銷售中產品的計算，佔整體營收約 100%。"
Output:
{{"promise_status":"Yes","promise_string":"研華積極開發產品碳足跡計算系統，除協助掌握各銷售中產品的碳足跡外，亦可接軌碳定價及產品碳足跡減量策略的推進。","verification_timeline":"within_2_years","evidence_status":"Yes","evidence_string":"系統採用生命週期評估 (LCA) 方法，參考 ISO14040、ISO 14067 和 GHG Protocol 等國際標準，透過整合研華內部的原物料、供應商資訊及生產管理系統，串接外部 API，並結合 AI 技術建立各原物料的碳足跡係數庫，","evidence_quality":"Clear"}}

Example 5 (promise=Yes, timeline=between_2_and_5_years, evidence=Not Clear):
Text: "在與供應商攜手邁向永續發展的過程中，台泥致力於建立一套以合作為基礎的價值體體系。核心原則是推動供應鏈的低碳轉型，同時維護人權、保護環境和促進生物多樣性，以建立永續的合作夥伴關係。"
Output:
{{"promise_status":"Yes","promise_string":"台泥致力於建立一套以合作為基礎的價值體體系。","verification_timeline":"between_2_and_5_years","evidence_status":"Yes","evidence_string":"核心原則是推動供應鏈的低碳轉型，同時維護人權、保護環境和促進生物多樣性，以建立永續的合作夥伴關係。","evidence_quality":"Not Clear"}}

Example 6 (promise=Yes, timeline=longer_than_5_years, evidence=Not Clear):
Text: "台達透過產品再設計、低碳材料議合及材料循環利用等措施，積極推動供應鏈範疇三的減碳行動，並依據以下流程，集團在各事業單位內逐步實施年度減碳對策。以 2021 年作為減碳目標的基準年，並依照集團的 SBT 目標，預計至 2030 年，範疇三碳排放將降低 25%。"
Output:
{{"promise_status":"Yes","promise_string":"台達透過產品再設計、低碳材料議合及材料循環利用等措施，積極推動供應鏈範疇三的減碳行動， ｜ 預計至 2030 年，範疇三碳排放將降低 25%","verification_timeline":"longer_than_5_years","evidence_status":"Yes","evidence_string":"依據以下流程，集團在各事業單位內逐步實施年度減碳對策。","evidence_quality":"Not Clear"}}

Example 7 (promise=Yes, timeline=longer_than_5_years, evidence=Clear):
Text: "供應鏈是日月光投控成為一家具有影響力公司的重要夥伴，積極與供應商共同規劃與推動五大低碳管理方針，涵蓋低碳選商永續策略、完善供應鏈碳資訊、推動材料與機台低碳轉型、導入上游低碳運輸與建立低碳供應鏈等，並透過日月光環保永續基金會舉辦低碳節能永續獎，攜手供應商夥伴一同實現 2050 年淨零排放承諾。  更進一步，日月光在行之有年的供應商評比中，除了品質與交期的評核外，首度納入 10% 的永續績效，同時日月光投控與供應商組成低碳供應聯盟，正式啟動低碳機台設備專案，攜手 19 家關鍵機台供應商合作推動機台節能設計以達成 2030 年節能 20% 的階段性目標。"
Output:
{{"promise_status":"Yes","promise_string":"積極與供應商共同規劃與推動五大低碳管理方針，涵蓋低碳選商永續策略、完善供應鏈碳資訊、推動材料與機台低碳轉型、導入上游低碳運輸與建立低碳供應鏈等，並透過日月光環保永續基金會舉辦低碳節能永續獎，攜手供應商夥伴一同實現 2050 年淨零排放承諾。","verification_timeline":"longer_than_5_years","evidence_status":"Yes","evidence_string":"除了品質與交期的評核外，首度納入 10% 的永續績效，同時日月光投控與供應商組成低碳供應聯盟，正式啟動低碳機台設備專案，攜手 19 家關鍵機台供應商合作推動機台節能設計以","evidence_quality":"Clear"}}

Example 8 (promise=Yes, timeline=longer_than_5_years, evidence=Not Clear):
Text: "廣達亦不定期進行小規模員工敬業度調查，2024年台灣廠區調查顯示，有80%員工願意在相同條件下繼續留任，其中女性員工投入度更達86%。為進一步了解績優員工的職場體驗，針對高績效同仁實施不記名投入度評估，結果顯示在成長機會與工作期待方面得分高，但在被肯定與讚賞的感受相對較低。對此，公司為主官規劃相關回饋訓練課程，透過情境模擬協助主管提升正向回饋能力。"
Output:
{{"promise_status":"Yes","promise_string":"廣達亦不定期進行小規模員工敬業度調查， ｜ 對此，公司為主官規劃相關回饋訓練課程，透過情境模擬協助主管提升正向回饋能力。","verification_timeline":"longer_than_5_years","evidence_status":"Yes","evidence_string":"2024年台灣廠區調查顯示，有80%員工願意在相同條件下繼續留任，其中女性員工投入度更達86%。為進一步了解績優員工的職場體驗，針對高績效同仁實施不記名投入度評估，結果顯示在成長機會與工作期待方面得分高，但在被肯定與讚賞的感受相對較低。對此，公司為主官規劃相關回饋訓練課程，透過情境模擬協助主管提升正向回饋能力。","evidence_quality":"Not Clear"}}

Example 9 (promise=Yes, timeline=already, evidence=Clear):
Text: "面對這些全球與國內政策的推動，富邦人壽秉持永續經營理念，積極透過永續金融的投資力量，引導被投資公司與企業進行永續轉型，除非資金明確用於綠能轉型計畫，不再新增投資燃煤比重超過 50% 的電廠。同時，針對燃料煤開採、運輸業、燃料煤發電及非典型油氣產業，制定嚴格的准入與撤資標準，積極引導資金流向低碳與可再生能源領域，展現對環境永續的堅定承諾，並連續 4 年榮獲「台灣永續投資典範機構獎 – 機構影響力 (壽險組)」殊榮，在責任投資與推動企業永續發展上的卓越表現備受肯定。"
Output:
{{"promise_status":"Yes","promise_string":"秉持永續經營理念，積極透過永續金融的投資力量，引導被投資公司與企業進行永續轉型，","verification_timeline":"already","evidence_status":"Yes","evidence_string":"不再新增投資燃煤比重超過 50% 的電廠。 ｜ 制定嚴格的准入與撤資標準，積極引導資金流向低碳與可再生能源領域，展現對環境永續的堅定承諾，並連續 4 年榮獲「台灣永續投資典範機構獎 – 機構影響力 (壽險組)」殊榮，","evidence_quality":"Clear"}}

Example 10 (promise=Yes, timeline=between_2_and_5_years, evidence=No):
Text: "近年來不時有便利商店發生職場暴力攻擊事件，勞動部訂定「便利商店職場不法侵害預防安全衛生指引」外，亦將列為勞動檢查方針。為響應主管機關推動職場不法侵害預防，公司透過跨單位合作，逐項檢視各項執行作為，設定短、中、長期執行目標，從軟體到硬體，進行檢視、補強與強化，增加安全保護機制營造友善職場，相關執行作為如下："
Output:
{{"promise_status":"Yes","promise_string":"為響應主管機關推動職場不法侵害預防，公司透過跨單位合作，逐項檢視各項執行作為，設定短、中、長期執行目標，從軟體到硬體，進行檢視、補強與強化，增加安全保護機制營造友善職場，","verification_timeline":"between_2_and_5_years","evidence_status":"No","evidence_string":"","evidence_quality":"N/A"}}
"""

def build_prompt(text, esg_type=None):
    return f"""{SYSTEM_PROMPT}

{FEW_SHOT_EXAMPLES}
Now classify the following text:

Text: "{text}"
ESG type (if any): "{esg_type or ''}"
"""


# ==========================================
# 5. Function: call LLM for ESG analysis
# ==========================================

VALID_PROMISE = {"Yes", "No"}
VALID_TIMELINE = {"already", "within_2_years", "between_2_and_5_years", "longer_than_5_years", "N/A"}
VALID_EVIDENCE = {"Yes", "No", "N/A"}
VALID_QUALITY = {"N/A", "Not Clear", "Clear", "Misleading"}

def parse_json_response(content):
    """Parse JSON from LLM response. Try json.loads first, then regex fallback."""
    # Try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Non-greedy regex to find first JSON object
    match = re.search(r"\{.*?\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Greedy fallback
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise ValueError(f"No valid JSON found in response: {content[:200]}")


def validate_result(result):
    """Validate and enforce cascading logic on parsed result."""
    # Validate promise_status
    if result.get("promise_status") not in VALID_PROMISE:
        result["promise_status"] = "No"

    # Enforce cascading: if No promise, force all downstream to N/A/empty
    if result["promise_status"] == "No":
        result["promise_string"] = ""
        result["verification_timeline"] = "N/A"
        result["evidence_status"] = "N/A"
        result["evidence_string"] = ""
        result["evidence_quality"] = "N/A"
        return result

    # Validate timeline
    if result.get("verification_timeline") not in VALID_TIMELINE:
        result["verification_timeline"] = "between_2_and_5_years"

    # Validate evidence_status
    if result.get("evidence_status") not in VALID_EVIDENCE:
        result["evidence_status"] = "N/A"

    # Enforce: if evidence=No or N/A, quality must be N/A
    if result.get("evidence_status") in ("No", "N/A"):
        result["evidence_string"] = ""
        result["evidence_quality"] = "N/A"
    else:
        if result.get("evidence_quality") not in VALID_QUALITY:
            result["evidence_quality"] = "N/A"

    return result


def call_esg_llm(text, esg_type=None, model=None, max_retries=2):
    model = model or MODEL
    prompt = build_prompt(text, esg_type)

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=PARAMS["temperature"],
            )
            content = resp.choices[0].message.content.strip()
            result = parse_json_response(content)
            return validate_result(result)

        except Exception as e:
            if attempt < max_retries:
                print(f"  Retry {attempt+1}/{max_retries} for model {model}: {e}")
                continue
            print(f"  LLM error after {max_retries+1} attempts: {e}")
            return {
                "promise_status": "No",
                "promise_string": "",
                "verification_timeline": "N/A",
                "evidence_status": "N/A",
                "evidence_string": "",
                "evidence_quality": "N/A"
            }


# ==========================================
# 6. Process each row incrementally
# ==========================================
batch_save = []

rows_to_process = list(df_input.itertuples(index=False))
if args.limit > 0:
    rows_to_process = rows_to_process[:args.limit]

for row in tqdm(rows_to_process, total=len(rows_to_process)):
    id_str = str(row.id)
    if id_str in processed_ids:
        continue

    data_text = row.data
    esg_type = getattr(row, 'esg_type', "")

    llm_result = call_esg_llm(data_text, esg_type)

    new_row = {
        "id": row.id,
        "data": data_text,
        "esg_type": esg_type,
        "promise_status": llm_result.get("promise_status", ""),
        "promise_string": llm_result.get("promise_string", ""),
        "verification_timeline": llm_result.get("verification_timeline", ""),
        "evidence_status": llm_result.get("evidence_status", ""),
        "evidence_string": llm_result.get("evidence_string", ""),
        "evidence_quality": llm_result.get("evidence_quality", "")
    }

    batch_save.append(new_row)
    processed_ids.add(id_str)

    if len(batch_save) >= BATCH_SIZE:
        df_output = pd.concat([df_output, pd.DataFrame(batch_save)], ignore_index=True)
        df_output.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        batch_save = []

# Save remaining rows
if batch_save:
    df_output = pd.concat([df_output, pd.DataFrame(batch_save)], ignore_index=True)
    df_output.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"Processing complete. Saved to: {OUTPUT_CSV}")
