import os
import pandas as pd
import json, re
from openai import OpenAI
from tqdm import tqdm

# ==========================================
# 1. Paths
# ==========================================
INPUT_CSV = r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_10 V1.csv"
OUTPUT_CSV = r"C:\Users\VU\Documents\NLP\AICup26\datasets\output_esg_10.csv"

# ==========================================
# 2. LLM Client
# ==========================================
BASE_URL = "http://localhost:11434/v1"
API_KEY = "not-needed"
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ==========================================
# 3. Load input and existing output
# ==========================================
df_input = pd.read_csv(INPUT_CSV)
print("Input rows:", len(df_input))

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
# 4. Function: call LLM for ESG analysis
# ==========================================

def call_esg_llm(text, esg_type=None):
    prompt = f"""
    You are an ESG (Environmental, Social, Governance) analyst and data annotator.
    Your task is to carefully analyze corporate ESG disclosures and commitments.

    Instructions:
    1. Return ONLY a single valid JSON object with EXACT keys:
       promise_status, promise_string, verification_timeline,
       evidence_status, evidence_string, evidence_quality
    2. Do NOT include greetings, explanations, or extra text.
    3. All keys must be present.
    4. Use the following rules:

    - promise_status: "Yes" if the text shows a commitment; "No" if there is no commitment.
    - promise_string: a concise excerpt from the text that shows the commitment.
    - verification_timeline:
        - If promise_status is "No", return "N/A".
        - "already" if the action is already implemented or ongoing.
        - "within_2_years" if the commitment will be achieved within 2 years.
        - "between_2_and_5_years" if the target is 2–5 years away.
        - "longer_than_5_years" if the target is more than 5 years away.
        - If a specific year is mentioned, calculate the difference between current year 2025 and the target year to assign the correct label.
        - If only qualitative terms like "short-term", "medium-term", or "long-term" appear, map as:
            short-term → within_2_years
            medium-term → between_2_and_5_years
            long-term → longer_than_5_years
    - evidence_status: "Yes" if there is supporting evidence, "No" if none, "N/A" if not applicable.
    - evidence_string: the part of the text that provides supporting evidence (empty if none).
    - evidence_quality: one of "N/A", "Not Clear", "Clear", "Misleading"

    Example 1:
    Text: "聯發科技除在「工作規則」中依照勞基法明確規定「員工在產假期間公司不得終止勞動契約」外，為支持同仁與其家人度過人生不同階段，自 2024 年起提供女性員工在分娩前後計有 12 週共 84 天的產假；男性員工則可於其配偶懷孕期間陪同產檢或生（流）產日及前後 15 日間請假陪伴，兩者合計共有 10 天陪產（檢）假可運用，陪產（檢）假期間工資照常給付。"
    Output:
    {{"promise_status":"Yes","promise_string":"為支持同仁與其家人度過人生不同階段，自 2024 年起提供女性員工在分娩前後計有 12 週共 84 天的產假；男性員工則可於其配偶懷孕期間陪同產檢或生（流）產日及前後 15 日間請假陪伴，兩者合計共有 10 天陪產（檢）假可運用，陪產（檢）假期間工資照常給付。","verification_timeline":"already","evidence_status":"No","evidence_string":"","evidence_quality":"N/A"}}

    Example 2:
    Text: "在與供應商攜手邁向永續發展的過程中，台泥致力於建立一套以合作為基礎的價值體體系。核心原則是推動供應鏈的低碳轉型，同時維護人權、保護環境和促進生物多樣性，以建立永續的合作夥伴關係。"
    Output:
    {{"promise_status":"Yes","promise_string":"台泥致力於建立一套以合作為基礎的價值體體系。","verification_timeline":"between_2_and_5_years","evidence_status":"Yes","evidence_string":"核心原則是推動供應鏈的低碳轉型，同時維護人權、保護環境和促進生物多樣性，以建立永續的合作夥伴關係。","evidence_quality":"Not Clear"}}

    Example 3:
    Text: "III. 產品危害物質減免 (HSF) 短、中、長期目標設定    **1. 2024 年 目標和實績**  *   已達標  *   詳述於 "產品各階段危害物質減免風險管理目標與成效"    **2. 短期目標 2025~2026 年**  *   監控 PFAS 中有害物質管制要求  *   減少 RoHS 豁免零件之使用，往 Lead Free 持續前進。    **3. 中長期目標 2026~2027 年**  *   優化綠色管理系統審查作業提升效率。"
    Output:
    {{"promise_status":"Yes","promise_string":"III. 產品危害物質減免 (HSF) 短、中、長期目標設定","verification_timeline":"between_2_and_5_years","evidence_status":"Yes","evidence_string":"1. 2024 年 目標和實績**  *   已達標  *   詳述於 "產品各階段危害物質減免風險管理目標與成效"    **2. 短期目標 2025~2026 年**  *   監控 PFAS 中有害物質管制要求  *   減少 RoHS 豁免零件之使用，往 Lead Free 持續前進。    **3. 中長期目標 2026~2027 年**  *   優化綠色管理系統審查作業提升效率。","evidence_quality":"Not Clear"}}

    Example 4:
    Text: "面對這些全球與國內政策的推動，富邦人壽秉持永續經營理念，積極透過永續金融的投資力量，引導被投資公司與企業進行永續轉型，除非資金明確用於綠能轉型計畫，不再新增投資燃煤比重超過 50% 的電廠。同時，針對燃料煤開採、運輸業、燃料煤發電及非典型油氣產業，制定嚴格的准入與撤資標準，積極引導資金流向低碳與可再生能源領域，展現對環境永續的堅定承諾，並連續 4 年榮獲「台灣永續投資典範機構獎 – 機構影響力 (壽險組)」殊榮，在責任投資與推動企業永續發展上的卓越表現備受肯定。"
    Output:
    {{"promise_status":"Yes","promise_string":"秉持永續經營理念，積極透過永續金融的投資力量，引導被投資公司與企業進行永續轉型，","verification_timeline":"already","evidence_status":"Yes","evidence_string":"不再新增投資燃煤比重超過 50% 的電廠。 ｜ 制定嚴格的准入與撤資標準，積極引導資金流向低碳與可再生能源領域，展現對環境永續的堅定承諾，並連續 4 年榮獲「台灣永續投資典範機構獎 – 機構影響力 (壽險組)」殊榮，","evidence_quality":"Clear"}}

    Example 5:
    Text: "關注人才關鍵議題，新加坡亞洲新聞臺（CNA）特別製作專題《Taiwan’s tech industry taps female talent pool amid labour shortage》，在報導中聚焦基金會所舉辦的 Girls! TECH Action 科技女孩工作坊，探討企業如何透過教育計畫來翻轉科技產業中的性別失衡，有效地減少女性在 STEM 領域的管漏現象，並鼓勵更多國高中女生投入 STEM（科學、技術、工程與數學）領域，並充分發揮女性在科技創新的潛力！在全球科技人才短缺的挑戰下，臺灣和新加坡對此議題都表現出高度關注。兩國的科技產業正積極尋找方法挖掘女性人才的潛力，希望藉此減緩人才短缺問題，並促進產業的多元樣貌化發展。透過教育和相關計畫，致力於提升女性在 STEM 領域的參與，確保科技創新能夠從更多元的觀點中受益，幫助解決當前的人才需求挑戰。"
    Output:
    {{"promise_status":"Yes","promise_string":"透過教育和相關計畫，致力於提升女性在 STEM 領域的參與，確保科技創新能夠從更多元的觀點中受益，幫助解決當前的人才需求挑戰。","verification_timeline":"already","evidence_status":"Yes","evidence_string":"關注人才關鍵議題，新加坡亞洲新聞臺（CNA）特別製作專題《Taiwan’s tech industry taps female talent pool amid labour shortage》，在報導中聚焦基金會所舉辦的 Girls! TECH Action 科技女孩工作坊，探討企業如何透過教育計畫來翻轉科技產業中的性別失衡，有效地減少女性在 STEM 領域的管漏現象，並鼓勵更多國高中女生投入 STEM（科學、技術、工程與數學）領域，並充分發揮女性在科技創新的潛力！在","evidence_quality":"Clear"}}

    Example 6:
    Text: "本集團對所使用的化學品已100%進行識別，主動管理危害物質，並對限制/監測使用的物質提前部署消減計劃，致力實現有害物質減免（「HSF」）。目前，集團79%（重量比）的高度關注物質已完成汰換方案的制定，剩餘之21%，因工藝限制、技術挑戰等複雜因素，其汰換方案尚處於積極討論階段。我們將持續關注這些高度關注物質並於官網及時更新其汰換與推動進展。"
    Output:
    {{"promise_status":"Yes","promise_string":"致力實現有害物質減免（「HSF」）。 ｜ 我們將持續關注這些高度關注物質並於官網及時更新其汰換與推動進展。","verification_timeline":"between_2_and_5_years","evidence_status":"Yes","evidence_string":"目前，集團79%（重量比）的高度關注物質已完成汰換方案的制定，剩餘之21%，因工藝限制、技術挑戰等複雜因素，其汰換方案尚處於積極討論階段。我","evidence_quality":"Clear"}}

    Example 7:
    Text: "統一超商積極透過推動綠色採購管理設備、耗材與建材，選擇綠建材進行門市裝修並採購取得節能標章、環保標章或驗證或具有實際環保效益的設備與耗材應用於門市，2023 年擴大範圍率通路之先全面採用「FSC 森林永續認證」咖啡紙杯，採購金額為 617,979 仟元，相較 2023 年成長近 2 倍，在維持營運順暢與服務品質的同時，也降低消耗天然資源與環境負面影響。我們 2024 年綠色採購總金額達 27.82 億元，佔統一超商整年度採購總金額達 20.73%，達成原定年度採購佔比達 15% 之目標，並將 2025 年目標設定提高至 18%。"
    Output:
    {{"promise_status":"Yes","promise_string":"統一超商積極透過推動綠色採購管理設備、耗材與建材，選擇綠建材進行門市裝修並採購取得節能標章、環保標章或驗證或具有實際環保效益的設備與耗材應用於門市，","verification_timeline":"between_2_and_5_years","evidence_status":"Yes","evidence_string":"2023 年擴大範圍率通路之先全面採用「FSC 森林永續認證」咖啡紙杯，採購金額為 617,979 仟元，相較 2023 年成長近 2 倍，在維持營運順暢與服務品質的同時，也降低消耗天然資源與環境負面影響。我們 2024 年綠色採購總金額達 27.82 億元，佔統一超商整年度採購總金額達 20.73%，達成原定年度採購佔比達 15% 之目標，並將 2025 年目標設定提高至 18%。","evidence_quality":"Clear"}}

    Example 8:
    Text: "和碩由董事長童子賢先生公開宣誓集團對長期節能減碳的決心，期能在集團的共同努力下，對全球溫室氣體減量能有所貢獻。為落實和碩之溫室氣體管理措施及政策，企總及各主要製造廠區皆成立溫室氣體盤查委員會，進行溫室氣體盤查與管理，釐清轄屬的溫室氣體排放源，以此為依據進一步擬定減量計畫及設定減量目標，降低因業務、生產、員工活動或服務所帶來對環境的衝擊，善盡保護地球環境之責任。"
    Output:
    {{"promise_status":"Yes","promise_string":"為落實和碩之溫室氣體管理措施及政策，","verification_timeline":"between_2_and_5_years","evidence_status":"Yes","evidence_string":"企總及各主要製造廠區皆成立溫室氣體盤查委員會，進行溫室氣體盤查與管理，釐清轄屬的溫室氣體排放源，以此為依據進一步擬定減量計畫及設定減量目標，降低因業務、生產、員工活動或服務所帶來對環境的衝擊，善盡保護地球環境之責任。","evidence_quality":"Clear"}}

    Example 9:
    Text: "近年來不時有便利商店發生職場暴力攻擊事件，勞動部訂定「便利商店職場不法侵害預防安全衛生指引」外，亦將列為勞動檢查方針。為響應主管機關推動職場不法侵害預防，公司透過跨單位合作，逐項檢視各項執行作為，設定短、中、長期執行目標，從軟體到硬體，進行檢視、補強與強化，增加安全保護機制營造友善職場，相關執行作為如下："
    Output:
    {{"promise_status":"Yes","promise_string":"為響應主管機關推動職場不法侵害預防，公司透過跨單位合作，逐項檢視各項執行作為，設定短、中、長期執行目標，從軟體到硬體，進行檢視、補強與強化，增加安全保護機制營造友善職場，","verification_timeline":"between_2_and_5_years","evidence_status":"No","evidence_string":"","evidence_quality":"N/A"}}

    Example 10:
    Text: "台光電子中山廠區 2024 失能傷害頻率 (FR) 平均為 0.9，失能傷害嚴重率 (SR) (取至整數) 為 2715，與過去一年相比，失能傷害頻率 (FR) 上升及失能傷害嚴重率 (SR) 上升 (2023 年 FR 及 SR 為 0、0)。主因為 2024 年發生一件人員走路扭傷腳、一件員工死亡事故，導致失能傷害頻率 (FR) 上升及失能傷害嚴重率 (SR) 上升。改善措施，加強員工安全宣導以提高員工安全意識；全面盤查機械設備，升級安全防護。"
    Output:
    {{"promise_status":"Yes","promise_string":"善措施，加強員工安全宣導以提高員工安全意識；全面盤查機械設備，升級安全防護。","verification_timeline":"between_2_and_5_years","evidence_status":"Yes","evidence_string":"失能傷害嚴重率 (SR) (取至整數) 為 2715，與過去一年相比，失能傷害頻率 (FR) 上升及失能傷害嚴重率 (SR) 上升 (2023 年 FR 及 SR 為 0、0)。主因為 2024 年發生一件人員走路扭傷腳、一件員工死亡事故，導致失能傷害頻率 (FR) 上升及失能傷害嚴重率 (SR) 上升。","evidence_quality":"Clear"}}

    Now classify the following text:

    Text: "{text}"
    ESG type (if any): "{esg_type}"
    """

    try:
        resp = client.chat.completions.create(
            model="llama3:70b",
            messages=[{"role":"user","content":prompt}],
            temperature=0.0
        )
        content = resp.choices[0].message.content.strip()

        # --- New: extract first JSON object in text ---
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            json_text = match.group()
            result = json.loads(json_text)
        else:
            print("⚠️ No JSON found in LLM response:", content)
            raise ValueError("No JSON")

        # Post-check for evidence_status
        if result.get("evidence_status") not in ["Yes", "No", "N/A"]:
            result["evidence_status"] = "N/A"

        # Post-check for evidence_quality
        if result.get("evidence_quality") not in ["N/A", "Not Clear", "Clear", "Misleading"]:
            result["evidence_quality"] = "N/A"

        return result

    except Exception as e:
        print(f"⚠️ LLM error: {e}")
        return {
            "promise_status":"No",
            "promise_string":"",
            "verification_timeline":"N/A",
            "evidence_status":"N/A",
            "evidence_string":"",
            "evidence_quality":"N/A"
        }

# ==========================================
# 5. Process each row incrementally
# ==========================================
batch_save = []
BATCH_SIZE = 20

for row in tqdm(df_input.itertuples(index=False), total=len(df_input)):
    id_str = str(row.id)
    if id_str in processed_ids:
        continue  # skip already processed

    data_text = row.data
    esg_type = getattr(row, 'esg_type', "")

    # Call LLM
    llm_result = call_esg_llm(data_text, esg_type)

    # Prepare new row
    new_row = {
        "id": row.id,
        "data": data_text,
        "esg_type": esg_type,
        "promise_status": llm_result.get("promise_status",""),
        "promise_string": llm_result.get("promise_string",""),
        "verification_timeline": llm_result.get("verification_timeline",""),
        "evidence_status": llm_result.get("evidence_status",""),
        "evidence_string": llm_result.get("evidence_string",""),
        "evidence_quality": llm_result.get("evidence_quality","")
    }

    batch_save.append(new_row)
    processed_ids.add(id_str)

    # Save per batch
    if len(batch_save) >= BATCH_SIZE:
        df_output = pd.concat([df_output, pd.DataFrame(batch_save)], ignore_index=True)
        df_output.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        batch_save = []

# Save remaining rows
if batch_save:
    df_output = pd.concat([df_output, pd.DataFrame(batch_save)], ignore_index=True)
    df_output.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("✅ Processing complete. Saved to:", OUTPUT_CSV)