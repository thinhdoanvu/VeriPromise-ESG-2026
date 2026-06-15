r"""
S4 ALL-IN-ONE — BERT (batch) + CoT (concurrent LLM) + merge strategies
==========================================================================
1 lan chay, khong file trung gian:
  - Load 168 held-out-200 eligible
  - BERT b/c/d (batch, ~5-10 phut)
  - CoT (LLM council, 6 fixed examples, concurrent, ~15-40 phut)
  - Tinh va in 6 strategies: A/B/E/F/G/H (giong merge_cot_bert_s4.py)

Chay truc tiep trong PyCharm (Run).
"""
import sys, json, random, re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import classification_report, f1_score

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, r"C:\Users\VU\Documents\NLP\AICup26\llm-council\backend")
try:
    from config import client, ModelManager
    print("OK Config loaded.")
except ImportError as e:
    print(f"ERROR loading config: {e}")
    sys.exit(1)

import torch
from transformers import pipeline as hf_pipeline

TRAIN_JSON = r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_1000 V1.json"
SEED       = 42
DEVICE     = 0 if torch.cuda.is_available() else -1
COT_MAX_TOKENS = 300
LLM_WORKERS    = 4

VALID_S4 = ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years"]
VOTERS   = ["qwen2.5-72b-esg", "deepseek-r1-70b-esg"]
CHAIRMAN = "qwen2.5-32b-esg"
LABEL_SYNONYMS = {"longer_than_5_years": "more_than_5_years"}

BERT_PATHS = {
    "b": r"D:\LLMs\BERT-ESG\grid_s4_v1\s4-v1-b",
    "c": r"D:\LLMs\BERT-ESG\grid_s4_v1\s4-v1-c",
    "d": r"D:\LLMs\BERT-ESG\grid_s4_v1\s4-v1-d",
}

# ==========================================
# LAYER1 (regex, khong dung trong final nhung giu de tham khao H)
# ==========================================
COMMIT_VERBS_RE = re.compile(
    r"(目標|計畫|預計|將於|承諾|規劃|達成|完成|邁向|推動|布局|落實|展開|啟動|目標年)"
)
YEAR_RE = re.compile(r"20[2-5][0-9]")

def layer1_override(data):
    for m in YEAR_RE.finditer(data):
        year = int(m.group())
        if year < 2025:
            continue
        window = data[max(0, m.start()-15): m.end()+15]
        if COMMIT_VERBS_RE.search(window):
            diff = year - 2024
            if diff <= 2:
                return "within_2_years"
            if diff <= 5:
                continue
            return "more_than_5_years"
    return None

# ==========================================
# COT PROMPT (6 fixed examples)
# ==========================================
COT_EXAMPLES = [
    {"data": "公司持續推動員工教育訓練計畫，提升整體職場安全意識，並定期舉辦相關講座。",
     "reasoning": "No specific year is mentioned. This describes an ongoing "
                   "activity with no future deadline.",
     "answer": "already"},
    {"data": "2023年公司已完成ISO 14001認證更新，並持續維持該管理系統運作至今。",
     "reasoning": "The only year mentioned is 2023, which is before 2024. "
                   "The certification is already completed and being "
                   "maintained. No future deadline.",
     "answer": "already"},
    {"data": "公司於2024年啟動供應商行為準則簽署計畫，預計2025年底完成全數供應商簽署。",
     "reasoning": "2024 is the baseline/start year (program launch). 2025 "
                   "is the target completion year. 2025-2024=1 year.",
     "answer": "within_2_years"},
    {"data": "為強化資訊安全管理，公司目標於2028年取得ISO 27001全公司認證，"
             "目前正進行各部門盤點作業。",
     "reasoning": "2028 is explicitly stated as the target year for "
                   "obtaining certification. 2028-2024=4 years.",
     "answer": "between_2_and_5_years"},
    {"data": "公司以全球據點推動2050年再生能源使用比例達100%之目標，"
             "並逐年提升現有廠區的綠電採購比例。",
     "reasoning": "2050 is explicitly stated as the target year for the "
                   "100% renewable energy goal. 2050-2024=26 years, "
                   "more than 5 years.",
     "answer": "more_than_5_years"},
    {"data": "公司秉持誠信經營理念，致力於提升企業治理透明度，"
             "並持續優化內部風險控管機制。",
     "reasoning": "No years are mentioned. This describes a general "
                   "ongoing corporate philosophy/practice with no "
                   "specific future deadline.",
     "answer": "already"},
]

def build_prompt_s4_cot(query_data):
    few_shot = "".join(
        f"Statement: {ex['data']}\n"
        f"Reasoning: {ex['reasoning']}\n"
        f"Answer: {ex['answer']}\n\n"
        for ex in COT_EXAMPLES
    )
    return (
        "You are an ESG analyst. The ESG report was published in 2024.\n"
        "Classify the expected completion timeframe of the commitment.\n"
        "The report year is 2024. Calculate years from 2024.\n"
        "Choose exactly one:\n"
        "- already        : action already done or ongoing AS OF 2024, no specific future deadline\n"
        "- within_2_years : deadline is year 2025 or 2026  (1-2 years from 2024)\n"
        "- between_2_and_5_years : deadline is year 2027, 2028, or 2029  (3-5 years from 2024)\n"
        "- more_than_5_years     : deadline is year 2030 or later  (more than 5 years from 2024)\n\n"
        "For each statement, first identify which years (if any) are "
        "mentioned and whether each is a baseline/historical reference or "
        "a future commitment deadline. Then give your final answer after "
        "'Answer:'.\n\n"
        f"{few_shot}"
        f"Statement: {query_data}\n"
        "Reasoning:"
    )


def call_llm(model_name, prompt):
    params  = ModelManager.get_params("extraction")
    timeout = ModelManager.get_timeout(model_name)
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=COT_MAX_TOKENS,
            top_p=params["top_p"],
            timeout=timeout,
        )
        raw = resp.choices[0].message.content.strip()
        if "<think>" in raw:
            end = raw.find("</think>")
            if end == -1:
                return "UNKNOWN"
            raw = raw[end+len("</think>"):].strip()

        search_text = raw
        if "answer:" in raw.lower():
            idx = raw.lower().rfind("answer:")
            search_text = raw[idx+len("answer:"):].strip()

        for label in VALID_S4:
            if label.lower() in search_text.lower():
                return label
        for old_label, new_label in LABEL_SYNONYMS.items():
            if old_label.lower() in search_text.lower() and new_label in VALID_S4:
                return new_label
        for label in VALID_S4:
            if label.lower() in raw.lower():
                return label
        return "UNKNOWN"
    except Exception:
        return "ERROR"


def council_cot(data):
    prompt = build_prompt_s4_cot(data)
    votes = {v: call_llm(v, prompt) for v in VOTERS}
    clean   = [vt for vt in votes.values() if vt in VALID_S4]
    counter = Counter(clean)
    council_majority = None
    if counter:
        top = counter.most_common(1)[0]
        council_majority = top[0] if top[1] > 1 else None
    if council_majority is None:
        chair = call_llm(CHAIRMAN, prompt)
        result = chair if chair in VALID_S4 else None
    else:
        result = council_majority
    if result not in VALID_S4:
        result = "already"
    return result


# ==========================================
# LOAD DATA
# ==========================================
def normalize_timeline(records):
    for r in records:
        if r.get("verification_timeline") == "longer_than_5_years":
            r["verification_timeline"] = "more_than_5_years"
    return records

print("Loading data...", flush=True)
with open(TRAIN_JSON, "r", encoding="utf-8") as f:
    train_records = json.load(f)
train_records = normalize_timeline(train_records)

random.seed(SEED)
random.shuffle(train_records)
test_200 = train_records[800:]

eligible = [
    r for r in test_200
    if str(r.get("promise_status","")).strip()=="Yes"
    and str(r.get("verification_timeline","")).strip() in VALID_S4
]
print(f"Eligible: {len(eligible)} rows", flush=True)

ids   = [str(r.get("id")) for r in eligible]
texts = [str(r["data"]).strip() for r in eligible]
gts   = [str(r["verification_timeline"]).strip() for r in eligible]

# ==========================================
# STEP 1: BERT b/c/d (batch)
# ==========================================
preds = {}
for name, path in BERT_PATHS.items():
    print(f"\n[BERT] Loading {name} from {path} ...", flush=True)
    clf = hf_pipeline("text-classification", model=path, tokenizer=path,
                       device=DEVICE, truncation=True, max_length=512, batch_size=16)
    outs = clf(texts)
    preds[name] = [o["label"] if o["label"] in VALID_S4 else "already" for o in outs]
    print(f"  Distribution: {dict(Counter(preds[name]))}", flush=True)
    del clf
    torch.cuda.empty_cache()

layer1 = [layer1_override(t) for t in texts]
print(f"\n[Layer1] fires on {sum(1 for x in layer1 if x)} rows", flush=True)

# ==========================================
# STEP 2: CoT (concurrent)
# ==========================================
print(f"\n[CoT] Running with {LLM_WORKERS} concurrent workers...", flush=True)
cot_preds = [None] * len(texts)
completed = 0
with ThreadPoolExecutor(max_workers=LLM_WORKERS) as executor:
    futures = {executor.submit(council_cot, texts[i]): i for i in range(len(texts))}
    for fut in as_completed(futures):
        i = futures[fut]
        cot_preds[i] = fut.result()
        completed += 1
        if completed % 20 == 0:
            print(f"  [CoT] {completed}/{len(texts)} done...", flush=True)

print(f"  [CoT] Distribution: {dict(Counter(cot_preds))}", flush=True)

# ==========================================
# STEP 3: STRATEGIES
# ==========================================
def ensemble_D(i):
    if preds["b"][i] == "already" and preds["c"][i] in {"between_2_and_5_years","more_than_5_years"} \
       and preds["c"][i] == preds["d"][i]:
        return preds["c"][i]
    return preds["b"][i]

def strategy_F(i):
    if cot_preds[i] == "already":
        return preds["b"][i]
    return cot_preds[i]

def strategy_G(i):
    if cot_preds[i] == "already":
        return ensemble_D(i)
    return cot_preds[i]

def strategy_H(i):
    if cot_preds[i] == "already":
        return layer1[i] if layer1[i] else preds["b"][i]
    return cot_preds[i]


def report_for(fn, name):
    y_true = [VALID_S4.index(gts[i]) for i in range(len(texts))]
    y_pred = []
    for i in range(len(texts)):
        p = fn(i)
        if p not in VALID_S4:
            p = "already"
        y_pred.append(VALID_S4.index(p))
    macro = f1_score(y_true, y_pred, average="macro")
    print(f"\n{'='*60}")
    print(f"  {name} — Macro F1 = {macro:.4f}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=VALID_S4,
                                  labels=list(range(len(VALID_S4))), zero_division=0))
    return macro


results = {}
results["A. BERT b alone"]                       = report_for(lambda i: preds["b"][i], "A. BERT b alone")
results["B. Ensemble D (b+c+d)"]                 = report_for(ensemble_D, "B. Ensemble D (b+c+d)")
results["E. CoT alone"]                          = report_for(lambda i: cot_preds[i], "E. CoT alone")
results["F. CoT primary + BERT-b on already"]    = report_for(strategy_F, "F. CoT primary + BERT-b on CoT='already'")
results["G. CoT primary + EnsembleD on already"] = report_for(strategy_G, "G. CoT primary + EnsembleD on CoT='already'")
results["H. CoT + Layer1/BERT-b on already"]     = report_for(strategy_H, "H. CoT + (Layer1 or BERT-b) on CoT='already'")

print("\nSUMMARY:")
baseline = results["A. BERT b alone"]
for name, macro in results.items():
    delta = macro - baseline
    print(f"  {name:<45} {macro:.4f}  (delta vs A: {delta:+.4f})")

"""
============================================================
  A. BERT b alone — Macro F1 = 0.6338
============================================================
                       precision    recall  f1-score   support

              already       0.62      0.77      0.69        74
       within_2_years       0.50      0.67      0.57         3
between_2_and_5_years       0.67      0.46      0.54        61
    more_than_5_years       0.73      0.73      0.73        30

             accuracy                           0.65       168
            macro avg       0.63      0.66      0.63       168
         weighted avg       0.65      0.65      0.64       168


============================================================
  B. Ensemble D (b+c+d) — Macro F1 = 0.6487
============================================================
                       precision    recall  f1-score   support

              already       0.67      0.68      0.67        74
       within_2_years       0.50      0.67      0.57         3
between_2_and_5_years       0.63      0.59      0.61        61
    more_than_5_years       0.72      0.77      0.74        30

             accuracy                           0.66       168
            macro avg       0.63      0.67      0.65       168
         weighted avg       0.66      0.66      0.66       168


============================================================
  E. CoT alone — Macro F1 = 0.4817
============================================================
                       precision    recall  f1-score   support

              already       0.50      0.96      0.66        74
       within_2_years       0.50      1.00      0.67         3
between_2_and_5_years       0.80      0.07      0.12        61
    more_than_5_years       0.69      0.37      0.48        30

             accuracy                           0.53       168
            macro avg       0.62      0.60      0.48       168
         weighted avg       0.64      0.53      0.43       168


============================================================
  F. CoT primary + BERT-b on CoT='already' — Macro F1 = 0.6403
============================================================
                       precision    recall  f1-score   support

              already       0.62      0.74      0.68        74
       within_2_years       0.50      1.00      0.67         3
between_2_and_5_years       0.68      0.46      0.55        61
    more_than_5_years       0.64      0.70      0.67        30

             accuracy                           0.64       168
            macro avg       0.61      0.73      0.64       168
         weighted avg       0.65      0.64      0.63       168


============================================================
  G. CoT primary + EnsembleD on CoT='already' — Macro F1 = 0.6553
============================================================
                       precision    recall  f1-score   support

              already       0.68      0.65      0.66        74
       within_2_years       0.50      1.00      0.67         3
between_2_and_5_years       0.64      0.59      0.62        61
    more_than_5_years       0.63      0.73      0.68        30

             accuracy                           0.65       168
            macro avg       0.61      0.74      0.66       168
         weighted avg       0.65      0.65      0.65       168


============================================================
  H. CoT + (Layer1 or BERT-b) on CoT='already' — Macro F1 = 0.6075
============================================================
                       precision    recall  f1-score   support

              already       0.63      0.74      0.68        74
       within_2_years       0.38      1.00      0.55         3
between_2_and_5_years       0.68      0.44      0.53        61
    more_than_5_years       0.64      0.70      0.67        30

             accuracy                           0.63       168
            macro avg       0.58      0.72      0.61       168
         weighted avg       0.64      0.63      0.62       168


SUMMARY:
  A. BERT b alone                               0.6338  (delta vs A: +0.0000)
  B. Ensemble D (b+c+d)                         0.6487  (delta vs A: +0.0149)
  E. CoT alone                                  0.4817  (delta vs A: -0.1521)
  F. CoT primary + BERT-b on already            0.6403  (delta vs A: +0.0065)
  G. CoT primary + EnsembleD on already         0.6553  (delta vs A: +0.0215)
  H. CoT + Layer1/BERT-b on already             0.6075  (delta vs A: -0.0263)
"""