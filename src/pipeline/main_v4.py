"""
ESG Promise Verification — Pipeline v7
==========================================
S1 (promise_status):        BERT MacBERT-large       Macro F1=0.8282  (BATCHED)
S2 (evidence_status):       BERT RoBERTa-wwm-large   Macro F1=0.7952  (BATCHED, grid-c MOI)
S3 (evidence_quality):      BERT MacBERT-large       Macro F1=0.7003  (BATCHED)
S4 (verification_timeline): RAG + LLM Council        3 voters + chairman
                             + rule_based_s4 safety net  (CONCURRENT, resume)

Thay doi vs v4:
  - S1/S2/S3: BATCH INFERENCE (HF pipeline + batch_size) thay vi
    row-by-row -> nhanh hon nhieu, giong v6
  - S2: dung model MOI (roberta-wwm-large-s2-v1, grid-c, val
    Macro F1=0.6591 -> 0.7952)
  - S4: GIU LLM Council + RAG (theo yeu cau), nhung CONCURRENT
    (ThreadPoolExecutor) qua nhieu rows -> giam thoi gian cho doi
    Ollama tra loi tung row mot. Resume support giu nguyen cho
    stage nay (phan cham nhat).

Usage:
    python main_pipeline_v7.py
    python main_pipeline_v7.py --batch-size 64 --llm-workers 4
    python main_pipeline_v7.py --rebuild-rag
"""

import os
import sys
import argparse
import json
import random
import re
import threading
import warnings
import pandas as pd
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from transformers import pipeline as hf_pipeline
import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# LOAD CONFIG — client + ModelManager (cho LLM Council S4)
# ==========================================
sys.path.insert(0, r"C:\Users\VU\Documents\NLP\AICup26\llm-council\backend")
try:
    from config import client, ModelManager
    print("OK Config loaded.")
except ImportError as e:
    print(f"ERROR loading config: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ==========================================
# CONFIG
# ==========================================
BERT_MODELS = {
    "s1": r"D:\LLMs\BERT-ESG\macbert-large-s1-v6",       # Macro F1=0.8282
    "s2": r"D:\LLMs\BERT-ESG\roberta-wwm-large-s2-v1",   # grid-c -> Macro F1=0.7952 (NEW)
    "s3": r"D:\LLMs\BERT-ESG\macbert-large-s3-v6",       # Macro F1=0.7003
}

PRIORITY_MAP = {
    "COUNCIL_TIMELINE": ["qwen2.5-72b-esg", "deepseek-r1-70b-esg"],
    "CHAIRMAN":         ["qwen2.5-32b-esg"],
}

VALID_LABELS = {
    "s1": ["Yes", "No"],
    "s2": ["Yes", "No"],
    "s3": ["Clear", "Not Clear"],
    "s4": ["already", "within_2_years", "between_2_and_5_years", "more_than_5_years"],
}

OUTPUT_COLUMNS = ["id", "promise_status", "verification_timeline",
                  "evidence_status", "evidence_quality"]

# ==========================================
# PATHS
# ==========================================
JSON_TRAIN  = r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_1000 V1.json"
VAL_JSON    = r"C:\Users\VU\Documents\NLP\AICup26\datasets\validation\vpesg4k_val_1000.json"
TEST_CSV    = r"C:\Users\VU\Documents\NLP\AICup26\datasets\test\vpesg4k_test_2000.csv"
OUTPUT_DIR  = r"C:\Users\VU\Documents\NLP\AICup26\results"
CHROMA_DIR  = r"C:\Users\VU\Documents\NLP\AICup26\chroma_db"
EMBED_MODEL = "BAAI/bge-large-zh-v1.5"
TOP_K       = 6
DEVICE      = 0 if torch.cuda.is_available() else -1
BATCH_SIZE_DEFAULT  = 32
LLM_WORKERS_DEFAULT = 4
COT_MAX_TOKENS      = 300  # reasoning + answer can be longer than 50 (deepseek/qwen)


# ==========================================
# BERT — batched
# ==========================================
def load_bert_model(task, batch_size):
    path = BERT_MODELS[task]
    abs_path = os.path.abspath(path)
    classifier = hf_pipeline(
        "text-classification",
        model=abs_path,
        tokenizer=abs_path,
        device=DEVICE,
        truncation=True,
        max_length=512,
        batch_size=batch_size,
    )
    print(f"  OK {task} - {os.path.basename(abs_path)}")
    return classifier


def batch_predict(classifier, texts, valid_labels, default):
    if not texts:
        return []
    outputs = classifier(texts)
    return [o["label"] if o["label"] in valid_labels else default for o in outputs]


# ==========================================
# CHROMADB — S4 RAG (giu nguyen v4)
# ==========================================
def load_rag_data(train_json, val_json):
    with open(train_json, "r", encoding="utf-8") as f:
        train_records = json.load(f)
    with open(val_json, "r", encoding="utf-8") as f:
        val_records = json.load(f)
    random.seed(42)
    random.shuffle(train_records)
    combined = train_records[:800] + val_records
    print(f"  RAG index: {len(combined)} rows (800 train + {len(val_records)} val)")
    return combined


def _list_collection_names(chroma_client):
    try:
        cols = chroma_client.list_collections()
    except Exception as e:
        print(f"  [WARN] list_collections() failed: {e}")
        return set()
    names = set()
    for c in cols:
        names.add(c if isinstance(c, str) else getattr(c, "name", str(c)))
    return names


def build_chroma_s4(train_records, force_rebuild=False):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

    existing = _list_collection_names(chroma_client)
    print(f"  ChromaDB: existing collections = {existing or '(none)'}")

    expected_n = sum(
        1 for r in train_records
        if str(r.get("promise_status", "")).strip() == "Yes"
        and str(r.get("verification_timeline", "")).strip() in VALID_LABELS["s4"]
    )

    if "esg_s4" in existing and not force_rebuild:
        try:
            col = chroma_client.get_collection("esg_s4", embedding_function=ef)
            n = col.count()
            print(f"  ChromaDB S4: loaded from cache ({n} docs), expected={expected_n}")
            if n == expected_n and n > 0:
                return col
            print(f"  [WARN] Cache count mismatch ({n} != {expected_n}) -> rebuilding.")
        except Exception as e:
            print(f"  [WARN] get_collection failed ({e}) -> rebuilding.")

    print("  ChromaDB S4: building index...")
    if "esg_s4" in existing:
        try:
            chroma_client.delete_collection("esg_s4")
        except Exception as e:
            print(f"  [WARN] delete_collection failed (ignored): {e}")

    col = chroma_client.create_collection(
        "esg_s4", embedding_function=ef, metadata={"hnsw:space": "cosine"}
    )
    docs, ids, metas = [], [], []
    for i, r in enumerate(train_records):
        if str(r.get("promise_status", "")).strip() != "Yes":
            continue
        tl = str(r.get("verification_timeline", "")).strip()
        if tl not in VALID_LABELS["s4"]:
            continue
        docs.append(str(r["data"]).strip())
        ids.append(f"s4_{i}")
        metas.append({"verification_timeline": tl})

    print(f"  Adding {len(docs)} docs to ChromaDB...")
    for i in range(0, len(docs), 100):
        col.add(documents=docs[i:i+100], ids=ids[i:i+100], metadatas=metas[i:i+100])
    print(f"  ChromaDB S4: built ({col.count()} docs)")
    return col


def retrieve_s4(col, query_text):
    labels    = VALID_LABELS["s4"]
    per_label = max(1, TOP_K // len(labels))
    examples  = []
    for label in labels:
        try:
            res = col.query(
                query_texts=[query_text], n_results=per_label,
                where={"verification_timeline": {"$eq": label}}
            )
            for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
                examples.append({"data": doc, "verification_timeline": meta["verification_timeline"]})
        except Exception:
            continue
    return examples[:TOP_K]


# ==========================================
# LLM — S4 prompt + council vote (giu nguyen v4)
# ==========================================
def build_prompt_s4(query_data, examples):
    few_shot = "".join(
        f"Statement: {ex['data']}\nAnswer: {ex['verification_timeline']}\n\n"
        for ex in examples
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
        "Examples:\n"
        "  '2025年' -> within_2_years\n"
        "  '2028年' -> between_2_and_5_years\n"
        "  '2030年' -> more_than_5_years\n\n"
        f"{few_shot}"
        f"Statement: {query_data}\n"
        "Answer:"
    )


# ==========================================
# CHAIN-OF-THOUGHT PROMPT (--cot flag)
# ==========================================
# 6 vi du tay co Reasoning, cover:
#   1. already  (khong co nam)
#   2. already  (chi co nam <=2024)
#   3. within_2_years      (deadline nam 2025)
#   4. between_2_and_5_years (deadline nam 2028)
#   5. more_than_5_years   (deadline nam 2050, giong pattern id=10364
#      ma ca 2 voters da dung KHI co nam ro rang)
#   6. already  (vague, khong nam -- pattern PHO BIEN NHAT, 74/155)
# Thay RAG-retrieved few-shot (data+label, khong reasoning) bang
# 6 vi du nay -- day model THEO DUNG FORMAT "Reasoning: ... Answer: ..."
COT_EXAMPLES = [
    {
        "data": "公司持續推動員工教育訓練計畫，提升整體職場安全意識，並定期舉辦相關講座。",
        "reasoning": "No specific year is mentioned. This describes an ongoing "
                      "activity with no future deadline.",
        "answer": "already",
    },
    {
        "data": "2023年公司已完成ISO 14001認證更新，並持續維持該管理系統運作至今。",
        "reasoning": "The only year mentioned is 2023, which is before 2024. "
                      "The certification is already completed and being "
                      "maintained. No future deadline.",
        "answer": "already",
    },
    {
        "data": "公司於2024年啟動供應商行為準則簽署計畫，預計2025年底完成全數供應商簽署。",
        "reasoning": "2024 is the baseline/start year (program launch). 2025 "
                      "is the target completion year. 2025-2024=1 year.",
        "answer": "within_2_years",
    },
    {
        "data": "為強化資訊安全管理，公司目標於2028年取得ISO 27001全公司認證，"
                "目前正進行各部門盤點作業。",
        "reasoning": "2028 is explicitly stated as the target year for "
                      "obtaining certification. 2028-2024=4 years.",
        "answer": "between_2_and_5_years",
    },
    {
        "data": "公司以全球據點推動2050年再生能源使用比例達100%之目標，"
                "並逐年提升現有廠區的綠電採購比例。",
        "reasoning": "2050 is explicitly stated as the target year for the "
                      "100% renewable energy goal. 2050-2024=26 years, "
                      "more than 5 years.",
        "answer": "more_than_5_years",
    },
    {
        "data": "公司秉持誠信經營理念，致力於提升企業治理透明度，"
                "並持續優化內部風險控管機制。",
        "reasoning": "No years are mentioned. This describes a general "
                      "ongoing corporate philosophy/practice with no "
                      "specific future deadline.",
        "answer": "already",
    },
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


LABEL_SYNONYMS = {"longer_than_5_years": "more_than_5_years"}


def llm_predict(model_name, prompt, valid_labels, retries=2, max_tokens_override=None):
    params  = ModelManager.get_params("extraction")
    timeout = ModelManager.get_timeout(model_name)
    max_tokens = max_tokens_override or params["max_tokens"]
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=params["temperature"],
                max_tokens=max_tokens,
                top_p=params["top_p"],
                timeout=timeout,
            )
            raw = resp.choices[0].message.content.strip()
            if "<think>" in raw:
                think_end = raw.find("</think>")
                if think_end == -1:
                    print(f"  [WARN] {model_name} truncated <think> (attempt {attempt+1})")
                    continue
                raw = raw[think_end + len("</think>"):].strip()

            # CoT: tim "Answer:" marker, chi lay phan SAU no de tranh
            # bat nham label xuat hien trong phan Reasoning (vd reasoning
            # noi ve nam 2028 nhung ket luan la "already" vi 2028 chi la
            # baseline -- neu khong tim Answer: truoc, co the bat nham
            # "between_2_and_5_years" tu trong cau reasoning)
            search_text = raw
            if "answer:" in raw.lower():
                idx = raw.lower().rfind("answer:")
                search_text = raw[idx + len("answer:"):].strip()

            for label in valid_labels:
                if label.lower() in search_text.lower():
                    return label
            for old_label, new_label in LABEL_SYNONYMS.items():
                if old_label.lower() in search_text.lower() and new_label in valid_labels:
                    return new_label

            # Fallback: thu tim trong toan bo raw (truong hop khong co "Answer:")
            for label in valid_labels:
                if label.lower() in raw.lower():
                    return label
            print(f"  [WARN] {model_name} invalid: '{raw[-80:]}' (attempt {attempt+1})")
        except Exception as e:
            print(f"  [WARN] {model_name} error: {e} (attempt {attempt+1})")
            if attempt == retries - 1:
                ModelManager.mark_failed(model_name)
    return "UNKNOWN"


def predict_s4(prompt, max_tokens_override=None):
    voters  = PRIORITY_MAP["COUNCIL_TIMELINE"]
    votes   = [llm_predict(v, prompt, VALID_LABELS["s4"], max_tokens_override=max_tokens_override)
               for v in voters]
    clean   = [v for v in votes if v in VALID_LABELS["s4"]]
    counter = Counter(clean)
    result  = None
    if counter:
        top    = counter.most_common(1)[0]
        result = top[0] if top[1] > 1 else None
    if result is None:
        chairman = PRIORITY_MAP["CHAIRMAN"][0]
        result   = llm_predict(chairman, prompt, VALID_LABELS["s4"], max_tokens_override=max_tokens_override)
    return result if result in VALID_LABELS["s4"] else "already"


# ==========================================
# RULE-BASED SAFETY NET (giu nguyen v4)
# ==========================================
COMMIT_VERBS_RE = re.compile(
    r"(目標|計畫|預計|將於|承諾|規劃|達成|完成|邁向|推動|布局|落實|展開|啟動|目標年)"
)
YEAR_RE = re.compile(r"20[2-5][0-9]")


def rule_based_s4(data, llm_result):
    if llm_result != "already":
        return llm_result
    for m in YEAR_RE.finditer(data):
        year = int(m.group())
        if year < 2025:
            continue
        window = data[max(0, m.start() - 15): m.end() + 15]
        if COMMIT_VERBS_RE.search(window):
            diff = year - 2024
            if diff <= 2:
                return "within_2_years"
            if diff <= 5:
                return "between_2_and_5_years"
            return "more_than_5_years"
    return "already"


def build_output_row(row_id, promise_status, evidence_status=None,
                      evidence_quality=None, verification_timeline=None):
    if promise_status == "No":
        return {"id": row_id, "promise_status": "No",
                "verification_timeline": "",
                "evidence_status": "", "evidence_quality": ""}
    return {
        "id":                    row_id,
        "promise_status":        promise_status,
        "verification_timeline": verification_timeline or "",
        "evidence_status":       evidence_status       or "",
        "evidence_quality":      evidence_quality      or "",
    }


# ==========================================
# MAIN PIPELINE
# ==========================================
def run_pipeline(test_file, output_file, batch_size, llm_workers, rebuild_rag=False, use_cot=False):
    print("\n" + "=" * 60)
    print("  ESG Pipeline v7 - BERT(S1/S2/S3) BATCHED + LLM Council(S4) CONCURRENT")
    print("=" * 60)

    test_df = pd.read_csv(test_file)
    total = len(test_df)
    print(f"\nTest set: {total} rows  (batch_size={batch_size}, llm_workers={llm_workers})")

    ids   = test_df["id"].astype(str).tolist()
    texts = test_df["data"].astype(str).str.strip().tolist()

    # -------- Stage 1: S1 (promise_status) on ALL rows --------
    print("\n[Stage 1/4] S1 promise_status (batched)...")
    bert_s1 = load_bert_model("s1", batch_size)
    promise_status = batch_predict(bert_s1, texts, VALID_LABELS["s1"], default="No")
    del bert_s1
    torch.cuda.empty_cache()

    yes_idx = [i for i, p in enumerate(promise_status) if p == "Yes"]
    print(f"  promise_status=Yes: {len(yes_idx)}/{total}")

    evidence_status_arr  = [""] * total
    evidence_quality_arr = [""] * total
    verification_arr     = [""] * total

    if yes_idx:
        yes_texts = [texts[i] for i in yes_idx]

        # -------- Stage 2: S2 (evidence_status, batched) --------
        print("\n[Stage 2/4] S2 evidence_status (batched, new grid-c model)...")
        bert_s2 = load_bert_model("s2", batch_size)
        ev_status = batch_predict(bert_s2, yes_texts, VALID_LABELS["s2"], default="No")
        del bert_s2
        torch.cuda.empty_cache()
        for local_i, global_i in enumerate(yes_idx):
            evidence_status_arr[global_i] = ev_status[local_i]

        ev_yes_local_idx = [li for li, e in enumerate(ev_status) if e == "Yes"]
        print(f"  evidence_status=Yes: {len(ev_yes_local_idx)}/{len(yes_idx)}")

        # -------- Stage 3: S3 (evidence_quality, batched) --------
        if ev_yes_local_idx:
            print("\n[Stage 3/4] S3 evidence_quality (batched)...")
            bert_s3 = load_bert_model("s3", batch_size)
            ev_yes_texts = [yes_texts[li] for li in ev_yes_local_idx]
            eq = batch_predict(bert_s3, ev_yes_texts, VALID_LABELS["s3"], default="Clear")
            del bert_s3
            torch.cuda.empty_cache()
            for k, li in enumerate(ev_yes_local_idx):
                evidence_quality_arr[yes_idx[li]] = eq[k]
        else:
            print("\n[Stage 3/4] S3 evidence_quality... (skip, 0 rows)")

        # -------- Stage 4: S4 (LLM Council + RAG + rule, CONCURRENT) --------
        if use_cot:
            print("\n[Stage 4/4] S4 verification_timeline (CoT, 6 fixed examples, concurrent)...")
            col_s4 = None
        else:
            print("\n[Stage 4/4] S4 verification_timeline (LLM Council + RAG, concurrent)...")
            print("Initializing ChromaDB for RAG...")
            rag_records = load_rag_data(JSON_TRAIN, VAL_JSON)
            col_s4 = build_chroma_s4(rag_records, force_rebuild=rebuild_rag)

        # Resume support: file phu rieng cho S4 (id -> verification_timeline)
        s4_cache_file = output_file + ".s4cache.csv"
        s4_done = {}
        if os.path.exists(s4_cache_file):
            df_cache = pd.read_csv(s4_cache_file, dtype=str)
            s4_done = dict(zip(df_cache["id"].astype(str), df_cache["verification_timeline"]))
            print(f"  Resuming S4: {len(s4_done)}/{len(yes_idx)} rows already done.")
        else:
            pd.DataFrame(columns=["id", "verification_timeline"]).to_csv(
                s4_cache_file, index=False, encoding="utf-8-sig"
            )

        write_lock = threading.Lock()

        def process_s4_row(local_i, global_i):
            row_id = ids[global_i]
            if row_id in s4_done:
                return row_id, s4_done[row_id]
            data = yes_texts[local_i]
            try:
                if use_cot:
                    prompt = build_prompt_s4_cot(data)
                    result = predict_s4(prompt, max_tokens_override=COT_MAX_TOKENS)
                else:
                    examples = retrieve_s4(col_s4, data)
                    prompt   = build_prompt_s4(data, examples)
                    result   = predict_s4(prompt)
                result = rule_based_s4(data, result)
            except Exception as e:
                print(f"  [ERROR] id={row_id}: {e}")
                result = "already"
            with write_lock:
                pd.DataFrame([{"id": row_id, "verification_timeline": result}]).to_csv(
                    s4_cache_file, mode="a", header=False, index=False, encoding="utf-8-sig"
                )
            return row_id, result

        pending = [(li, gi) for li, gi in enumerate(yes_idx) if ids[gi] not in s4_done]
        print(f"  Processing {len(pending)} rows with {llm_workers} concurrent workers...")

        s4_results = dict(s4_done)
        completed = 0
        with ThreadPoolExecutor(max_workers=llm_workers) as executor:
            futures = [executor.submit(process_s4_row, li, gi) for li, gi in pending]
            for fut in as_completed(futures):
                row_id, result = fut.result()
                s4_results[row_id] = result
                completed += 1
                if completed % 50 == 0:
                    print(f"    [{completed}/{len(pending)}] done...")

        for global_i in yes_idx:
            verification_arr[global_i] = s4_results.get(ids[global_i], "already")
    else:
        print("\n[Stage 2-4/4] skip, 0 rows with promise=Yes")

    # -------- Assemble output --------
    print("\nAssembling output...")
    out_rows = [
        build_output_row(
            ids[i],
            promise_status        = promise_status[i],
            evidence_status       = evidence_status_arr[i],
            evidence_quality      = evidence_quality_arr[i],
            verification_timeline = verification_arr[i],
        )
        for i in range(total)
    ]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_out = pd.DataFrame(out_rows, columns=OUTPUT_COLUMNS)
    df_out.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\n{'='*60}")
    print(f"  Done! {len(df_out)}/{total} rows -> {output_file}")
    print(f"{'='*60}")
    print(df_out.head(10).to_string(index=False))
    print()
    print("verification_timeline distribution:")
    print(df_out.loc[df_out["promise_status"]=="Yes", "verification_timeline"]
          .value_counts(normalize=True))
    print()
    print("evidence_status distribution:")
    print(df_out.loc[df_out["promise_status"]=="Yes", "evidence_status"]
          .value_counts(normalize=True))
    print()
    print("evidence_quality distribution:")
    print(df_out.loc[df_out["evidence_quality"]!="", "evidence_quality"]
          .value_counts(normalize=True))
    return df_out


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESG Pipeline v7")
    parser.add_argument("--test", default=TEST_CSV)
    parser.add_argument(
        "--output",
        default=os.path.join(
            OUTPUT_DIR,
            f"predictions_v7_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--llm-workers", type=int, default=LLM_WORKERS_DEFAULT)
    parser.add_argument("--rebuild-rag", action="store_true")
    parser.add_argument("--cot", action="store_true",
                         help="Dung Chain-of-Thought prompt (6 vi du tay co Reasoning) "
                              "thay cho RAG-retrieved few-shot")
    args = parser.parse_args()
    run_pipeline(args.test, args.output, batch_size=args.batch_size,
                  llm_workers=args.llm_workers, rebuild_rag=args.rebuild_rag,
                  use_cot=args.cot)