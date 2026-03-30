"""
ESG Promise Verification — Main Pipeline
=========================================
S1 (promise_status):       BERT MacBERT-large
S2 (evidence_status):      BERT RoBERTa-wwm-large
S3 (evidence_quality):     RAG + Council LLM
S4 (verification_timeline): RAG + Council LLM
Thay doi prompt nhung ket qua te hon v1
Usage:
    python main_pipeline.py --test  TEST_CSV  --train TRAIN_JSON  --output OUTPUT_CSV
"""

import os
import sys
import argparse
import json
import random
import pandas as pd
from datetime import datetime
from collections import Counter

import warnings
import torch
from transformers import pipeline as hf_pipeline
import chromadb
from chromadb.utils import embedding_functions

# --- Load config gốc (chỉ lấy client và ModelManager) ---
sys.path.insert(0, r"/llm-council/backend")
try:
    import config
    from config import client, ModelManager
    print("✅ Config & ModelManager loaded.")
except ImportError as e:
    print(f"❌ Error loading config: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ==========================================
# LOCAL CONFIG — BERT, Labels, Output format
# ==========================================
BERT_MODELS = {
    "s1": r"D:\LLMs\BERT-ESG\macbert-large-s1",
    "s2": r"D:\LLMs\BERT-ESG\roberta-wwm-large-s2",
}

PRIORITY_MAP = {
    "COUNCIL_CLARITY":  ["qwen2.5-14b-esg", "qwen2.5-72b-esg", "deepseek-r1-70b-esg"],
    "COUNCIL_TIMELINE": ["qwen2.5-14b-esg", "qwen2.5-72b-esg", "deepseek-r1-70b-esg"],
    "CHAIRMAN":         ["qwen2.5-32b-esg"],
}

VALID_LABELS = {
    "s1": ["Yes", "No"],
    "s2": ["Yes", "No"],
    "s3": ["Clear", "Not Clear", "Misleading", "N/A"],
    "s4": ["already", "within_2_years", "between_2_and_5_years", "longer_than_5_years"],
}

OUTPUT_COLUMNS = ["id", "promise_status", "verification_timeline", "evidence_status", "evidence_quality"]

def build_output_row(row_id, promise_status, evidence_status=None,
                     evidence_quality=None, verification_timeline=None):
    if promise_status == "No":
        return {
            "id":                    row_id,
            "promise_status":        "No",
            "verification_timeline": "N/A",
            "evidence_status":       "N/A",
            "evidence_quality":      "N/A",
        }
    return {
        "id":                    row_id,
        "promise_status":        promise_status,
        "verification_timeline": verification_timeline or "N/A",
        "evidence_status":       evidence_status or "N/A",
        "evidence_quality":      evidence_quality or "N/A",
    }

# ==========================================
# CONFIG
# ==========================================
CHROMA_DIR  = r"/AICup26/chroma_db"
EMBED_MODEL = "BAAI/bge-large-zh-v1.5"
TOP_K       = 6
DEVICE      = 0 if torch.cuda.is_available() else -1

# ==========================================
# BERT — S1 và S2
# ==========================================
def load_bert_models():
    print("Loading BERT models...")
    s1 = hf_pipeline(
        "text-classification",
        model=BERT_MODELS["s1"],
        tokenizer=BERT_MODELS["s1"],
        device=DEVICE,
        truncation=True,
        max_length=512,
    )
    s2 = hf_pipeline(
        "text-classification",
        model=BERT_MODELS["s2"],
        tokenizer=BERT_MODELS["s2"],
        device=DEVICE,
        truncation=True,
        max_length=512,
    )
    print("  BERT S1 loaded:", BERT_MODELS["s1"])
    print("  BERT S2 loaded:", BERT_MODELS["s2"])
    return s1, s2

def predict_s1(classifier, text):
    out = classifier(text)[0]
    return out["label"]

def predict_s2(classifier, text):
    out = classifier(text)[0]
    return out["label"]

# ==========================================
# CHROMADB — Build index từ train data
# ==========================================
def load_train_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    random.seed(42)
    random.shuffle(records)
    return records[:800]

def build_chroma_collections(train_records, chroma_dir):
    os.makedirs(chroma_dir, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    # Kiểm tra nếu collections đã tồn tại → dùng lại, không build lại
    existing = {c.name for c in chroma_client.list_collections()}
    if "esg_s3" in existing and "esg_s4" in existing:
        col_s3 = chroma_client.get_collection("esg_s3", embedding_function=ef)
        col_s4 = chroma_client.get_collection("esg_s4", embedding_function=ef)
        print(f"ChromaDB loaded from cache: S3={col_s3.count()} docs, S4={col_s4.count()} docs")
        return col_s3, col_s4

    # Build mới
    print(f"Building ChromaDB index at: {chroma_dir}")
    for col_name in ["esg_s3", "esg_s4"]:
        try:
            chroma_client.delete_collection(col_name)
        except Exception:
            pass

    col_s3 = chroma_client.create_collection("esg_s3", embedding_function=ef,
                                              metadata={"hnsw:space": "cosine"})
    col_s4 = chroma_client.create_collection("esg_s4", embedding_function=ef,
                                              metadata={"hnsw:space": "cosine"})

    docs_s3, ids_s3, meta_s3 = [], [], []
    docs_s4, ids_s4, meta_s4 = [], [], []

    for i, r in enumerate(train_records):
        data    = str(r["data"]).strip()
        promise = str(r.get("promise_status") or "").strip()
        quality = str(r.get("evidence_quality") or "N/A").strip()
        tl      = str(r.get("verification_timeline") or "").strip()

        if promise != "Yes":
            continue

        if quality in VALID_LABELS["s3"]:
            docs_s3.append(data)
            ids_s3.append(f"s3_{i}")
            meta_s3.append({"evidence_quality": quality})

        if tl in VALID_LABELS["s4"]:
            docs_s4.append(data)
            ids_s4.append(f"s4_{i}")
            meta_s4.append({"verification_timeline": tl})

    BATCH = 100
    for i in range(0, len(docs_s3), BATCH):
        col_s3.add(documents=docs_s3[i:i+BATCH],
                   ids=ids_s3[i:i+BATCH],
                   metadatas=meta_s3[i:i+BATCH])

    for i in range(0, len(docs_s4), BATCH):
        col_s4.add(documents=docs_s4[i:i+BATCH],
                   ids=ids_s4[i:i+BATCH],
                   metadatas=meta_s4[i:i+BATCH])

    print(f"ChromaDB built: S3={col_s3.count()} docs, S4={col_s4.count()} docs")
    return col_s3, col_s4

# ==========================================
# RAG — Retrieve balanced examples
# ==========================================
def retrieve_balanced(collection, query_text, task, k=TOP_K):
    label_key = "evidence_quality" if task == "s3" else "verification_timeline"

    # Ưu tiên Not Clear cho S3
    if task == "s3":
        label_quota = {"Not Clear": 3, "Clear": 1, "N/A": 1, "Misleading": 1}
    else:
        labels     = VALID_LABELS["s4"]
        per_label  = max(1, k // len(labels))
        label_quota = {l: per_label for l in labels}

    examples = []
    for label, quota in label_quota.items():
        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=quota,
                where={label_key: {"$eq": label}}
            )
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                examples.append({"data": doc, label_key: meta[label_key]})
        except Exception:
            continue

    return examples[:k]

# ==========================================
# RAG — Build prompts
# ==========================================
def build_prompt_s3(query_data, examples):
    few_shot = "".join(
        f"Statement: {ex['data']}\nAnswer: {ex['evidence_quality']}\n\n"
        for ex in examples
    )
    return (
        "You are an ESG analyst. Classify the evidence quality of the ESG statement below.\n\n"
        "RULES — apply in order:\n"
        "1. If the statement contains NO evidence at all → N/A\n"
        "2. If the statement contains vague, uncommitted, or non-measurable language → Not Clear\n"
        "   Examples of Not Clear language: '致力於', '積極推動', '逐步實施', '持續改善', '努力',\n"
        "   '推動', '促進', '強化', '提升', '研擬', '規劃中', '期望', '目標為', general commitments\n"
        "   without specific numbers, deadlines, or measurable targets.\n"
        "3. If the statement uses misleading or greenwashing language → Misleading\n"
        "4. Only if the statement provides SPECIFIC, MEASURABLE, VERIFIABLE evidence\n"
        "   (e.g. exact percentages, concrete targets with deadlines, quantified data) → Clear\n\n"
        "IMPORTANT: When in doubt between Clear and Not Clear, choose Not Clear.\n\n"
        f"{few_shot}"
        f"Statement: {query_data}\n"
        "Answer:"
    )

def build_prompt_s4(query_data, examples):
    few_shot = "".join(
        f"Statement: {ex['data']}\nAnswer: {ex['verification_timeline']}\n\n"
        for ex in examples
    )
    return (
        "You are an ESG analyst. The ESG report was published in 2024.\n"
        "Classify the expected completion timeframe of the commitment.\n\n"
        "RULES:\n"
        "- already: action is ALREADY completed or continuously ongoing with NO future target year.\n"
        "  (e.g. '已導入', '目前', '持續執行中' with no future deadline)\n"
        "- within_2_years: explicit target year is 2025 or 2026.\n"
        "- between_2_and_5_years: explicit target year is 2027, 2028, or 2029.\n"
        "- longer_than_5_years: explicit target year is 2030 or beyond.\n\n"
        "IMPORTANT: If the statement mentions ANY specific future year (2025~2050),\n"
        "classify by that year — do NOT choose 'already'.\n\n"
        f"{few_shot}"
        f"Statement: {query_data}\n"
        "Answer:"
    )

# ==========================================
# LLM — Ollama inference
# ==========================================
def llm_predict(model_name, prompt, valid_labels, retries=2):
    params  = ModelManager.get_params("extraction")
    timeout = ModelManager.get_timeout(model_name)

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=params["temperature"],
                max_tokens=params["max_tokens"],
                top_p=params["top_p"],
                timeout=timeout,
            )
            raw = resp.choices[0].message.content.strip()
            for label in valid_labels:
                if label.lower() in raw.lower():
                    return label
            # Response không chứa valid label → retry
            print(f"  [WARN] {model_name} invalid response: '{raw[:50]}' (attempt {attempt+1})")

        except Exception as e:
            print(f"  [WARN] {model_name} error: {e} (attempt {attempt+1})")
            if attempt == retries - 1:
                ModelManager.mark_failed(model_name)

    return "UNKNOWN"

# ==========================================
# Council Vote
# ==========================================
def council_vote(votes, valid_labels):
    counter = Counter(v for v in votes if v in valid_labels)
    if not counter:
        return valid_labels[0]
    top = counter.most_common(1)[0]
    if top[1] == 1:
        return None   # tie
    return top[0]

def predict_with_council(prompt, council_key, valid_labels):
    voters = PRIORITY_MAP[council_key]
    votes  = [llm_predict(v, prompt, valid_labels) for v in voters]
    result = council_vote(votes, valid_labels)
    if result is None:
        # Tie → Chairman
        chairman = PRIORITY_MAP["CHAIRMAN"][0]
        result   = llm_predict(chairman, prompt, valid_labels)
        if result not in valid_labels:
            result = valid_labels[0]
    return result

# ==========================================
# MAIN PIPELINE
# ==========================================
def run_pipeline(test_file, train_json, output_file):
    # --- Load models ---
    bert_s1, bert_s2 = load_bert_models()

    # --- Build ChromaDB ---
    train_records  = load_train_data(train_json)
    col_s3, col_s4 = build_chroma_collections(train_records, CHROMA_DIR)

    # --- Load test data ---
    test_df = pd.read_csv(test_file)
    total   = len(test_df)
    print(f"\nTest set: {total} rows")

    # --- Resume support: skip rows đã xử lý ---
    done_ids = set()
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file, sep="\t", dtype=str)
        done_ids    = set(df_existing["id"].astype(str).tolist())
        print(f"Resuming: {len(done_ids)} rows already done, skipping...")
    else:
        # Ghi header lần đầu
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(
            output_file, index=False, encoding="utf-8-sig", sep="\t"
        )

    print("Starting inference...\n")

    for idx, (_, row) in enumerate(test_df.iterrows()):
        row_id = str(row["id"])

        # Skip nếu đã xử lý
        if row_id in done_ids:
            continue

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{total}] Processing...")

        data = str(row["data"]).strip()

        try:
            # ---- S1: BERT ----
            promise_status = predict_s1(bert_s1, data)

            if promise_status == "No":
                out_row = build_output_row(row_id, "No")
            else:
                # ---- S2: BERT ----
                evidence_status = predict_s2(bert_s2, data)

                # ---- S3: RAG + Council (chỉ khi evidence=Yes) ----
                if evidence_status == "Yes":
                    examples_s3      = retrieve_balanced(col_s3, data, "s3")
                    prompt_s3        = build_prompt_s3(data, examples_s3)
                    evidence_quality = predict_with_council(
                        prompt_s3, "COUNCIL_CLARITY", VALID_LABELS["s3"]
                    )
                else:
                    evidence_quality = "N/A"   # S2=No → skip S3

                # ---- S4: RAG + Council (luôn chạy khi promise=Yes) ----
                examples_s4           = retrieve_balanced(col_s4, data, "s4")
                prompt_s4             = build_prompt_s4(data, examples_s4)
                verification_timeline = predict_with_council(
                    prompt_s4, "COUNCIL_TIMELINE", VALID_LABELS["s4"]
                )

                out_row = build_output_row(
                    row_id,
                    promise_status        = promise_status,
                    evidence_status       = evidence_status,
                    evidence_quality      = evidence_quality,
                    verification_timeline = verification_timeline,
                )

        except Exception as e:
            print(f"  [ERROR] row {row_id}: {e}")
            continue

        # --- Lưu ngay từng dòng ---
        pd.DataFrame([out_row], columns=OUTPUT_COLUMNS).to_csv(
            output_file, mode="a", header=False,
            index=False, encoding="utf-8-sig", sep="\t"
        )

    # --- Load và print kết quả cuối ---
    df_out = pd.read_csv(output_file, sep="\t", dtype=str)
    print(f"\nDone! {len(df_out)}/{total} rows saved to: {output_file}")
    print(df_out.head(10).to_string(index=False))
    return df_out


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESG Promise Verification Pipeline")
    parser.add_argument("--test",  required=False,
                        default=r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_1000 V1_test200.csv",
                        help="Path to test CSV file")
    parser.add_argument("--train", required=False,
                        default=r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_1000 V1.json",
                        help="Path to train JSON file")
    parser.add_argument("--output", required=False,
                        default=os.path.join(
                            r"/AICup26/results",
                            f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv"
                        ),
                        help="Path to output TSV file")
    args = parser.parse_args()

    run_pipeline(
        test_file   = args.test,
        train_json  = args.train,
        output_file = args.output,
    )

