"""
ESG Promise Verification — Pipeline v4
========================================
S1 (promise_status):        BERT MacBERT-large
S2 (evidence_status):       BERT RoBERTa-wwm-large
S3 (evidence_quality):      BERT MacBERT-large (Binary: Clear / Not Clear)
S4 (verification_timeline): RAG + LLM Council (3 voters + chairman)

Notes:
- S3 N/A   → pipeline tự set khi S2=No (không predict)
- S3 Misleading → chấp nhận bỏ qua (1 case trong toàn dataset)
- S4 LLM Council giữ nguyên vì đạt 0.73 Macro-F1

Usage:
    python main_pipeline_v3.py
    python main_pipeline_v3.py --output results/predictions_v3.tsv
"""

import os
import sys
import argparse
import json
import random
import warnings
import pandas as pd
from datetime import datetime
from collections import Counter

import torch
from transformers import pipeline as hf_pipeline
import chromadb
from chromadb.utils import embedding_functions

# --- Load config ---
sys.path.insert(0, r"/llm-council/backend")
try:
    from config import client, ModelManager
    print("✅ Config & ModelManager loaded.")
except ImportError as e:
    print(f"❌ Error loading config: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ==========================================
# LOCAL CONFIG
# ==========================================
BERT_MODELS = {
    "s1": r"D:\LLMs\BERT-ESG\macbert-large-s1",         # MacBERT  — 98.0%
    "s2": r"D:\LLMs\BERT-ESG\roberta-wwm-large-s2",     # RoBERTa  — 97.0%
    "s3": r"D:\LLMs\BERT-ESG\macbert-large-s3",         # MacBERT  — Binary Clear/Not Clear
}

PRIORITY_MAP = {
    "COUNCIL_TIMELINE": ["qwen2.5-14b-esg", "qwen2.5-72b-esg", "deepseek-r1-70b-esg"],
    "CHAIRMAN":         ["qwen2.5-32b-esg"],
}

VALID_LABELS = {
    "s1": ["Yes", "No"],
    "s2": ["Yes", "No"],
    "s3": ["Clear", "Not Clear"],   # Binary — N/A handled by pipeline
    "s4": ["already", "within_2_years", "between_2_and_5_years", "longer_than_5_years"],
}

OUTPUT_COLUMNS = ["id", "promise_status", "verification_timeline",
                  "evidence_status", "evidence_quality"]

JSON_TRAIN  = r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_1000 V1.json"
TEST_CSV    = r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_1000 V1_test200.csv"
OUTPUT_DIR  = r"/AICup26/results"
CHROMA_DIR  = r"/AICup26/chroma_db"
EMBED_MODEL = "BAAI/bge-large-zh-v1.5"
TOP_K       = 6
DEVICE      = 0 if torch.cuda.is_available() else -1

# ==========================================
# OUTPUT BUILDER
# ==========================================
def build_output_row(row_id, promise_status, evidence_status=None,
                     evidence_quality=None, verification_timeline=None):
    if promise_status == "No":
        return {"id": row_id, "promise_status": "No",
                "verification_timeline": "N/A",
                "evidence_status": "N/A", "evidence_quality": "N/A"}
    return {
        "id":                    row_id,
        "promise_status":        promise_status,
        "verification_timeline": verification_timeline or "N/A",
        "evidence_status":       evidence_status or "N/A",
        "evidence_quality":      evidence_quality or "N/A",
    }

# ==========================================
# BERT — Load S1, S2, S3
# ==========================================
def load_bert_models():
    print("\nLoading BERT models...")
    models = {}
    for task, path in BERT_MODELS.items():
        models[task] = hf_pipeline(
            "text-classification",
            model=path,
            tokenizer=path,
            device=DEVICE,
            truncation=True,
            max_length=512,
        )
        print(f"  ✅ BERT {task.upper()} loaded: {path}")
    return models

def bert_predict(classifier, text):
    return classifier(text)[0]["label"]

# ==========================================
# CHROMADB — S4 only
# ==========================================
def load_train_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    random.seed(42)
    random.shuffle(records)
    return records[:800]

def build_chroma_s4(train_records, chroma_dir):
    os.makedirs(chroma_dir, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    # Check cache
    existing = {c.name for c in chroma_client.list_collections()}
    if "esg_s4" in existing:
        col_s4 = chroma_client.get_collection("esg_s4", embedding_function=ef)
        print(f"  ChromaDB S4 loaded from cache: {col_s4.count()} docs")
        return col_s4

    # Build
    print(f"  Building ChromaDB S4 index...")
    try:
        chroma_client.delete_collection("esg_s4")
    except Exception:
        pass

    col_s4 = chroma_client.create_collection(
        "esg_s4", embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    docs, ids, metas = [], [], []
    for i, r in enumerate(train_records):
        promise  = str(r.get("promise_status", "")).strip()
        timeline = str(r.get("verification_timeline", "")).strip()
        if promise != "Yes" or timeline not in VALID_LABELS["s4"]:
            continue
        docs.append(str(r["data"]).strip())
        ids.append(f"s4_{i}")
        metas.append({"verification_timeline": timeline})

    BATCH = 100
    for i in range(0, len(docs), BATCH):
        col_s4.add(documents=docs[i:i+BATCH],
                   ids=ids[i:i+BATCH],
                   metadatas=metas[i:i+BATCH])

    print(f"  ChromaDB S4 built: {col_s4.count()} docs")
    return col_s4

# ==========================================
# RAG — S4 retrieve balanced
# ==========================================
def retrieve_balanced_s4(col_s4, query_text, k=TOP_K):
    labels    = VALID_LABELS["s4"]
    per_label = max(1, k // len(labels))
    examples  = []

    for label in labels:
        try:
            results = col_s4.query(
                query_texts=[query_text],
                n_results=per_label,
                where={"verification_timeline": {"$eq": label}}
            )
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                examples.append({"data": doc, "verification_timeline": meta["verification_timeline"]})
        except Exception:
            continue

    return examples[:k]

# ==========================================
# LLM — S4 prompt + inference
# ==========================================
def build_prompt_s4(query_data, examples):
    few_shot = "".join(
        f"Statement: {ex['data']}\nAnswer: {ex['verification_timeline']}\n\n"
        for ex in examples
    )
    return (
        "You are an ESG analyst. The ESG report was published in 2024.\n"
        "Classify the expected completion timeframe of the commitment.\n"
        "Choose exactly one:\n"
        "- already (action already implemented or ongoing in 2024, no future deadline)\n"
        "- within_2_years (target year 2025 or 2026)\n"
        "- between_2_and_5_years (target year 2027-2029)\n"
        "- longer_than_5_years (target year 2030 or beyond)\n\n"
        f"{few_shot}"
        f"Statement: {query_data}\n"
        "Answer:"
    )

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
            print(f"  [WARN] {model_name} invalid response: '{raw[:60]}' (attempt {attempt+1})")
        except Exception as e:
            print(f"  [WARN] {model_name} error: {e} (attempt {attempt+1})")
            if attempt == retries - 1:
                ModelManager.mark_failed(model_name)

    return "UNKNOWN"

def council_vote(votes, valid_labels):
    clean   = [v for v in votes if v in valid_labels]
    if not clean:
        return None
    counter = Counter(clean)
    top     = counter.most_common(1)[0]
    return top[0] if top[1] > 1 else None   # None = tie

def predict_s4_council(prompt, valid_labels):
    voters = PRIORITY_MAP["COUNCIL_TIMELINE"]
    votes  = [llm_predict(v, prompt, valid_labels) for v in voters]
    result = council_vote(votes, valid_labels)

    if result is None:
        # Tie hoặc tất cả fail → Chairman
        chairman = PRIORITY_MAP["CHAIRMAN"][0]
        result   = llm_predict(chairman, prompt, valid_labels)

    if result not in valid_labels:
        result = "already"   # fallback

    return result

# ==========================================
# MAIN PIPELINE
# ==========================================
def run_pipeline(test_file, train_json, output_file):
    print("\n" + "="*60)
    print("  ESG Pipeline v4")
    print("  S1/S2/S3 → BERT  |  S4 → RAG + LLM Council")
    print("="*60)

    # --- Load BERT models ---
    bert = load_bert_models()

    # --- Build ChromaDB S4 ---
    print("\nInitializing ChromaDB (S4 only)...")
    train_records = load_train_data(train_json)
    col_s4        = build_chroma_s4(train_records, CHROMA_DIR)

    # --- Load test data ---
    test_df = pd.read_csv(test_file)
    total   = len(test_df)
    print(f"\nTest set: {total} rows")

    # --- Resume support ---
    done_ids = set()
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file, sep="\t", dtype=str)
        done_ids    = set(df_existing["id"].astype(str).tolist())
        print(f"Resuming: {len(done_ids)} rows already done.")
    else:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(
            output_file, index=False, encoding="utf-8-sig", sep="\t"
        )

    print("\nStarting inference...\n")

    for idx, (_, row) in enumerate(test_df.iterrows()):
        row_id = str(row["id"])
        if row_id in done_ids:
            continue

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{total}] Processing...")

        data = str(row["data"]).strip()

        try:
            # ---- S1: BERT ----
            promise_status = bert_predict(bert["s1"], data)

            if promise_status == "No":
                out_row = build_output_row(row_id, "No")

            else:
                # ---- S2: BERT ----
                evidence_status = bert_predict(bert["s2"], data)

                # ---- S3: BERT Binary ----
                if evidence_status == "Yes":
                    evidence_quality = bert_predict(bert["s3"], data)
                    # BERT S3 chỉ output Clear/Not Clear
                    if evidence_quality not in VALID_LABELS["s3"]:
                        evidence_quality = "Clear"
                else:
                    evidence_quality = "N/A"   # S2=No → skip S3

                # ---- S4: RAG + LLM Council ----
                examples_s4           = retrieve_balanced_s4(col_s4, data)
                prompt_s4             = build_prompt_s4(data, examples_s4)
                verification_timeline = predict_s4_council(prompt_s4, VALID_LABELS["s4"])

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

    # --- Summary ---
    df_out = pd.read_csv(output_file, sep="\t", dtype=str)
    print(f"\n{'='*60}")
    print(f"  Done! {len(df_out)}/{total} rows saved → {output_file}")
    print(f"{'='*60}")
    print(df_out.head(10).to_string(index=False))
    return df_out


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESG Pipeline v4")
    parser.add_argument("--test",  default=TEST_CSV,   help="Path to test CSV")
    parser.add_argument("--train", default=JSON_TRAIN, help="Path to train JSON")
    parser.add_argument(
        "--output",
        default=os.path.join(
            OUTPUT_DIR,
            f"predictions_v3.tsv"
        ),
        help="Path to output TSV"
    )
    args = parser.parse_args()

    run_pipeline(
        test_file   = args.test,
        train_json  = args.train,
        output_file = args.output,
    )