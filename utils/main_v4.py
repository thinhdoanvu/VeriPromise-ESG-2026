"""
ESG Promise Verification — Pipeline v4 (Final)
================================================
S1 (promise_status):        BERT MacBERT-large       98.0%
S2 (evidence_status):       BERT RoBERTa-wwm-large   97.0%
S3 (evidence_quality):      BERT MacBERT-large       Binary Clear/Not Clear
S4 (verification_timeline): RAG + LLM Council        3 voters + 1 chairman

Pipeline logic:
  promise=No  → tất cả N/A, dừng
  promise=Yes
    evidence=No  → quality=N/A, vẫn predict S4
    evidence=Yes → BERT S3 → quality=Clear/Not Clear, predict S4

Usage:
    python main_pipeline_v4.py
    python main_pipeline_v4.py --output results/predictions_v4.tsv
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

# ==========================================
# LOAD CONFIG
# ==========================================
sys.path.insert(0, r"C:\Users\VU\Documents\NLP\AICup26\llm-council\backend")
try:
    from config import (
        client, ModelManager,
        BERT_MODELS, PRIORITY_MAP, VALID_LABELS,
        OUTPUT_COLUMNS, build_output_row
    )
    print("✅ Config loaded.")
except ImportError as e:
    print(f"❌ Error loading config: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ==========================================
# PATHS
# ==========================================
JSON_TRAIN  = r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_1000 V1.json"
TEST_CSV    = r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_1000 V1_test200.csv"
OUTPUT_DIR  = r"C:\Users\VU\Documents\NLP\AICup26\results"
CHROMA_DIR  = r"C:\Users\VU\Documents\NLP\AICup26\chroma_db"
EMBED_MODEL = "BAAI/bge-large-zh-v1.5"
TOP_K       = 6
DEVICE      = 0 if torch.cuda.is_available() else -1

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
        print(f"  ✅ S{task[1]} — {os.path.basename(path)}")
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

def build_chroma_s4(train_records):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    existing = {c.name for c in chroma_client.list_collections()}
    if "esg_s4" in existing:
        col = chroma_client.get_collection("esg_s4", embedding_function=ef)
        print(f"  ChromaDB S4: loaded from cache ({col.count()} docs)")
        return col

    print(f"  ChromaDB S4: building index...")
    try:
        chroma_client.delete_collection("esg_s4")
    except Exception:
        pass

    col = chroma_client.create_collection(
        "esg_s4", embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
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

    for i in range(0, len(docs), 100):
        col.add(documents=docs[i:i+100],
                ids=ids[i:i+100],
                metadatas=metas[i:i+100])

    print(f"  ChromaDB S4: built ({col.count()} docs)")
    return col

# ==========================================
# RAG — S4 balanced retrieve
# ==========================================
def retrieve_s4(col, query_text):
    labels    = VALID_LABELS["s4"]
    per_label = max(1, TOP_K // len(labels))
    examples  = []
    for label in labels:
        try:
            res = col.query(
                query_texts=[query_text],
                n_results=per_label,
                where={"verification_timeline": {"$eq": label}}
            )
            for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
                examples.append({
                    "data": doc,
                    "verification_timeline": meta["verification_timeline"]
                })
        except Exception:
            continue
    return examples[:TOP_K]

# ==========================================
# LLM — S4 prompt + council vote
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
            print(f"  [WARN] {model_name} invalid: '{raw[:60]}' (attempt {attempt+1})")
        except Exception as e:
            print(f"  [WARN] {model_name} error: {e} (attempt {attempt+1})")
            if attempt == retries - 1:
                ModelManager.mark_failed(model_name)
    return "UNKNOWN"

def predict_s4(prompt):
    voters = PRIORITY_MAP["COUNCIL_TIMELINE"]
    votes  = [llm_predict(v, prompt, VALID_LABELS["s4"]) for v in voters]

    clean   = [v for v in votes if v in VALID_LABELS["s4"]]
    counter = Counter(clean)
    result  = None
    if counter:
        top    = counter.most_common(1)[0]
        result = top[0] if top[1] > 1 else None   # majority only

    if result is None:
        # Tie hoặc tất cả UNKNOWN → Chairman
        chairman = PRIORITY_MAP["CHAIRMAN"][0]
        result   = llm_predict(chairman, prompt, VALID_LABELS["s4"])

    return result if result in VALID_LABELS["s4"] else "already"

# ==========================================
# MAIN PIPELINE
# ==========================================
def run_pipeline(test_file, train_json, output_file):
    print("\n" + "="*60)
    print("  ESG Pipeline v4 — Final")
    print("  S1/S2/S3: BERT  |  S4: RAG + LLM Council")
    print("="*60)

    bert          = load_bert_models()
    print("\nInitializing ChromaDB...")
    train_records = load_train_data(train_json)
    col_s4        = build_chroma_s4(train_records)

    test_df = pd.read_csv(test_file)
    total   = len(test_df)
    print(f"\nTest set: {total} rows")

    # Resume
    done_ids = set()
    if os.path.exists(output_file):
        df_done  = pd.read_csv(output_file, sep="\t", dtype=str)
        done_ids = set(df_done["id"].astype(str).tolist())
        print(f"Resuming: {len(done_ids)}/{total} rows already done.")
    else:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
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

                # ---- S3: BERT (evidence=Yes only) ----
                if evidence_status == "Yes":
                    eq = bert_predict(bert["s3"], data)
                    evidence_quality = eq if eq in VALID_LABELS["s3"] else "Clear"
                else:
                    evidence_quality = "N/A"

                # ---- S4: RAG + LLM Council ----
                examples_s4           = retrieve_s4(col_s4, data)
                prompt_s4             = build_prompt_s4(data, examples_s4)
                verification_timeline = predict_s4(prompt_s4)

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

        # Lưu ngay từng dòng
        pd.DataFrame([out_row], columns=OUTPUT_COLUMNS).to_csv(
            output_file, mode="a", header=False,
            index=False, encoding="utf-8-sig", sep="\t"
        )

    df_out = pd.read_csv(output_file, sep="\t", dtype=str)
    print(f"\n{'='*60}")
    print(f"  Done! {len(df_out)}/{total} rows → {output_file}")
    print(f"{'='*60}")
    print(df_out.head(10).to_string(index=False))
    return df_out


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESG Pipeline v4")
    parser.add_argument("--test",  default=TEST_CSV)
    parser.add_argument("--train", default=JSON_TRAIN)
    parser.add_argument(
        "--output",
        default=os.path.join(
            OUTPUT_DIR,
            f"predictions_v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv"
        )
    )
    args = parser.parse_args()
    run_pipeline(args.test, args.train, args.output)