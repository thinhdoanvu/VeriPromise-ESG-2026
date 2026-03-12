"""
RAG Council Voting — S3 và S4 dùng ChromaDB
- Embedding: BAAI/bge-large-zh-v1.5
- Vector store: ChromaDB (local persistent)
- LLM: Ollama (qwen2.5-14b-esg, qwen2.5-72b-esg, deepseek-r1-70b-esg)
- Chairman: qwen2.5-32b-esg (chỉ khi tie)
"""

import json
import os
import requests
import pandas as pd
from datetime import datetime
from collections import Counter
import chromadb
from chromadb.utils import embedding_functions

# ==========================================
# CONFIG
# ==========================================
JSON_TRAIN  = r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_1000 V1.json"
TEST_CSV    = r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_1000 V1_test200.csv"
OUTPUT_DIR  = r"/AICup26/results"
CHROMA_DIR  = r"/AICup26/chroma_db"
OLLAMA_URL  = "http://localhost:11434/api/generate"

EMBED_MODEL = "BAAI/bge-large-zh-v1.5"
TOP_K       = 6

VOTERS      = ["qwen2.5-14b-esg", "qwen2.5-72b-esg", "deepseek-r1-70b-esg"]
CHAIRMAN    = "qwen2.5-32b-esg"

VALID_LABELS = {
    "s3": ["Clear", "Not Clear", "Misleading", "N/A"],
    "s4": ["already", "within_2_years", "between_2_and_5_years", "longer_than_5_years"],
}

# ==========================================
# LOAD TRAINING DATA
# ==========================================
def load_train_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        records = json.load(f)
    import random
    random.seed(42)
    random.shuffle(records)
    train = records[:800]
    print(f"Train records loaded: {len(train)}")
    return train

# ==========================================
# BUILD CHROMADB COLLECTIONS
# ==========================================
def build_chroma_collections(train_records, embed_model_name, chroma_dir):
    os.makedirs(chroma_dir, exist_ok=True)
    print(f"\nInitializing ChromaDB at: {chroma_dir}")

    client = chromadb.PersistentClient(path=chroma_dir)

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embed_model_name
    )

    # Reset collections nếu đã tồn tại
    for col_name in ["esg_s3", "esg_s4"]:
        try:
            client.delete_collection(col_name)
        except Exception:
            pass

    col_s3 = client.create_collection(
        name="esg_s3",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )
    col_s4 = client.create_collection(
        name="esg_s4",
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"}
    )

    docs_s3, ids_s3, meta_s3 = [], [], []
    docs_s4, ids_s4, meta_s4 = [], [], []

    for i, r in enumerate(train_records):
        data           = str(r["data"]).strip()
        promise_status = str(r.get("promise_status") or "").strip()
        quality        = str(r.get("evidence_quality") or "N/A").strip()
        timeline       = str(r.get("verification_timeline") or "").strip()

        if promise_status != "Yes":
            continue

        if quality in VALID_LABELS["s3"]:
            docs_s3.append(data)
            ids_s3.append(f"s3_{i}")
            meta_s3.append({"evidence_quality": quality})

        if timeline in VALID_LABELS["s4"]:
            docs_s4.append(data)
            ids_s4.append(f"s4_{i}")
            meta_s4.append({"verification_timeline": timeline})

    # Add in batches of 100
    BATCH = 100
    print(f"Adding {len(docs_s3)} docs to S3 collection...")
    for i in range(0, len(docs_s3), BATCH):
        col_s3.add(
            documents=docs_s3[i:i+BATCH],
            ids=ids_s3[i:i+BATCH],
            metadatas=meta_s3[i:i+BATCH]
        )

    print(f"Adding {len(docs_s4)} docs to S4 collection...")
    for i in range(0, len(docs_s4), BATCH):
        col_s4.add(
            documents=docs_s4[i:i+BATCH],
            ids=ids_s4[i:i+BATCH],
            metadatas=meta_s4[i:i+BATCH]
        )

    print(f"ChromaDB ready: S3={col_s3.count()} docs, S4={col_s4.count()} docs")
    return col_s3, col_s4

# ==========================================
# RETRIEVE BALANCED EXAMPLES
# ChromaDB where filter → lấy từng label riêng → cân bằng hơn FAISS
# ==========================================
def retrieve_balanced(collection, query_text, task, k=6):
    labels    = VALID_LABELS[task]
    label_key = "evidence_quality" if task == "s3" else "verification_timeline"
    per_label = max(1, k // len(labels))
    examples  = []

    for label in labels:
        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=per_label,
                where={label_key: {"$eq": label}}
            )
            docs  = results["documents"][0]
            metas = results["metadatas"][0]
            for doc, meta in zip(docs, metas):
                examples.append({"data": doc, label_key: meta[label_key]})
        except Exception:
            continue

    return examples[:k]

# ==========================================
# BUILD RAG PROMPTS
# ==========================================
def build_rag_prompt_s3(query_data, examples):
    few_shot = ""
    for ex in examples:
        few_shot += f"Statement: {ex['data']}\nAnswer: {ex['evidence_quality']}\n\n"

    return (
        "You are an ESG analyst. "
        "Evaluate whether the following ESG statement contains semantically clear evidence.\n"
        "Choose exactly one:\n"
        "- Clear (specific, measurable, verifiable, no vague wording)\n"
        "- Not Clear (vague phrases like '致力於', '積極推動', '逐步實施', '持續改善')\n"
        "- Misleading (potentially misleading or greenwashing language)\n"
        "- N/A (no evidence present)\n\n"
        f"{few_shot}"
        f"Statement: {query_data}\n"
        "Answer:"
    )

def build_rag_prompt_s4(query_data, examples):
    few_shot = ""
    for ex in examples:
        few_shot += f"Statement: {ex['data']}\nAnswer: {ex['verification_timeline']}\n\n"

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

# ==========================================
# OLLAMA INFERENCE
# ==========================================
def ollama_predict(model_name, prompt, valid_labels):
    payload = {
        "model":  model_name,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 20, "top_p": 1}
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        for label in valid_labels:
            if label.lower() in raw.lower():
                return label, raw
        return "UNKNOWN", raw
    except Exception as e:
        return "ERROR", str(e)

# ==========================================
# COUNCIL VOTE
# ==========================================
def council_vote(votes, valid_labels):
    counter = Counter(v for v in votes if v in valid_labels)
    if not counter:
        return valid_labels[0], False
    top = counter.most_common(1)[0]
    if top[1] == 1:
        return None, True   # tie
    return top[0], False

# ==========================================
# MAIN EVALUATION
# ==========================================
def evaluate(col_s3, col_s4, test_df):
    results  = []
    chair_s3 = chair_s4 = 0
    total    = len(test_df)

    for idx, (_, row) in enumerate(test_df.iterrows()):
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{total}] Processing...")

        data        = str(row["data"]).strip()
        promise_gt  = str(row["promise_status"]).strip()
        quality_gt  = str(row["evidence_quality"]).strip()  if pd.notna(row.get("evidence_quality"))  else "N/A"
        timeline_gt = str(row["verification_timeline"]).strip() if pd.notna(row.get("verification_timeline")) else ""

        s3_pred = s4_pred = "SKIP"
        s3_chair = s4_chair = False

        if promise_gt == "Yes":
            # ---- S3 ----
            examples_s3 = retrieve_balanced(col_s3, data, "s3", TOP_K)
            prompt_s3   = build_rag_prompt_s3(data, examples_s3)

            votes_s3 = [ollama_predict(v, prompt_s3, VALID_LABELS["s3"])[0] for v in VOTERS]
            s3_pred, need_chair = council_vote(votes_s3, VALID_LABELS["s3"])
            if need_chair:
                s3_pred, _ = ollama_predict(CHAIRMAN, prompt_s3, VALID_LABELS["s3"])
                s3_chair = True
                chair_s3 += 1

            # ---- S4 ----
            if timeline_gt in VALID_LABELS["s4"]:
                examples_s4 = retrieve_balanced(col_s4, data, "s4", TOP_K)
                prompt_s4   = build_rag_prompt_s4(data, examples_s4)

                votes_s4 = [ollama_predict(v, prompt_s4, VALID_LABELS["s4"])[0] for v in VOTERS]
                s4_pred, need_chair = council_vote(votes_s4, VALID_LABELS["s4"])
                if need_chair:
                    s4_pred, _ = ollama_predict(CHAIRMAN, prompt_s4, VALID_LABELS["s4"])
                    s4_chair = True
                    chair_s4 += 1

        results.append({
            "id":         row["id"],
            "s3_gt":      quality_gt  if promise_gt == "Yes" else "SKIP",
            "s4_gt":      timeline_gt if promise_gt == "Yes" and timeline_gt in VALID_LABELS["s4"] else "SKIP",
            "s3_pred":    s3_pred,
            "s4_pred":    s4_pred,
            "s3_correct": (s3_pred == quality_gt)  if promise_gt == "Yes" else None,
            "s4_correct": (s4_pred == timeline_gt) if promise_gt == "Yes" and timeline_gt in VALID_LABELS["s4"] else None,
            "s3_chair":   s3_chair,
            "s4_chair":   s4_chair,
        })

    df = pd.DataFrame(results)

    print(f"\n{'='*60}")
    print(f"  ACCURACY REPORT: RAG Council (ChromaDB)")
    print(f"{'='*60}")

    for task, label in [("s3", "S3 Clarity"), ("s4", "S4 Timeline")]:
        valid = df[df[f"{task}_correct"].notna()]
        if len(valid) == 0:
            continue
        correct = int(valid[f"{task}_correct"].sum())
        count   = len(valid)
        print(f"  {label:20s}: {correct}/{count} = {correct/count*100:.1f}%")
        for gt_label in sorted(valid[f"{task}_gt"].unique()):
            if gt_label in ["SKIP", ""]:
                continue
            mask = valid[f"{task}_gt"] == gt_label
            lc   = int(valid[mask][f"{task}_correct"].sum())
            lct  = int(mask.sum())
            print(f"    L {gt_label:30s}: {lc:3d}/{lct} = {lc/lct*100:.1f}%")

    print(f"\n  Chairman called: S3={chair_s3}, S4={chair_s4}")
    print(f"{'='*60}")
    return df


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_records  = load_train_data(JSON_TRAIN)
    col_s3, col_s4 = build_chroma_collections(train_records, EMBED_MODEL, CHROMA_DIR)

    test_df = pd.read_csv(TEST_CSV)
    print(f"\nTest set: {len(test_df)} rows")

    df_results = evaluate(col_s3, col_s4, test_df)

    out_csv = os.path.join(OUTPUT_DIR, f"rag_chroma_council_{timestamp}.csv")
    df_results.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\nCSV saved: {out_csv}")