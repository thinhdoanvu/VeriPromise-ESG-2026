import pandas as pd
import jieba
import re
import json
import nltk
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from tqdm import tqdm

# Chạy thử nghiệm thôi, không dùng khi viết paper
# =========================
# Paths
# =========================
INPUT_PATH = r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_100 V1.csv"
OUTPUT_PATH = r"C:\Users\VU\Documents\NLP\AICup26\datasets\slang_esc.json"

# =========================
# Load
# =========================
df = pd.read_csv(INPUT_PATH)
print("Samples:", len(df))

# =========================
# Clean
# =========================
def clean_text(text):
    text = str(text)
    text = text.replace("’", "'")
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9%.,!?;:'()（）\- ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

df["clean_data"] = df["data"].apply(clean_text)

# =========================
# Tokenize
# =========================
nltk.download("punkt")
def tokenize_mixed(text):
    text = re.sub(r"([A-Za-z]+)", r" \1 ", text)
    tokens = jieba.lcut(text)
    final_tokens = []
    for t in tokens:
        if re.match(r"[A-Za-z]+", t):
            final_tokens.extend(nltk.word_tokenize(t))
        else:
            final_tokens.append(t)
    return [t for t in final_tokens if len(t) > 1 and len(t) < 20]

df["tokens"] = df["clean_data"].apply(tokenize_mixed)

# =========================
# Embedding model
# =========================
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# =========================
# LLM client
# =========================
client = OpenAI(base_url="http://localhost:11434/v1", api_key="not-needed")

# =========================
# Pure LLM labeling
# =========================
slang_dict = {"timeline": [], "evidence": [], "clarity": []}

for idx, row in tqdm(df.iterrows(), total=len(df)):
    text = row["clean_data"]
    if not text:
        continue

    prompt = f"""
You are an ESG expert. 

Classify all meaningful tokens or phrases in this text into exactly three categories:
- timeline: any mention of time, schedule, year, deadline, frequency
- evidence: numeric data, measurements, emission, unit, quantity
- clarity: vague action, general statement, improvement, effort

Text: "{text}"

Answer ONLY in JSON format with keys: timeline, evidence, clarity.
Include all tokens/phrases you consider relevant. Do NOT invent tokens.
"""
    try:
        resp = client.chat.completions.create(
            model="llama3:70b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        parsed = json.loads(resp.choices[0].message.content.strip())
        for key in ["timeline", "evidence", "clarity"]:
            slang_dict[key].extend(parsed.get(key, []))
    except Exception:
        # fallback: all tokens go to clarity
        slang_dict["clarity"].extend(row["tokens"])

# =========================
# Post-processing
# =========================
for k in slang_dict:
    clean_tokens = set()
    for t in slang_dict[k]:
        t = t.strip()
        if len(t) > 1 and not re.match(r"^[^\w\u4e00-\u9fff]+$", t):
            clean_tokens.add(t)
    slang_dict[k] = list(clean_tokens)

# =========================
# Save
# =========================
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(slang_dict, f, ensure_ascii=False, indent=2)

print("Saved pure LLM slang dictionary:", OUTPUT_PATH)