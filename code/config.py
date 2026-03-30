import os
from openai import OpenAI

# ==========================================
# 1. KET NOI SERVER (OLLAMA LOCAL)
# ==========================================
BASE_URL = "http://localhost:11434/v1"
API_KEY  = "not-needed"
client   = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ==========================================
# 2. CAU HINH XU LY (PERFORMANCE)
# ==========================================
CHUNK_SIZE    = 50
TIMEOUT_LONG  = 1200   # 70B/72B — 20 phut
TIMEOUT_SHORT = 600    # 14B/32B — 10 phut
TIMEOUT_CHAIR = 1200   # Chairman 32B

# ==========================================
# 3. BERT MODELS
# S1: MacBERT-large  (98.0%)
# S2: RoBERTa-wwm    (97.0%)
# S3: MacBERT-large  (Binary: Clear/Not Clear)
# ==========================================
BERT_MODELS = {
    "s1": r"D:\LLMs\BERT-ESG\macbert-large-s1",
    "s2": r"D:\LLMs\BERT-ESG\roberta-wwm-large-s2",
    "s3": r"D:\LLMs\BERT-ESG\macbert-large-s3",
}

# ==========================================
# 4. PRIORITY_MAP — LLM Council S4 only
# Voters (3):   14b / 72b / deepseek-70b
# Chairman (1): 32b (chi goi khi tie)
# ==========================================
PRIORITY_MAP = {
    "COUNCIL_TIMELINE": [
        "qwen2.5-14b-esg",
        "qwen2.5-72b-esg",
        "deepseek-r1-70b-esg",
    ],
    "CHAIRMAN": [
        "qwen2.5-32b-esg",
    ],
}

# ==========================================
# 5. PARAMS
# ==========================================
PARAMS = {
    "synthesis":  {"temperature": 0.0, "max_tokens": 20, "top_p": 1.0},
    "extraction": {"temperature": 0.0, "max_tokens": 20, "top_p": 1.0},
}

# ==========================================
# 6. MODEL MANAGER
# ==========================================
class ModelManager:
    FAILED_RESOURCES_MODELS = set()

    @classmethod
    def mark_failed(cls, model_name):
        if model_name not in cls.FAILED_RESOURCES_MODELS:
            print(f"[SYSTEM] Blacklisting {model_name}")
            cls.FAILED_RESOURCES_MODELS.add(model_name)

    @staticmethod
    def get_model(role, attempt=0):
        role_clean = role.upper().strip()
        if role_clean not in PRIORITY_MAP:
            print(f"[ERROR] Role '{role_clean}' not in PRIORITY_MAP.")
            return None
        models = PRIORITY_MAP[role_clean]
        if attempt < len(models):
            return models[attempt]
        print(f"[WARN] No more models for role {role_clean}.")
        return None

    @staticmethod
    def get_params(role):
        if "CHAIRMAN" in role.upper():
            return PARAMS["synthesis"]
        return PARAMS["extraction"]

    @staticmethod
    def get_timeout(model_name):
        n = model_name.lower()
        if any(s in n for s in ["72b", "70b"]):
            return TIMEOUT_LONG
        if "32b" in n:
            return TIMEOUT_CHAIR
        return TIMEOUT_SHORT

# ==========================================
# 7. OUTPUT FORMAT
# ==========================================
OUTPUT_COLUMNS = [
    "id", "promise_status", "verification_timeline",
    "evidence_status", "evidence_quality"
]

DEFAULT_WHEN_NO = {
    "verification_timeline": "N/A",
    "evidence_status":       "N/A",
    "evidence_quality":      "N/A",
}

VALID_LABELS = {
    "s1": ["Yes", "No"],
    "s2": ["Yes", "No"],
    "s3": ["Clear", "Not Clear"],
    "s4": ["already", "within_2_years", "between_2_and_5_years", "longer_than_5_years"],
}

def build_output_row(row_id, promise_status, evidence_status=None,
                     evidence_quality=None, verification_timeline=None):
    if promise_status == "No":
        return {
            "id":                    row_id,
            "promise_status":        "No",
            "verification_timeline": DEFAULT_WHEN_NO["verification_timeline"],
            "evidence_status":       DEFAULT_WHEN_NO["evidence_status"],
            "evidence_quality":      DEFAULT_WHEN_NO["evidence_quality"],
        }
    return {
        "id":                    row_id,
        "promise_status":        promise_status,
        "verification_timeline": verification_timeline or "N/A",
        "evidence_status":       evidence_status       or "N/A",
        "evidence_quality":      evidence_quality      or "N/A",
    }

print(f"ESG Config ready: BERT(S1/S2/S3) + LLM Council(S4) | "
      f"{len(PRIORITY_MAP['COUNCIL_TIMELINE'])} voters + 1 chairman")