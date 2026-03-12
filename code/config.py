import os
from openai import OpenAI

# ==========================================
# 1. KET NOI SERVER (OLLAMA LOCAL)
# ==========================================
BASE_URL = "http://localhost:11434/v1"
API_KEY = "not-needed"
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ==========================================
# 2. CAU HINH XU LY (PERFORMANCE)
# ==========================================
CHUNK_SIZE   = 50     # So dong CSV gui di moi luot
TIMEOUT_LONG = 600    # Model lon (70B-72B)
TIMEOUT_SHORT = 300   # Model nho (14B-32B)

# ==========================================
# 3. BERT MODELS — S1 và S2
#
# S1 (promise_status):  MacBERT-large   (98.0%)
# S2 (evidence_status): RoBERTa-wwm     (97.0%)
# ==========================================
BERT_MODELS = {
    "s1": r"D:\LLMs\BERT-ESG\macbert-large-s1",
    "s2": r"D:\LLMs\BERT-ESG\roberta-wwm-large-s2",
}

# ==========================================
# 4. PRIORITY_MAP — LLM Council cho S3 và S4
#
# Voters (3):   qwen2.5-14b-esg / qwen2.5-72b-esg / deepseek-r1-70b-esg
# Chairman (1): qwen2.5-32b-esg (chi goi khi voters hoa 1-1-1)
# ==========================================
PRIORITY_MAP = {
    # --- Clarity Council (S3) ---
    "COUNCIL_CLARITY": [
        "qwen2.5-14b-esg",
        "qwen2.5-72b-esg",
        "deepseek-r1-70b-esg",
    ],
    # --- Timeline Council (S4) ---
    "COUNCIL_TIMELINE": [
        "qwen2.5-14b-esg",
        "qwen2.5-72b-esg",
        "deepseek-r1-70b-esg",
    ],
    # --- Chairman (chi goi khi voters hoa 1-1-1) ---
    "CHAIRMAN": [
        "qwen2.5-32b-esg",
    ],
}

# ==========================================
# 4. THONG SO TOI UU (PARAMS)
# ==========================================
PARAMS = {
    "synthesis": {      # Chairman
        "temperature": 0.0,
        "max_tokens": 20,
        "top_p": 1.0
    },
    "extraction": {     # Voters
        "temperature": 0.0,
        "max_tokens": 20,
        "top_p": 1.0
    }
}

# ==========================================
# 5. QUAN LY MODEL (MODEL MANAGER)
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
            print(f"[ERROR] Role '{role_clean}' khong ton tai trong PRIORITY_MAP.")
            return None
        models = PRIORITY_MAP[role_clean]
        if attempt < len(models):
            return models[attempt]
        print(f"[WARN] Da thu het model cho role {role_clean}.")
        return None

    @staticmethod
    def get_params(role):
        role_upper = role.upper()
        if "CHAIRMAN" in role_upper:
            return PARAMS["synthesis"]
        return PARAMS["extraction"]

    @staticmethod
    def get_timeout(model_name):
        name_lower = model_name.lower()
        if any(size in name_lower for size in ["72b", "70b"]):
            return TIMEOUT_LONG
        return TIMEOUT_SHORT

# ==========================================
# 6. OUTPUT FORMAT
# ==========================================
OUTPUT_COLUMNS = ["id", "promise_status", "verification_timeline", "evidence_status", "evidence_quality"]

# Giá trị mặc định khi promise_status = No
DEFAULT_WHEN_NO = {
    "verification_timeline": "N/A",
    "evidence_status":       "N/A",
    "evidence_quality":      "N/A",
}

# Valid labels cho từng subtask
VALID_LABELS = {
    "s1": ["Yes", "No"],
    "s2": ["Yes", "No"],
    "s3": ["Clear", "Not Clear", "Misleading", "N/A"],
    "s4": ["already", "within_2_years", "between_2_and_5_years", "longer_than_5_years"],
}

def build_output_row(row_id, promise_status, evidence_status=None,
                     evidence_quality=None, verification_timeline=None):
    """
    Tạo 1 row output đúng format cuộc thi.
    Nếu promise_status = No → các cột còn lại = N/A
    """
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
        "evidence_status":       evidence_status or "N/A",
        "evidence_quality":      evidence_quality or "N/A",
    }

print(f"ESG Council ready: {len(PRIORITY_MAP)} councils | 3 voters + 1 chairman")