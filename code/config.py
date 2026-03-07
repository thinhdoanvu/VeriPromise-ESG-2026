import os
from openai import OpenAI

# ==========================================
# 1. KẾT NỐI SERVER (OLLAMA LOCAL)
# ==========================================
BASE_URL = "http://localhost:11434/v1"
API_KEY = "not-needed"
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ==========================================
# 2. CẤU HÌNH XỬ LÝ (PERFORMANCE)
# ==========================================
CHUNK_SIZE = 100      # Số dòng CSV gửi đi mỗi lượt
TIMEOUT_LONG = 600    # Dành cho Model lớn (70B) khi tổng hợp
TIMEOUT_SHORT = 180   # Dành cho Model nhỏ (8B–27B) khi bóc tách

# ==========================================
# 3. CHIA LẠI HỘI ĐỒNG (PRIORITY_MAP)
# Mỗi Key tương ứng với một Council trong Quy trình 2
# ==========================================
PRIORITY_MAP = {
    # --- Commitment Council ---
    "COUNCIL_COMMITMENT": [
        "qwen2:72b",
        "gemma2:27b",
        "llama3:70b",
        "deepseek-r1:70b",
    ],

    # --- Timeline Council ---
    "COUNCIL_TIMELINE": [
        "qwen2:72b",
        "gemma2:27b",
        "llama3:70b",
        "deepseek-r1:70b",
    ],

    # --- Evidence Council ---
    "COUNCIL_EVIDENCE": [
        "qwen2:72b",
        "gemma2:27b",
        "llama3:70b",
        "deepseek-r1:70b",
    ],

    # --- Clarity Council ---
    "COUNCIL_CLARITY": [
        "gemma2:27b",
        "llama3:70b",
        "qwen2:72b",
        "deepseek-r1:70b",
    ],

    # --- Chairman (tổng hợp kết quả) ---
    "CHAIRMAN": [
        "llama3:70b",
        "qwen2:72b",
        "deepseek-r1:70b",
    ],

    # --- Client (các model nhỏ/trung bình) ---
    "CLIENT": [
        "qwen2:72b",
        "gemma2:27b",
        "llama3:70b",
        "deepseek-r1:70b",
    ],
}

# ==========================================
# 4. THÔNG SỐ TỐI ƯU (PARAMS)
# ==========================================
PARAMS = {
    "synthesis": {  # Dành cho Chairman, cần suy luận
        "temperature": 0.0,
        "max_tokens": 8000,
        "top_p": 0.1
    },
    "extraction": { # Dành cho Clients, cần chính xác
        "temperature": 0.0,
        "max_tokens": 4000,
        "top_p": 0.1
    }
}

# ==========================================
# 5. QUẢN LÝ XOAY VÒNG & BLACKLIST (MODEL MANAGER)
# ==========================================
class ModelManager:
    FAILED_RESOURCES_MODELS = set()  # Danh sách đen các model gây lỗi VRAM

    @classmethod
    def mark_failed(cls, model_name):
        """Đánh dấu model bị lỗi để các Council sau không gọi nhầm"""
        if model_name not in cls.FAILED_RESOURCES_MODELS:
            print(f"⚠️ [SYSTEM] Blacklisting {model_name} do lỗi VRAM/Resource.")
            cls.FAILED_RESOURCES_MODELS.add(model_name)

    @staticmethod
    def get_model(role, attempt=0):
        """
        Lấy model theo thứ tự ưu tiên từ trên xuống dưới trong PRIORITY_MAP.
        Nếu attempt vượt quá số lượng model hiện có, trả về None.
        """
        role_clean = role.upper().strip()

        if role_clean not in PRIORITY_MAP:
            print(f"[ERROR] Role '{role_clean}' không tồn tại trong PRIORITY_MAP.")
            return None

        models = PRIORITY_MAP[role_clean]

        if attempt < len(models):
            return models[attempt]
        else:
            print(f"[WARN] Đã thử hết toàn bộ model cho role {role_clean}.")
            return None

    @staticmethod
    def get_params(role):
        """Tự động trả về bộ Params dựa trên tính chất công việc của Agent"""
        role_upper = role.upper()
        synthesis_keywords = ["SUMMARY", "COMMITMENT", "TIMELINE", "EVIDENCE", "CLARITY"]

        if any(keyword in role_upper for keyword in synthesis_keywords):
            return PARAMS["synthesis"]
        return PARAMS["extraction"]

    @staticmethod
    def get_timeout(model_name):
        """Quyết định thời gian chờ dựa trên kích thước model"""
        name_lower = model_name.lower()
        if any(size in name_lower for size in ["70b", "qwen2", "deepseek-r1"]):
            return TIMEOUT_LONG
        return TIMEOUT_SHORT


print(f"✅ ModelManager: Ready for {len(PRIORITY_MAP)} ESG councils.")
