# How to install Ollama on local machine
**Author:** thinhdv  
**Date:** 7 January 2026  

---

## Step 1. Create NLP environment in Conda  
(Install Conda first)

```
conda create -n NLP python=3.10
conda activate NLP
```

## Step 2. Install depedencies for NLP env via pip
```
pip install tiktoken chromadb pandas transformers sentence-transformers nltk spacy jieba tqdm openai 
```

## Step 3. Install Torch compatible with CUDA  
Get the right command from PyTorch official site.  
Example:  
```
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## Step 4. Download Ollama 
Download here from https://ollama.com/download

## Step 5. Install Ollama 

## Step 6. Create folders in D:\ (to save disk space in C:\)  
```
D:\
  └── .ollama\
       ├── models\
       ├── manifests\
       ├── blobs\
       └── config\
```
DON't DELETE .ollama folder in C:\USERS\...

## Step 7. Set models location
```
Open Ollama Settings  
Set Model location = D:\.ollama\models  
```

## Step 8. Install models via pull command
```
ollama pull llama3:8b
ollama pull llama3:70b
ollama pull deepseek-r1:8b
ollama pull deepseek-r1:70b
ollama pull gemma2:27b
ollama pull qwen2:72b
ollama pull gpt-oss:120b
```

## Step 9. Download llcouncil
```
git clone https://github.com/karpathy/llm-council.git
```

## Step 10. Edit config.py
```
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

```

## Step 11. Testing
```
import sys, os, importlib
from openai import OpenAI

# Thêm thư mục backend vào sys.path
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, "backend"))

# Import config
config = importlib.import_module("config")

client = config.client  # dùng client đã định nghĩa trong config.py

def test_models():
    for role, models in config.PRIORITY_MAP.items():
        print(f"\n=== Testing Council: {role} ===")
        for model in models:
            if model in config.ModelManager.FAILED_RESOURCES_MODELS:
                print(f"⏭️ Skipping blacklisted model: {model}")
                continue
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Say hello"}],
                    temperature=0
                )
                if resp and resp.choices:
                    print(f"✅ {model} Response:", resp.choices[0].message.content)
                else:
                    print(f"⚠️ {model} returned no choices:", resp)
            except Exception as e:
                print(f"❌ {model} API error:", e)
                config.ModelManager.mark_failed(model)

if __name__ == "__main__":

    # client = OpenAI(
    #     base_url="http://localhost:11434/v1",
    #     api_key="ollama"  # any string works
    # )
    # resp = client.models.list()
    # print(resp)

    test_models()
```

## Expected Output

C:\ProgramData\anaconda3\envs\NLP\python.exe E:\15.Demo1\llm-council\check_models_answer.py   
✅ ModelManager Streamlined: Ready for 16 synchronized councils.  

=== Testing Council: COUNCIL_ALLERGIES ===  
✅ llama3:8b Response: Hello! It's nice to meet you. Is there something I can help you with, or would you like to chat?  
✅ deepseek-r1:8b Response: Hello! 😊 How can I assist you today?  

