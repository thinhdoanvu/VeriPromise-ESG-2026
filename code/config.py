import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

# ==========================================
# 1. API BACKEND SELECTION
# ==========================================
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"

if USE_OLLAMA:
    BASE_URL = "http://localhost:11434/v1"
    API_KEY = "not-needed"
    DEFAULT_MODEL = "gemma3:27b"
else:
    BASE_URL = "https://openrouter.ai/api/v1"
    API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

# ==========================================
# 2. MODEL PRESETS
# ==========================================
MODELS = {
    "deepseek-v3": "deepseek/deepseek-chat-v3-0324",
    "qwen3": "qwen/qwen3-235b-a22b",
    "qwen2.5-72b": "qwen/qwen-2.5-72b-instruct",
    "gemini-flash": "google/gemini-2.5-flash",
    "deepseek-r1": "deepseek/deepseek-r1",
    "llama4-maverick": "meta-llama/llama-4-maverick",
    "gpt-4.1-mini": "openai/gpt-4.1-mini",
    # Local Ollama models
    "gemma3:27b": "gemma3:27b",
    "qwen2:72b": "qwen2:72b",
    "llama3:70b": "llama3:70b",
}

# ==========================================
# 3. COUNCIL DEFINITIONS (for ensemble voting)
# ==========================================
ENSEMBLE_MODELS = [
    "deepseek/deepseek-chat-v3-0324",
    "qwen/qwen-2.5-72b-instruct",
    "google/gemini-2.5-flash",
]

# ==========================================
# 4. PARAMETERS
# ==========================================
PARAMS = {
    "temperature": 0.0,
    "max_tokens": 4000,
}

# ==========================================
# 5. PERFORMANCE CONFIG
# ==========================================
BATCH_SIZE = 20
CHUNK_SIZE = 100
TIMEOUT_LONG = 600
TIMEOUT_SHORT = 180

print(f"Config: backend={'Ollama' if USE_OLLAMA else 'OpenRouter'}, model={DEFAULT_MODEL}")
