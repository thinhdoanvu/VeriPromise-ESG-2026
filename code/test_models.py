import sys, os, importlib
from openai import OpenAI

# Thêm thư mục backend vào sys.path
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, "backend"))

# Import config
import config

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
    test_models()
