from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import gc
import os
import shutil

# ==========================================
# CONFIG
# ==========================================
# Dùng ổ C (NVMe) làm offload tạm cho model lớn 72B/70B
OFFLOAD_DIR = r"C:\LLM_offload_tmp"

# Kiểm soát bộ nhớ chặt — tránh OOM
# VRAM:  90GB dùng + 6GB buffer  = 96GB tổng
# RAM:  120GB dùng + 8GB buffer  = 128GB tổng
# NVMe: phần còn lại tự động offload
MAX_MEMORY = {
    0: "90GB",       # GPU 0 (RTX A6000 PRO 96GB)
    "cpu": "120GB"   # RAM (128GB - 8GB buffer cho Windows)
}

MODELS = [
    # ✅ Đã merge — comment lại
    # {
    #     "name": "qwen2.5-14b",
    #     "base_path":   r"D:\LLMs\Qwen2.5-14B",
    #     "lora_path":   r"D:\LLMs\Qwen2.5-14B\esg-lora-v1",
    #     "merged_path": r"D:\LLMs\Qwen2.5-14B\esg-merged",
    #     "need_offload": False,   # 28GB — đủ RAM
    # },
    # {
    #     "name": "qwen2.5-32b",
    #     "base_path":   r"D:\LLMs\Qwen2.5-32B",
    #     "lora_path":   r"D:\LLMs\Qwen2.5-32B\esg-lora-v1",
    #     "merged_path": r"D:\LLMs\Qwen2.5-32B\esg-merged",
    #     "need_offload": False,   # 65GB — đủ RAM
    # },

    # ⏳ Chưa merge
    # {
    #     "name": "qwen2.5-72b",
    #     "base_path":   r"D:\LLMs\Qwen2.5-72B",
    #     "lora_path":   r"D:\LLMs\Qwen2.5-72B\esg-lora-v1",
    #     "merged_path": r"D:\LLMs\Qwen2.5-72B\esg-merged",
    #     "need_offload": True,    # 144GB — cần offload sang NVMe
    # },

    # ⏳ Chưa merge (uncomment sau khi train xong)
    {
        "name": "deepseek-r1-70b",
        "base_path":   r"D:\LLMs\DeepSeek-R1-70B",
        "lora_path":   r"D:\LLMs\DeepSeek-R1-70B\esg-lora-v1",
        "merged_path": r"D:\LLMs\DeepSeek-R1-70B\esg-merged",
        "need_offload": True,    # 140GB — cần offload sang NVMe
    },
]

# ==========================================
# MERGE 1 MODEL
# ==========================================
def merge_model(cfg):
    print(f"\n{'='*60}")
    print(f"🔀 Merging: {cfg['name']}")
    print(f"   Base:         {cfg['base_path']}")
    print(f"   LoRA:         {cfg['lora_path']}")
    print(f"   Output:       {cfg['merged_path']}")
    print(f"   Need offload: {cfg['need_offload']}")
    print(f"{'='*60}")

    # Tạo offload dir và output dir
    if cfg["need_offload"]:
        os.makedirs(OFFLOAD_DIR, exist_ok=True)
        print(f"   📁 Offload dir: {OFFLOAD_DIR}")
        print(f"   💾 Max memory: VRAM=90GB | RAM=120GB | NVMe=overflow")
    os.makedirs(cfg["merged_path"], exist_ok=True)

    # [1/4] Load base model
    print(f"   [1/4] Loading base model...")
    if cfg["need_offload"]:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_path"],
            dtype=torch.float16,
            device_map="auto",
            max_memory=MAX_MEMORY,        # ← giới hạn VRAM + RAM, phần dư → NVMe
            offload_folder=OFFLOAD_DIR,   # ← offload layers xuống NVMe
            offload_state_dict=True,      # ← offload state dict tiết kiệm RAM
            local_files_only=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["base_path"],
            dtype=torch.float16,
            device_map="auto",
            local_files_only=True
        )

    # [2/4] Load tokenizer
    print(f"   [2/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["lora_path"],
        local_files_only=True
    )

    # [3/4] Load LoRA adapter & merge
    print(f"   [3/4] Loading LoRA adapter & merging...")
    if cfg["need_offload"]:
        model = PeftModel.from_pretrained(
            model,
            cfg["lora_path"],
            device_map="auto",
            max_memory=MAX_MEMORY,        # ← giới hạn bộ nhớ cho LoRA
            offload_dir=OFFLOAD_DIR,      # ← chỉ dùng offload_dir, không dùng offload_folder
        )
    else:
        model = PeftModel.from_pretrained(
            model,
            cfg["lora_path"],
            device_map="auto"
        )

    model = model.merge_and_unload()
    print(f"   ✅ LoRA merged successfully!")

    # [4/4] Save merged model
    print(f"   [4/4] Saving merged model...")
    model.save_pretrained(
        cfg["merged_path"],
        safe_serialization=True,
        max_shard_size="5GB"          # ← 5GB mỗi shard để tiết kiệm RAM khi save
    )
    tokenizer.save_pretrained(cfg["merged_path"])
    print(f"✅ [{cfg['name']}] Saved to {cfg['merged_path']}")

    # Xóa offload tmp
    if cfg["need_offload"] and os.path.exists(OFFLOAD_DIR):
        shutil.rmtree(OFFLOAD_DIR, ignore_errors=True)
        print(f"   🗑️  Offload tmp cleaned: {OFFLOAD_DIR}")

    # Giải phóng VRAM + RAM
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"🧹 VRAM + RAM cleared after {cfg['name']}\n")


# ==========================================
# MAIN — Merge tuần tự
# ==========================================
if __name__ == "__main__":
    total = len(MODELS)
    for i, cfg in enumerate(MODELS):
        print(f"\n[{i+1}/{total}] Starting {cfg['name']}...")
        merge_model(cfg)

    print("\n🎉 Tất cả models đã được merge xong!")
    print("\nMerged models saved tại:")
    for cfg in MODELS:
        print(f"  → {cfg['merged_path']}")

    print("\n⏭️  Bước tiếp theo: Convert sang GGUF bằng llama.cpp")
    print("   cd D:\\LLMs\\llama.cpp")
    print("   python convert_hf_to_gguf.py <merged_path> --outtype f16")