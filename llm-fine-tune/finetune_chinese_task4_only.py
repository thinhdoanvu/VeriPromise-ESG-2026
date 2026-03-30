"""
Train LLM v3 — S4 Only (verification_timeline)
================================================
Fix already bias bằng cách:
1. Undersample already ×0.4 (chỉ giữ 40%)
2. Oversample between_2_and_5_years ×3
3. Oversample longer_than_5_years ×3
4. Oversample within_2_years ×15 (giữ nguyên)

Output: esg-lora-v3 (S4 only)
"""

import json
import random
import gc
from collections import Counter

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ==========================================
# CONFIG
# ==========================================
JSON_FILE = r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_1000 V1.json"

MODELS = [
    {
        "name":           "qwen2.5-14b",
        "model_path":     r"D:\LLMs\Qwen2.5-14B",
        "output_dir":     r"D:\LLMs\Qwen2.5-14B\esg-lora-v3",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "batch_size":     2,
        "lora_r":         16,
    },
    {
        "name":           "qwen2.5-32b",
        "model_path":     r"D:\LLMs\Qwen2.5-32B",
        "output_dir":     r"D:\LLMs\Qwen2.5-32B\esg-lora-v3",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "batch_size":     1,
        "lora_r":         16,
    },
    {
        "name":           "qwen2.5-72b",
        "model_path":     r"D:\LLMs\Qwen2.5-72B",
        "output_dir":     r"D:\LLMs\Qwen2.5-72B\esg-lora-v3",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "batch_size":     1,
        "lora_r":         16,
    },
    {
        "name":           "deepseek-r1-70b",
        "model_path":     r"D:\LLMs\DeepSeek-R1-70B",
        "output_dir":     r"D:\LLMs\DeepSeek-R1-70B\esg-lora-v3",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "batch_size":     1,
        "lora_r":         16,
    },
]

# Oversample config
REPEAT = {
    "already":                 1,    # giữ nguyên (sẽ undersample sau)
    "within_2_years":          15,   # rất ít samples
    "between_2_and_5_years":   3,    # recall thấp → tăng
    "longer_than_5_years":     3,    # recall thấp → tăng
}
ALREADY_KEEP_RATIO = 0.4   # undersample already → giữ 40%

VALID_S4 = ["already", "within_2_years", "between_2_and_5_years", "longer_than_5_years"]

# ==========================================
# BUILD DATASET — S4 only
# ==========================================
def build_dataset(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    random.seed(42)
    random.shuffle(records)
    train_records = records[:800]

    # Khai báo S4 prompt
    def make_s4_example(data, hint_promise, timeline):
        return {"text": (
            "You are an ESG analyst. The ESG report was published in 2024.\n"
            "Classify the expected completion timeframe of the ESG commitment.\n"
            "Choose exactly one:\n"
            "- already (action already completed or continuously ongoing in 2024, NO specific future year)\n"
            "- within_2_years (explicit target year 2025 or 2026)\n"
            "- between_2_and_5_years (explicit target year 2027, 2028, or 2029)\n"
            "- longer_than_5_years (explicit target year 2030 or beyond)\n\n"
            "IMPORTANT: If a specific future year is mentioned, classify by that year.\n"
            "Only choose 'already' if there is NO future target year.\n\n"
            f"Statement: {data}{hint_promise}\n"
            f"Answer: {timeline}"
        )}

    # Build examples với oversample
    examples_by_label = {label: [] for label in VALID_S4}

    for row in train_records:
        promise  = str(row.get("promise_status", "")).strip()
        timeline = str(row.get("verification_timeline", "")).strip()

        if promise  != "Yes":       continue
        if timeline not in VALID_S4: continue

        data           = str(row["data"]).strip()
        promise_string = str(row.get("promise_string") or "").strip()
        hint_promise   = f"\nKey commitment: {promise_string}" if promise_string else ""

        example = make_s4_example(data, hint_promise, timeline)
        repeat  = REPEAT[timeline]

        for _ in range(repeat):
            examples_by_label[timeline].append(example)

    # Undersample already
    random.shuffle(examples_by_label["already"])
    keep_n = max(1, int(len(examples_by_label["already"]) * ALREADY_KEEP_RATIO))
    examples_by_label["already"] = examples_by_label["already"][:keep_n]

    # Merge và shuffle
    all_examples = []
    for label, exs in examples_by_label.items():
        all_examples.extend(exs)
        print(f"  {label:<30}: {len(exs)} examples")

    random.shuffle(all_examples)

    print(f"\n  Total: {len(all_examples)} examples")

    full  = Dataset.from_list(all_examples)
    split = full.train_test_split(test_size=0.1, seed=42)
    print(f"  Train: {len(split['train'])} | Eval: {len(split['test'])}")

    return split["train"], split["test"]

# ==========================================
# TRAIN 1 MODEL
# ==========================================
def train_model(cfg, train_dataset, eval_dataset):
    print(f"\n{'='*60}")
    print(f"  Fine-tuning: {cfg['name']}")
    print(f"  Output:      {cfg['output_dir']}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_path"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        local_files_only=True
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_r"] * 2,
        target_modules=cfg["target_modules"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    sft_config = SFTConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=6,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        report_to="none",
        max_length=1024,
        dataset_text_field="text",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        args=sft_config,
    )

    trainer.train()
    trainer.model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"  ✅ Saved: {cfg['output_dir']}")

    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  VRAM cleared\n")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("  LLM v3 — S4 Timeline (fix already bias)")
    print("=" * 60)

    print("\nBuilding dataset...")
    train_dataset, eval_dataset = build_dataset(JSON_FILE)

    for i, cfg in enumerate(MODELS):
        print(f"\n[{i+1}/{len(MODELS)}] {cfg['name']}")
        train_model(cfg, train_dataset, eval_dataset)

    print("\n" + "=" * 60)
    print("Done! Next steps:")
    print("=" * 60)
    for cfg in MODELS:
        print(f"  {cfg['name']:20s} → {cfg['output_dir']}")
    print("1. Merge: merge_all_models.py  (chỉ v3 models)")
    print("2. GGUF: convert_guff.bat")
    print("3. Quantize để giảm size (tùy VRAM): quantize.bat")
    print("4. Tạo Modelfile cho Ollama -> Import vào Ollama: Modelfiles\ollama_import_all.bat")
    print("5. Dọn dẹp ổ cứng: cleanup.bat")
    print("6. Evaluate: main_pipeline_v4.py  (dùng esg-lora-v3)")