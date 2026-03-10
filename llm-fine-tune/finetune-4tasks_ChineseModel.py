import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch
import gc

# ==========================================
# CONFIG
# ==========================================
CSV_FILE = r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_1000 V1.csv"

MODELS = [
    # --- Voters ---
    {
        "name": "qwen2.5-14b",
        "role": "voter",
        "model_path": r"D:\LLMs\Qwen2.5-14B",
        "output_dir": r"D:\LLMs\Qwen2.5-14B\esg-lora-v1",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "batch_size": 2,
        "lora_r": 16,
    },
    {
        "name": "qwen2.5-72b",
        "role": "voter",
        "model_path": r"D:\LLMs\Qwen2.5-72B",
        "output_dir": r"D:\LLMs\Qwen2.5-72B\esg-lora-v1",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "batch_size": 1,
        "lora_r": 16,
    },
    {
        "name": "deepseek-r1-70b",
        "role": "voter",
        "model_path": r"D:\LLMs\DeepSeek-R1-70B",
        "output_dir": r"D:\LLMs\DeepSeek-R1-70B\esg-lora-v1",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "batch_size": 1,
        "lora_r": 16,
    },
    # --- Chairman ---
    {
        "name": "qwen2.5-32b",
        "role": "chairman",
        "model_path": r"D:\LLMs\Qwen2.5-32B",
        "output_dir": r"D:\LLMs\Qwen2.5-32B\esg-lora-v1",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "batch_size": 1,
        "lora_r": 16,
    },
]

# ==========================================
# BUILD DATASET — 800 train / 200 test
#
# Subtask 1: ALL rows          → promise_status       (Yes / No)
# Subtask 2: promise = Yes     → evidence_status      (Yes / No)
# Subtask 3: promise = Yes     → evidence_quality     (Clear / Not Clear / Misleading / N/A)
#            ↑ bao gồm cả evidence=No → N/A
# Subtask 4: promise = Yes     → verification_timeline
#
# 800 dòng → build examples
#     ↓ train_test_split
#     ├── 90% → train_dataset
#     └── 10% → eval_dataset
# ==========================================
def build_dataset(csv_file):
    df = pd.read_csv(csv_file)

    # Shuffle với seed cố định để reproducible
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df_train = df.iloc[:800]   # 80%
    df_test  = df.iloc[800:]   # 20%

    # Lưu test set ra file để dùng sau
    test_path = csv_file.replace(".csv", "_test200.csv")
    df_test.to_csv(test_path, index=False)
    print(f"✅ Test set (200 rows) saved to: {test_path}")

    examples = []
    s1_count = s2_count = s3_count = s4_count = 0

    for _, row in df_train.iterrows():
        data             = str(row["data"]).strip()
        promise_status   = str(row["promise_status"]).strip()
        evidence_status  = str(row["evidence_status"]).strip() if pd.notna(row["evidence_status"]) else "No"
        evidence_quality = str(row["evidence_quality"]).strip() if pd.notna(row["evidence_quality"]) else "N/A"
        timeline         = str(row["verification_timeline"]).strip() if pd.notna(row["verification_timeline"]) else ""

        # --------------------------------------------------
        # Subtask 1: Commitment Classification
        # Input:  data
        # Output: Yes / No
        # Condition: ALL rows
        # --------------------------------------------------
        examples.append({"text": (
            "You are an ESG analyst. "
            "Does the following ESG statement express a concrete corporate commitment or promise toward future actions?\n"
            "Answer only: Yes or No\n\n"
            f"Statement: {data}\n"
            f"Answer: {promise_status}"
        )})
        s1_count += 1

        if promise_status == "Yes":

            # --------------------------------------------------
            # Subtask 2: Evidence Identification
            # Input:  data
            # Output: Yes / No
            # Condition: promise_status = Yes
            # --------------------------------------------------
            examples.append({"text": (
                "You are an ESG analyst. "
                "Is the following ESG commitment supported by concrete evidence such as action plans, data, methodologies, or implementation records?\n"
                "Answer only: Yes or No\n\n"
                f"Statement: {data}\n"
                f"Answer: {evidence_status}"
            )})
            s2_count += 1

            # --------------------------------------------------
            # Subtask 3: Clarity Classification
            # Input:  data
            # Output: Clear / Not Clear / Misleading / N/A
            # Condition: promise_status = Yes
            #   - evidence = Yes → Clear / Not Clear / Misleading
            #   - evidence = No  → N/A
            # --------------------------------------------------
            examples.append({"text": (
                "You are an ESG analyst. "
                "Evaluate whether the following ESG statement contains semantically clear evidence.\n"
                "Choose exactly one:\n"
                "- Clear (specific, measurable, verifiable, no vague wording)\n"
                "- Not Clear (vague phrases like 'continuously improving', 'striving to achieve', 'actively promoting')\n"
                "- Misleading (potentially misleading or greenwashing language)\n"
                "- N/A (no evidence present)\n\n"
                f"Statement: {data}\n"
                f"Answer: {evidence_quality}"
            )})
            s3_count += 1

            # --------------------------------------------------
            # Subtask 4: Timeline Classification
            # Input:  data
            # Output: already / within_2_years / between_2_and_5_years / longer_than_5_years
            # Condition: promise_status = Yes
            # --------------------------------------------------
            if timeline:
                examples.append({"text": (
                    "You are an ESG analyst. The ESG report was published in 2024.\n"
                    "Based on the statement, classify the expected completion timeframe.\n"
                    "Choose exactly one:\n"
                    "- already (action already implemented or ongoing in 2024, no future deadline mentioned)\n"
                    "- within_2_years (specific target year 2025 or 2026 mentioned)\n"
                    "- between_2_and_5_years (target year 2027, 2028, or 2029 mentioned)\n"
                    "- longer_than_5_years (target year 2030 or beyond mentioned, e.g. 2030, 2050)\n\n"
                    f"Statement: {data}\n"
                    f"Answer: {timeline}"
                )})
                s4_count += 1

    # Tách 90% train / 10% eval từ tập examples
    full = Dataset.from_list(examples)
    split = full.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    eval_dataset  = split["test"]

    print(f"\n✅ Dataset built:")
    print(f"   Subtask 1 (Commitment):  {s1_count} examples")
    print(f"   Subtask 2 (Evidence):    {s2_count} examples")
    print(f"   Subtask 3 (Clarity):     {s3_count} examples  ← bao gồm N/A")
    print(f"   Subtask 4 (Timeline):    {s4_count} examples")
    print(f"   Total:                   {len(examples)} examples")
    print(f"   → Train split:           {len(train_dataset)} examples (90%)")
    print(f"   → Eval split:            {len(eval_dataset)} examples (10%)")

    return train_dataset, eval_dataset


# ==========================================
# TRAIN 1 MODEL
# ==========================================
def train_model(cfg, train_dataset, eval_dataset):
    print(f"\n{'='*60}")
    print(f"🚀 Fine-tuning: {cfg['name']} ({cfg['role']})")
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
        dtype=torch.bfloat16
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
        # --- Validation settings ---
        eval_strategy="epoch",              # evaluate sau mỗi epoch
        save_strategy="epoch",              # lưu checkpoint sau mỗi epoch
        load_best_model_at_end=True,        # tự động load epoch tốt nhất
        metric_for_best_model="eval_loss",  # chọn model dựa trên eval_loss
        greater_is_better=False,            # eval_loss càng thấp càng tốt
        # ---------------------------
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        report_to="none",
        max_length=1024,
        dataset_text_field="text",
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
    print(f"✅ [{cfg['name']}] Saved to {cfg['output_dir']}")

    # Giải phóng VRAM trước model tiếp theo
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"🧹 VRAM cleared after {cfg['name']}\n")


# ==========================================
# MAIN — Train tuần tự
# ==========================================
if __name__ == "__main__":
    # Build dataset 1 lần, dùng cho cả 4 models
    train_dataset, eval_dataset = build_dataset(CSV_FILE)

    total = len(MODELS)
    for i, cfg in enumerate(MODELS):
        role_label = "👑 Chairman" if cfg["role"] == "chairman" else "🗳️  Voter"
        print(f"\n[{i+1}/{total}] {role_label} — Starting {cfg['name']}...")
        train_model(cfg, train_dataset, eval_dataset)

    print("\n🎉 Tất cả model đã được fine-tune xong!")
    print("\nVoter adapters:")
    for cfg in [m for m in MODELS if m["role"] == "voter"]:
        print(f"  🗳️   → {cfg['output_dir']}")
    print("\nChairman adapter:")
    for cfg in [m for m in MODELS if m["role"] == "chairman"]:
        print(f"  👑  → {cfg['output_dir']}")