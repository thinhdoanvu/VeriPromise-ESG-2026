import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer
from transformers import BitsAndBytesConfig
import torch
from trl import SFTConfig

# =================== CONFIG ===================
MODEL_PATH = r"D:\LLMs\Gemma2-27B"
CSV_FILE = r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_1000 V1.csv"
OUTPUT_DIR = r"D:\LLMs\Gemma2-27B\gemma_promise_lora"

# =================== 1. Load CSV ===================
df = pd.read_csv(CSV_FILE)

texts = []
for _, row in df.iterrows():
    prompt = (
        f"Determine whether the ESG statement contains a commitment or promise. "
        f"Answer only Yes or No.\n"
        f"Statement: {row['data']}\n"
        f"Answer: {row['promise_status']}"
    )
    texts.append({"text": prompt})

dataset = Dataset.from_list(texts)

# =================== 2. Tokenizer ===================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # FIX: needed for batching

# =================== 3. Load model — use bfloat16 compute ===================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,   # FIX: bfloat16 is stable, float16 overflows
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.bfloat16   # FIX: use dtype= not torch_dtype= (removes deprecation warning)
)
model.config.use_cache = False  # FIX: required for gradient checkpointing

# =================== 4. LoRA Config ===================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",           # FIX: explicit
    task_type="CAUSAL_LM"
)

# =================== 5. Training Arguments ===================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    max_grad_norm=1.0,
    fp16=False,
    bf16=True,             # FIX: match compute dtype — use bf16=True
    gradient_checkpointing=True,  # FIX: reduces VRAM, prevents OOM-related NaN
    optim="paged_adamw_8bit",     # FIX: stable optimizer for QLoRA
    warmup_ratio=0.03,            # FIX: warmup prevents early NaN spikes
    report_to="none"
)

# =================== 6. SFT Trainer ===================
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    max_grad_norm=1.0,
    fp16=False,
    bf16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    warmup_ratio=0.03,
    report_to="none",
    max_length=512,              # FIX: was max_seq_length in older TRL
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=sft_config,
)

# =================== 7. Train ===================
trainer.train()

# =================== 8. Save ===================
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ LoRA fine-tuned model saved at", OUTPUT_DIR)