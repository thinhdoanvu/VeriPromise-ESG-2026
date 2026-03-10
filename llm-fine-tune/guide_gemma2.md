Pull model to ollama  
   ↓  
Download model  
   ↓  
Fine-tune bằng HuggingFace (QLoRA / LoRA): chỉ có mỗi promise_status  
   ↓  
save adapter checkpoint  
   ↓  
merge adapter → model  
   ↓  
convert → GGUF  
   ↓  
load vào Ollama  


## Pull model to ollama
```
ollama pull qwen2.5:32b
```
---

## Download model using huggingface-cli
```
huggingface-cli download google/gemma-2-27b --local-dir D:\LLMs\gemma2-27b --local-dir-use-symlinks False
```
---

## Cài 1 số thư viện bổ sung
```
pip install transformers datasets peft accelerate bitsandbytes trl
```

## Script train LoRA (VRAM ~16GB): finetune_promise_status_gemma.py
```
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
```

## Checkpoint sẽ nằm ở đâu?
Dung lượng: ~300MB – 600MB (chỉ LoRA adapter)  
```
D:\LLMs\Gemma2-27B\gemma_promise_lora\
├── adapter_config.json       ← LoRA config
├── adapter_model.safetensors ← LoRA weights (nhỏ, ~100MB)
├── tokenizer.json
└── tokenizer_config.json
```
---

## Merge LoRA vào model
Ollama không hỗ trợ LoRA adapter trực tiếp — bạn cần merge LoRA vào base model trước, rồi convert sang định dạng GGUF. 
```
# merge_model.py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch

MODEL_PATH = r"D:\LLMs\Gemma2-27B"
LORA_PATH = r"D:\LLMs\Gemma2-27B\gemma_promise_lora"
MERGED_PATH = r"D:\LLMs\Gemma2-27B\gemma_promise_merged"

# Load base model trong float16 (KHÔNG dùng 4bit để merge)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="cpu"   # merge trên CPU để tránh OOM
)

tokenizer = AutoTokenizer.from_pretrained(LORA_PATH)

# Load và merge LoRA
model = PeftModel.from_pretrained(model, LORA_PATH)
model = model.merge_and_unload()

# Lưu merged model
model.save_pretrained(MERGED_PATH)
tokenizer.save_pretrained(MERGED_PATH)
print("✅ Merged model saved!")
```
---

###  Convert sang GGUF bằng llama.cpp
Clone llama.cpp (nếu chưa có)
Check lại là vẫn đang ở trong thư mục `D:\LLMs\`
```
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
pip install -r requirements.txt
```

### Convert sang GGUF
Cần `copy D:\LLMs\Gemma2-27B\tokenizer.model` đến folder `D:\LLMs\Gemma2-27B\gemma_promise_merged`. Sau đó
```
python convert_hf_to_gguf.py D:\LLMs\Gemma2-27B\gemma_promise_merged --outfile D:\LLMs\Gemma2-27B\gemma_promise_merged\gemma-promise.gguf --outtype f16
```
---

## Sau đó quantize để giảm size (tùy VRAM):

Build llama.cpp trước (Windows)
Vẫn đang trong thư mục: `D:\LLMs\llama.cpp`
```
cmake -B build
cmake --build build --config Release
```
Quantize xuống Q4_K_M (~15GB cho 27B thay vì 54GB)
```
.\build\bin\Release\llama-quantize.exe D:\LLMs\Gemma2-27B\gemma_promise_merged\gemma-promise.gguf D:\LLMs\Gemma2-27B\gemma_promise_merged\gemma-promise-q4km.gguf Q4_K_M
```
---

## Tạo Modelfile cho Ollama  
Tạo file text tên `Modelfile` (không có extension) tại `D:\LLMs\Gemma2-27B\gemma_promise_merged\`, sao chép nội dung sau vào Modelfile file
```
`D:\LLMs\Gemma2-27B\gemma_promise_merged\`
```

```
FROM D:\LLMs\Gemma2-27B\gemma_promise_merged\gemma-promise-q4km.gguf

SYSTEM """
You are an ESG analyst specialized in Chinese ESG reports.
Analyze ESG statements and answer classification questions accurately.
Answer only with the exact option requested.
"""

PARAMETER temperature 0
PARAMETER top_p 1
PARAMETER num_predict 20
```
---

## Import vào Ollama

```
ollama create gemma2-esg:27b -f D:\LLMs\Gemma2-27B\gemma_promise_merged\Modelfile
```
Kết quả: `sha256-84d2b791f5d44c909787d80fd960acca97f1c046135c3ac778e2d199cd9c6a46` có dung lượng `16GB` được lưu trữ ở `D:\.ollama\models\blobs`

## Test thử
Lưu ý tên model của mình bây giờ là  `gemma2-esg:27b`

```
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "gemma2-esg:27b",
        "prompt": "Does this ESG statement express a commitment? Answer Yes or No.\n\nStatement: 為支持同仁與其家人度過人生不同階段，自 2024 年起提供女性員工在分娩前後計有 12 週共 84 天的產假\n\nAnswer:",
        "stream": False,
        "options": {"temperature": 0, "num_predict": 20}
    }
)
print(response.json()["response"])
```
OUTPUT là `YES`.
