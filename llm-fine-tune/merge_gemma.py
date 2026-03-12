# merge gemma2:72b to base model
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