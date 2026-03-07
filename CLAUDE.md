# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VeriPromise-ESG-2026 is a competition entry for the [AICup VeriPromise ESG challenge](https://www.aidea-web.tw/aicup_veripromiseesg). The goal is to analyze corporate ESG (Environmental, Social, Governance) disclosures in Traditional Chinese and classify them for:
- **promise_status** / **promise_string** — whether the text contains a commitment
- **verification_timeline** — when the commitment will be fulfilled (already, within_2_years, between_2_and_5_years, longer_than_5_years, N/A)
- **evidence_status** / **evidence_string** / **evidence_quality** — supporting evidence and its quality

## Architecture

The system uses local LLMs via **Ollama** (OpenAI-compatible API at `localhost:11434/v1`) with a "council" pattern — multiple large models (qwen2:72b, gemma2:27b, llama3:70b, deepseek-r1:70b) organized into councils that evaluate different aspects.

### Key Files

- `code/config.py` — Central configuration: Ollama connection, council definitions (`PRIORITY_MAP`), model parameters, and `ModelManager` class (model selection with priority fallback and VRAM failure blacklisting)
- `code/main.py` — Main pipeline: reads CSV dataset, sends each row to LLM with a detailed few-shot prompt, parses JSON response, saves results incrementally in batches of 20
- `code/slang.py` — Experimental script to build a slang/token dictionary by classifying tokens into timeline/evidence/clarity categories using LLM + jieba tokenization
- `code/test_models.py` — Tests all configured models across all councils
- `code/check_hethong.py` — System check: PyTorch, CUDA, GPU info, AMP verification

### Data

- `datasets/vpesg4k_train_1000 V1.csv` / `.json` — Training data (1000 samples)
- `datasets/slang_esc.json` — Generated slang dictionary

## Development Setup

Requires Conda environment with Python 3.10:
```bash
conda activate NLP
```

Key dependencies: `openai`, `pandas`, `transformers`, `sentence-transformers`, `jieba`, `nltk`, `torch`, `tqdm`, `chromadb`, `tiktoken`, `spacy`

Ollama must be running locally with the required models pulled (see `code/install.md` for full setup).

## Running

```bash
# Check system/GPU status
python code/check_hethong.py

# Test all council models are responding
python code/test_models.py

# Run main ESG analysis pipeline
python code/main.py
```

## Notes

- Code comments and documentation are primarily in Vietnamese
- The main pipeline supports resume — it skips already-processed IDs found in the output CSV
- `main.py` currently hardcodes `llama3:70b` rather than using `ModelManager` from config
- `slang.py` is marked as experimental ("Chay thu nghiem thoi, khong dung khi viet paper")
