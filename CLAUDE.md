# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VeriPromise-ESG-2026 is a competition entry for the [AICup VeriPromise ESG challenge](https://www.aidea-web.tw/aicup_veripromiseesg). The goal is to classify Traditional Chinese corporate ESG disclosures into:
- **promise_status** ("Yes"/"No") + **promise_string** — whether text contains a commitment
- **verification_timeline** — "already", "within_2_years", "between_2_and_5_years", "longer_than_5_years", "N/A"
- **evidence_status** ("Yes"/"No"/"N/A") + **evidence_string** + **evidence_quality** ("Clear"/"Not Clear"/"Misleading"/"N/A")

**Competition scoring:** `Total = Commitment_F1 × 0.20 + Evidence_F1 × 0.30 + Clarity_MacroF1 × 0.35 + Timeline_MacroF1 × 0.15`

Test submission window: June 10-17 2026.

## Development Setup

```bash
conda activate NLP
```

API keys are in `.env` (gitignored). Default backend is **OpenRouter** (`OPENROUTER_API_KEY`). Set `USE_OLLAMA=true` in `.env` to use local Ollama instead.

## Commands

```bash
# Run the main ESG classification pipeline (default: DeepSeek V3 via OpenRouter)
conda run -n NLP python code/main.py

# Run with a specific model (preset name or full model ID)
conda run -n NLP python code/main.py --model gemini-flash
conda run -n NLP python code/main.py --model qwen2.5-72b

# Run on a subset for testing
conda run -n NLP python code/main.py --limit 50 --output datasets/output_test.csv

# Evaluate predictions against training ground truth
conda run -n NLP python code/evaluate.py
conda run -n NLP python code/evaluate.py --predictions datasets/output_test.csv

# System checks
conda run -n NLP python code/check_hethong.py   # GPU/CUDA status
conda run -n NLP python code/test_models.py      # Test Ollama model connectivity
```

## Architecture

The pipeline sends each ESG text sample to an LLM with a few-shot prompt, parses the JSON response, validates fields with cascading logic, and saves results incrementally in batches of 20.

### Key Files

- `code/config.py` — API backend selection (OpenRouter vs Ollama), model presets (`MODELS` dict), `ENSEMBLE_MODELS` list, shared `client` instance. Loads `.env` for API keys.
- `code/main.py` — Main pipeline. Contains the full prompt (`SYSTEM_PROMPT` + `FEW_SHOT_EXAMPLES`), JSON parsing with regex fallback, field validation (`validate_result`), and retry logic. CLI args: `--model`, `--input`, `--output`, `--limit`.
- `code/evaluate.py` — Computes exact competition score. Loads GT from JSON (not CSV, to preserve "N/A" strings). Outputs per-task metrics, confusion matrices, and saves error details to `datasets/evaluation_errors.csv`.

### Data Flow

1. Input: `datasets/vpesg4k_train_1000 V1.csv` (or `.json` for ground truth)
2. Each row → `build_prompt()` → LLM API → `parse_json_response()` → `validate_result()` → output CSV
3. Resume support: skips IDs already present in output CSV

### Cascading Validation Logic (in `validate_result`)

- `promise_status="No"` → forces all downstream fields to N/A/empty
- `evidence_status` in ("No", "N/A") → forces `evidence_string=""`, `evidence_quality="N/A"`

### Model Presets (in `config.MODELS`)

Available via `--model` flag: `deepseek-v3`, `qwen3`, `qwen2.5-72b`, `gemini-flash`, `deepseek-r1`, `llama4-maverick`, `gpt-4.1-mini`, plus local Ollama models.

## Baseline Score (gemma3:27b local)

**0.6333** — Main weaknesses: Clarity Macro-F1 = 0.3181 (fails on "Not Clear"), Timeline Macro-F1 = 0.5127 (over-predicts `between_2_and_5_years`).

## Notes

- Code comments are primarily in Vietnamese
- The prompt uses 2026 as the reference year for timeline calculations
- Ground truth JSON should be used for evaluation (CSV loses "N/A" as NaN)
- `code/slang.py` is experimental and not part of the main pipeline
