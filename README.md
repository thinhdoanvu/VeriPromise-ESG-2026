# AICup26 — ESG Promise Verification

Introduction: https://veripromiseesg.github.io/  
Dataset: https://www.aidea-web.tw/aicup_veripromiseesg  
Register: https://go.aicup.tw/  

A 4-stage pipeline that classifies ESG (Environmental, Social, Governance)
commitment statements from corporate sustainability reports along four
dimensions: whether a statement is a **commitment** (S1), whether it is
**backed by evidence** (S2), how **clear** that evidence is (S3), and the
**timeline** for fulfilling the commitment (S4).

The pipeline combines fine-tuned BERT classifiers (S1–S3 and an S4 ensemble)
with a chain-of-thought LLM prompt (S4), reaching a weighted composite score
of **0.5939** (baseline pipeline: 0.5571).

## Repository Structure

```
config/      LLM endpoint configuration
data/        train / validation / test data
models/      trained BERT checkpoints (not included — see Setup)
src/
  training/  scripts to train each BERT classifier
  pipeline/  scripts to run inference and produce the final CSV
  utils/     submission-format validation
results/     generated prediction CSVs
report/      LaTeX write-up of methodology and results
```

## Setup

1. Install dependencies:
   ```bash
   pip install torch transformers scikit-learn pandas openai
   ```

2. Place the dataset files under `data/` (train/val/test JSON and CSV).

3. Edit `config/config.py` to point at your LLM endpoint (used only by S4's
   chain-of-thought step) — set the base URL and the three model names
   (two voters + one chairman).

4. Edit the path constants at the top of each script in `src/training/` and
   `src/pipeline/` to point at your local `data/`, `models/`, and `results/`
   directories.

## Usage

### 1. Train the classifiers (one-time, GPU required)

```bash
python src/training/train_bert_s1_s2_v6_grid.py   # -> models/macbert-large-s1-v6, models/roberta-wwm-large-s2-v6
python src/training/train_bert_s4_v1.py           # -> models/grid_s4_v1/{a,b,c,d}
python src/training/train_bert_s3_grid.py         # -> models/macbert-large-s3-v2
```

### 2. Run the base pipeline

```bash
python src/pipeline/main_pipeline_v4.py --output results/old.csv
```

Runs S1–S3 (BERT) and S4 (LLM council) end-to-end on the test set.

### 3. Apply the S4 and S3 refinements

```bash
python src/pipeline/verify_already_with_bert.py   # results/old.csv -> results/predictions_v9.csv
python src/pipeline/apply_s3_v2.py                # results/predictions_v9.csv -> results/predictions_v10.csv
```

### 4. Validate before submission

```bash
python src/utils/fix_v13.py                       # checks/fixes results/predictions_v10.csv
```

`results/predictions_v10.csv` is the final file to submit.

## Results

| Subtask | Description | Weight | Macro-F1 |
|---|---|---|---|
| S1 — `promise_status` | Identify ESG commitment statements | 20% | 0.7682 |
| S2 — `evidence_status` | Is the commitment backed by evidence? | 30% | 0.6629 |
| S3 — `evidence_quality` | Evidence clarity (Clear / Not Clear / Misleading) | 35% | 0.4322 |
| S4 — `verification_timeline` | Commitment completion timeframe | 15% | 0.6011 |
| **Composite** | | 100% | **0.5939** |

Two refinements account for most of the improvement over the baseline
(0.5571):

- **S4 (step 3a):** the LLM council is precise when a deadline year is
  explicitly mentioned, but defaults to "already" otherwise. BERT is better
  at classifying those default cases. Combining both raised S4 from
  0.5277 → 0.5945.
- **S3 (step 3b):** retraining S3 produced a predicted class distribution
  much closer to the training distribution, raising S3 from 0.4210 → 0.4317.

Full methodology, ablations, and discussion are in
[`report/results_section.tex`](report/results_section.tex),
[`report/system_overview.tex`](report/system_overview.tex), and
[`report/cot_methodology.tex`](report/cot_methodology.tex).
