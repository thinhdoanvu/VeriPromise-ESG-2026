r"""
S3 ENSEMBLE — Test 4 grid configs (s3-v2-a/b/c/d) tren held-out-200
========================================================================
S3-v2-a (os=2,w=4.0, ~=v6+noise) won overall (val=0.7101), nhung b/c/d
(oversample NotClear cao hon: 4/5/6) co the la "Not Clear specialists"
-- giong S4's c/d (oversample khac -> decision boundary khac, errors
it tuong quan -> ensemble giup).

v10 (dung 'a' alone) da cai thien S3 0.4210->0.4317 (+0.0107, dung
huong distribution-matching-train: Clear 90.6%->78.2%, gan train 81.7%).

Test cac strategies ensemble (tuong tu debug_ensemble_s4.py):
  A. a alone                    (baseline, val=0.7101)
  B. Majority vote (4 models, tie->a)
  C. a + override->NotClear neu b,c,d DEU noi NotClear (b/c/d agree)
  D. a + override->NotClear neu (b OR c OR d) noi NotClear (any-agree,
     more aggressive)

Cach dung: chay truc tiep (PyCharm Run hoac python -u debug_ensemble_s3.py)
"""
import json, random
from collections import Counter
from sklearn.metrics import classification_report, f1_score

import torch
from transformers import pipeline as hf_pipeline

TRAIN_JSON = r"C:\Users\VU\Documents\NLP\AICup26\datasets\vpesg4k_train_1000 V1.json"
SEED = 42
DEVICE = 0 if torch.cuda.is_available() else -1

VALID_S3 = ["Clear", "Not Clear"]

MODEL_PATHS = {
    "a": r"D:\LLMs\BERT-ESG\grid_s3_v2\s3-v2-a",
    "b": r"D:\LLMs\BERT-ESG\grid_s3_v2\s3-v2-b",
    "c": r"D:\LLMs\BERT-ESG\grid_s3_v2\s3-v2-c",
    "d": r"D:\LLMs\BERT-ESG\grid_s3_v2\s3-v2-d",
}

# ==========================================
# LOAD DATA — 141 held-out (S3-eligible, Clear/NotClear only)
# ==========================================
print("Loading data...", flush=True)
with open(TRAIN_JSON, "r", encoding="utf-8") as f:
    train_records = json.load(f)

random.seed(SEED)
random.shuffle(train_records)
held_out_200 = train_records[800:]

eligible = [
    r for r in held_out_200
    if str(r.get("promise_status","")).strip()=="Yes"
    and str(r.get("evidence_status","")).strip()=="Yes"
    and str(r.get("evidence_quality","")).strip() in VALID_S3
]
print(f"Eligible: {len(eligible)} rows", flush=True)

texts = [str(r["data"]).strip() for r in eligible]
gts   = [str(r["evidence_quality"]).strip() for r in eligible]
print(f"GT distribution: {dict(Counter(gts))}", flush=True)

# ==========================================
# LOAD ALL 4 MODELS, GET PREDICTIONS (batch)
# ==========================================
preds = {}
for name, path in MODEL_PATHS.items():
    print(f"\nLoading model {name} from {path} ...", flush=True)
    clf = hf_pipeline(
        "text-classification", model=path, tokenizer=path,
        device=DEVICE, truncation=True, max_length=512, batch_size=32,
    )
    outs = clf(texts)
    labels = [o["label"] if o["label"] in VALID_S3 else "Clear" for o in outs]
    preds[name] = labels
    print(f"  Distribution: {dict(Counter(labels))}", flush=True)
    del clf
    torch.cuda.empty_cache()

# ==========================================
# STRATEGIES
# ==========================================
def strategy_A():
    return list(preds["a"])

def strategy_B():
    out = []
    for i in range(len(texts)):
        votes = [preds[m][i] for m in ["a","b","c","d"]]
        counter = Counter(votes)
        top = counter.most_common()
        if len(top)==1 or top[0][1] > top[1][1]:
            out.append(top[0][0])
        else:
            out.append(preds["a"][i])
    return out

def strategy_C():
    """a + override->NotClear if b,c,d ALL agree NotClear AND a='Clear'"""
    out = []
    for i in range(len(texts)):
        if preds["a"][i]=="Clear" and all(preds[m][i]=="Not Clear" for m in ["b","c","d"]):
            out.append("Not Clear")
        else:
            out.append(preds["a"][i])
    return out

def strategy_D():
    """a + override->NotClear if ANY of b/c/d say NotClear AND a='Clear'"""
    out = []
    for i in range(len(texts)):
        if preds["a"][i]=="Clear" and any(preds[m][i]=="Not Clear" for m in ["b","c","d"]):
            out.append("Not Clear")
        else:
            out.append(preds["a"][i])
    return out


def report_for(pred_labels, name):
    y_true = [VALID_S3.index(g) for g in gts]
    y_pred = [VALID_S3.index(p) for p in pred_labels]
    macro = f1_score(y_true, y_pred, average="macro")
    print(f"\n{'='*60}")
    print(f"  {name} — Macro F1 = {macro:.4f}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=VALID_S3,
                                  labels=list(range(len(VALID_S3))), zero_division=0))
    pred_dist = Counter(pred_labels)
    total = len(pred_labels)
    print(f"  Predicted distribution: "
          f"{ {k: f'{v/total*100:.1f}%' for k,v in pred_dist.items()} }")
    return macro


results = {}
results["A. a alone"]                  = report_for(strategy_A(), "A. a alone")
results["B. Majority vote (4)"]        = report_for(strategy_B(), "B. Majority vote (a/b/c/d)")
results["C. a + b&c&d agree NotClear"] = report_for(strategy_C(), "C. a + (b,c,d ALL agree NotClear)")
results["D. a + any of b/c/d NotClear"]= report_for(strategy_D(), "D. a + (ANY of b/c/d agree NotClear)")

print("\nSUMMARY:")
baseline = results["A. a alone"]
for name, macro in results.items():
    delta = macro - baseline
    print(f"  {name:<35} {macro:.4f}  (delta vs A: {delta:+.4f})")

print(f"\nTrain distribution (target): Clear=81.7%, NotClear=18.3%")
print(f"Current v10 test (S3-v2-a):    composite evidence_quality=0.4322")