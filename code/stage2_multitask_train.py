"""
Stage 2: Multi-task BERT for joint promise/evidence/quality classification.

3 heads sharing one encoder:
  Head 1: promise_status  (Yes/No)      — all 1000 samples, weight=0.2353
  Head 2: evidence_status (Yes/No)      — 814 samples (promise=Yes), weight=0.3529
  Head 3: evidence_quality(Clear/NotClear) — 677 samples (evidence=Yes), weight=0.4118

Weights are competition weights (0.20/0.30/0.35) renormalized without timeline (÷0.85).
Misleading (1 sample) merged into Not Clear.

Keeps all improvements from Stage 1:
  FGM, R-Drop, EMA, weighted CE, cosine schedule, threshold tuning, early stopping.

Usage:
    conda run -n NLP python code/stage2_multitask_train.py
    conda run -n NLP python code/stage2_multitask_train.py --models hfl/chinese-macbert-large
    conda run -n NLP python code/stage2_multitask_train.py --epochs 20 --lr 2e-5
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

# ─── Config ───────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "datasets", "vpesg4k_train_1000 V1.json")
MODEL_DIR = os.path.join(BASE_DIR, "models", "stage2")

ENCODER_MODELS = [
    "hfl/chinese-roberta-wwm-ext-large",
    "ckiplab/bert-base-chinese",
    "hfl/chinese-macbert-large",
]

MAX_LEN = 512
SEED = 42

# Competition weights renormalized (without timeline 0.15)
TASK_WEIGHTS = {
    "promise": 0.20 / 0.85,   # 0.2353
    "evidence": 0.30 / 0.85,  # 0.3529
    "quality": 0.35 / 0.85,   # 0.4118
}

MULTI_SAMPLE_DROPOUT_K = 5


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ─── FGM Adversarial Training ────────────────────────────────────────────────

class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name="word_embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ─── EMA ──────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


# ─── Multi-Task Model ────────────────────────────────────────────────────────

class MultiTaskBERT(nn.Module):
    def __init__(self, model_name, n_promise=2, n_evidence=2, n_quality=2,
                 dropout_rate=0.1, multi_sample_k=MULTI_SAMPLE_DROPOUT_K):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.multi_sample_k = multi_sample_k
        self.dropout = nn.Dropout(dropout_rate)
        self.head_promise = nn.Linear(hidden_size, n_promise)
        self.head_evidence = nn.Linear(hidden_size, n_evidence)
        self.head_quality = nn.Linear(hidden_size, n_quality)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0]  # [CLS] token

        if self.training and self.multi_sample_k > 1:
            # Multi-sample dropout: apply dropout K times, average logits
            logits_p = torch.mean(torch.stack(
                [self.head_promise(self.dropout(cls)) for _ in range(self.multi_sample_k)]
            ), dim=0)
            logits_e = torch.mean(torch.stack(
                [self.head_evidence(self.dropout(cls)) for _ in range(self.multi_sample_k)]
            ), dim=0)
            logits_q = torch.mean(torch.stack(
                [self.head_quality(self.dropout(cls)) for _ in range(self.multi_sample_k)]
            ), dim=0)
        else:
            cls = self.dropout(cls)
            logits_p = self.head_promise(cls)
            logits_e = self.head_evidence(cls)
            logits_q = self.head_quality(cls)

        return logits_p, logits_e, logits_q


# ─── Dataset ──────────────────────────────────────────────────────────────────

class MultiTaskESGDataset(Dataset):
    def __init__(self, texts, labels_promise, labels_evidence, labels_quality,
                 masks_evidence, masks_quality, tokenizer, max_len=MAX_LEN):
        self.texts = texts
        self.labels_promise = labels_promise
        self.labels_evidence = labels_evidence
        self.labels_quality = labels_quality
        self.masks_evidence = masks_evidence
        self.masks_quality = masks_quality
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label_promise": torch.tensor(self.labels_promise[idx], dtype=torch.long),
            "label_evidence": torch.tensor(self.labels_evidence[idx], dtype=torch.long),
            "label_quality": torch.tensor(self.labels_quality[idx], dtype=torch.long),
            "mask_evidence": torch.tensor(self.masks_evidence[idx], dtype=torch.float),
            "mask_quality": torch.tensor(self.masks_quality[idx], dtype=torch.float),
        }


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_multitask_data():
    """Load data with labels for all 3 tasks + masks + combined label for stratification."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    texts = []
    labels_promise = []
    labels_evidence = []
    labels_quality = []
    masks_evidence = []
    masks_quality = []
    combined_labels = []  # for stratified K-fold

    quality_map = {"Clear": 1, "Not Clear": 0, "Misleading": 0}  # merge Misleading → Not Clear

    for d in raw:
        texts.append(d["data"])

        # Promise
        p = 1 if d["promise_status"] == "Yes" else 0
        labels_promise.append(p)

        # Evidence (valid only when promise=Yes)
        if p == 1:
            e = 1 if d["evidence_status"] == "Yes" else 0
            labels_evidence.append(e)
            masks_evidence.append(1)
        else:
            labels_evidence.append(0)  # placeholder, won't be used
            masks_evidence.append(0)

        # Quality (valid only when evidence=Yes AND promise=Yes)
        if p == 1 and d["evidence_status"] == "Yes":
            q = quality_map.get(d["evidence_quality"], 0)
            labels_quality.append(q)
            masks_quality.append(1)
        else:
            labels_quality.append(0)  # placeholder
            masks_quality.append(0)

        # Combined label for stratification (4 classes)
        if p == 0:
            combined_labels.append(0)  # promise=No
        elif d["evidence_status"] != "Yes":
            combined_labels.append(1)  # promise=Yes, evidence=No
        elif quality_map.get(d["evidence_quality"], 0) == 1:
            combined_labels.append(2)  # promise=Yes, evidence=Yes, Clear
        else:
            combined_labels.append(3)  # promise=Yes, evidence=Yes, Not Clear

    return (texts,
            np.array(labels_promise), np.array(labels_evidence), np.array(labels_quality),
            np.array(masks_evidence), np.array(masks_quality),
            np.array(combined_labels))


def compute_class_weights(labels, mask=None):
    """Compute inverse-frequency class weights, optionally masked."""
    if mask is not None:
        labels = labels[mask.astype(bool)]
    counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(counts) * counts.astype(float))
    return torch.FloatTensor(weights)


# ─── Loss computation ─────────────────────────────────────────────────────────

def compute_multitask_loss(logits_p, logits_e, logits_q,
                           labels_p, labels_e, labels_q,
                           mask_e, mask_q, ce_fns):
    """Masked multi-task loss with competition weights."""
    loss_p = ce_fns["promise"](logits_p, labels_p)

    loss_e = torch.tensor(0.0, device=logits_p.device)
    if mask_e.sum() > 0:
        idx_e = mask_e.bool()
        loss_e = ce_fns["evidence"](logits_e[idx_e], labels_e[idx_e])

    loss_q = torch.tensor(0.0, device=logits_p.device)
    if mask_q.sum() > 0:
        idx_q = mask_q.bool()
        loss_q = ce_fns["quality"](logits_q[idx_q], labels_q[idx_q])

    return (TASK_WEIGHTS["promise"] * loss_p +
            TASK_WEIGHTS["evidence"] * loss_e +
            TASK_WEIGHTS["quality"] * loss_q)


def compute_multitask_rdrop_loss(logits1, logits2, labels_p, labels_e, labels_q,
                                  mask_e, mask_q, ce_fns, alpha=0.5):
    """R-Drop: KL consistency between 2 forward passes for all 3 heads."""
    total_ce = torch.tensor(0.0, device=labels_p.device)
    total_kl = torch.tensor(0.0, device=labels_p.device)

    tasks = [
        ("promise", logits1[0], logits2[0], labels_p, None),
        ("evidence", logits1[1], logits2[1], labels_e, mask_e),
        ("quality", logits1[2], logits2[2], labels_q, mask_q),
    ]

    for task_name, l1, l2, labels, mask in tasks:
        if mask is not None:
            if mask.sum() == 0:
                continue
            idx = mask.bool()
            l1, l2, labels = l1[idx], l2[idx], labels[idx]

        ce = (ce_fns[task_name](l1, labels) + ce_fns[task_name](l2, labels)) / 2
        p1 = F.log_softmax(l1, dim=-1)
        p2 = F.log_softmax(l2, dim=-1)
        q1 = F.softmax(l1, dim=-1)
        q2 = F.softmax(l2, dim=-1)
        kl = (F.kl_div(p1, q2, reduction="batchmean") +
              F.kl_div(p2, q1, reduction="batchmean")) / 2

        w = TASK_WEIGHTS[task_name]
        total_ce += w * ce
        total_kl += w * kl

    return total_ce + alpha * total_kl


# ─── Training ─────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, device, fgm, ema,
                    ce_fns, use_rdrop=True, rdrop_alpha=0.5, grad_accum_steps=1):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels_p = batch["label_promise"].to(device)
        labels_e = batch["label_evidence"].to(device)
        labels_q = batch["label_quality"].to(device)
        mask_e = batch["mask_evidence"].to(device)
        mask_q = batch["mask_quality"].to(device)

        # Forward pass 1
        logits1 = model(input_ids, attention_mask)

        if use_rdrop:
            logits2 = model(input_ids, attention_mask)
            loss = compute_multitask_rdrop_loss(
                logits1, logits2, labels_p, labels_e, labels_q,
                mask_e, mask_q, ce_fns, rdrop_alpha
            )
        else:
            loss = compute_multitask_loss(
                logits1[0], logits1[1], logits1[2],
                labels_p, labels_e, labels_q,
                mask_e, mask_q, ce_fns
            )

        loss = loss / grad_accum_steps
        loss.backward()

        # FGM adversarial training
        fgm.attack(epsilon=1.0)
        logits_adv = model(input_ids, attention_mask)
        loss_adv = compute_multitask_loss(
            logits_adv[0], logits_adv[1], logits_adv[2],
            labels_p, labels_e, labels_q,
            mask_e, mask_q, ce_fns
        ) / grad_accum_steps
        loss_adv.backward()
        fgm.restore()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            ema.update()

        total_loss += loss.item() * grad_accum_steps
    return total_loss / len(loader)


# ─── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(model, loader, device):
    """Evaluate and return per-task probs, preds, labels, masks."""
    model.eval()
    all_probs = {"promise": [], "evidence": [], "quality": []}
    all_labels = {"promise": [], "evidence": [], "quality": []}
    all_masks = {"evidence": [], "quality": []}

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits_p, logits_e, logits_q = model(input_ids, attention_mask)

        all_probs["promise"].append(torch.softmax(logits_p, dim=-1).cpu().numpy())
        all_probs["evidence"].append(torch.softmax(logits_e, dim=-1).cpu().numpy())
        all_probs["quality"].append(torch.softmax(logits_q, dim=-1).cpu().numpy())

        all_labels["promise"].append(batch["label_promise"].numpy())
        all_labels["evidence"].append(batch["label_evidence"].numpy())
        all_labels["quality"].append(batch["label_quality"].numpy())
        all_masks["evidence"].append(batch["mask_evidence"].numpy())
        all_masks["quality"].append(batch["mask_quality"].numpy())

    result = {}
    for task in ["promise", "evidence", "quality"]:
        probs = np.concatenate(all_probs[task], axis=0)
        labels = np.concatenate(all_labels[task], axis=0)
        result[task] = {"probs": probs, "labels": labels}

    for task in ["evidence", "quality"]:
        result[task]["mask"] = np.concatenate(all_masks[task], axis=0)

    return result


def find_best_threshold(probs, labels, metric="binary"):
    """Find optimal threshold. metric='binary' for F1, 'macro' for macro-F1."""
    best_score = 0
    best_thr = 0.5
    for thr in np.arange(0.25, 0.75, 0.01):
        preds = (probs[:, 1] >= thr).astype(int)
        if metric == "binary":
            score = f1_score(labels, preds, average="binary", pos_label=1)
        else:
            score = f1_score(labels, preds, average="macro")
        if score > best_score:
            best_score = score
            best_thr = thr
    return best_thr, best_score


def compute_combined_metric(eval_result):
    """Compute competition-weighted combined metric from evaluation results."""
    # Promise: binary F1
    probs_p = eval_result["promise"]["probs"]
    labels_p = eval_result["promise"]["labels"]
    thr_p, f1_p = find_best_threshold(probs_p, labels_p, "binary")

    # Evidence: binary F1 (masked)
    mask_e = eval_result["evidence"]["mask"].astype(bool)
    f1_e = 0.0
    thr_e = 0.5
    if mask_e.sum() > 0:
        probs_e = eval_result["evidence"]["probs"][mask_e]
        labels_e = eval_result["evidence"]["labels"][mask_e]
        thr_e, f1_e = find_best_threshold(probs_e, labels_e, "binary")

    # Quality: macro F1 (masked)
    mask_q = eval_result["quality"]["mask"].astype(bool)
    f1_q = 0.0
    thr_q = 0.5
    if mask_q.sum() > 0:
        probs_q = eval_result["quality"]["probs"][mask_q]
        labels_q = eval_result["quality"]["labels"][mask_q]
        thr_q, f1_q = find_best_threshold(probs_q, labels_q, "macro")

    combined = (TASK_WEIGHTS["promise"] * f1_p +
                TASK_WEIGHTS["evidence"] * f1_e +
                TASK_WEIGHTS["quality"] * f1_q)

    return {
        "combined": combined,
        "promise_f1": f1_p, "promise_thr": thr_p,
        "evidence_f1": f1_e, "evidence_thr": thr_e,
        "quality_macro_f1": f1_q, "quality_thr": thr_q,
    }


# ─── Training loop ───────────────────────────────────────────────────────────

def train_model(model_name, texts, labels_p, labels_e, labels_q,
                masks_e, masks_q, combined_labels, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    short_name = model_name.split("/")[-1]

    print(f"\n{'='*60}")
    print(f"Training: {short_name} | Multi-task (promise + evidence + quality)")
    print(f"  FGM: ON | R-Drop: {'ON' if args.use_rdrop else 'OFF'} (alpha={args.rdrop_alpha}) | EMA: ON")
    print(f"  Multi-sample dropout (K={MULTI_SAMPLE_DROPOUT_K}) | Weighted CE | Cosine schedule")
    print(f"{'='*60}")

    # Class weights per task
    w_promise = compute_class_weights(labels_p).to(device)
    w_evidence = compute_class_weights(labels_e, masks_e).to(device)
    w_quality = compute_class_weights(labels_q, masks_q).to(device)
    print(f"  Weights — promise: {w_promise.cpu().numpy()}, "
          f"evidence: {w_evidence.cpu().numpy()}, quality: {w_quality.cpu().numpy()}")

    n_samples = len(texts)
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=SEED)

    # OOF storage
    oof_probs = {
        "promise": np.zeros((n_samples, 2)),
        "evidence": np.full((n_samples, 2), np.nan),
        "quality": np.full((n_samples, 2), np.nan),
    }

    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, combined_labels)):
        print(f"\n--- Fold {fold_idx+1}/{args.n_folds} ---")

        # Split data
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        tr_lp, tr_le, tr_lq = labels_p[train_idx], labels_e[train_idx], labels_q[train_idx]
        tr_me, tr_mq = masks_e[train_idx], masks_q[train_idx]
        va_lp, va_le, va_lq = labels_p[val_idx], labels_e[val_idx], labels_q[val_idx]
        va_me, va_mq = masks_e[val_idx], masks_q[val_idx]

        # Model + tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = MultiTaskBERT(model_name).to(device)

        train_ds = MultiTaskESGDataset(train_texts, tr_lp, tr_le, tr_lq, tr_me, tr_mq, tokenizer)
        val_ds = MultiTaskESGDataset(val_texts, va_lp, va_le, va_lq, va_me, va_mq, tokenizer)

        micro_batch = args.micro_batch
        grad_accum_steps = max(1, args.batch_size // micro_batch)

        train_loader = DataLoader(train_ds, batch_size=micro_batch, shuffle=True,
                                  drop_last=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=micro_batch * 2,
                                num_workers=2, pin_memory=True)

        # Loss functions
        ce_fns = {
            "promise": nn.CrossEntropyLoss(weight=w_promise),
            "evidence": nn.CrossEntropyLoss(weight=w_evidence),
            "quality": nn.CrossEntropyLoss(weight=w_quality),
        }

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
        optimizer_grouped = [
            {"params": [p for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.01},
            {"params": [p for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
             "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped, lr=args.lr)

        total_steps = (len(train_loader) // grad_accum_steps) * args.epochs
        warmup_steps = int(0.1 * total_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        fgm = FGM(model)
        ema = EMA(model, decay=0.999)

        best_combined = 0
        patience_counter = 0
        best_eval = None

        for epoch in range(args.epochs):
            loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, device,
                fgm, ema, ce_fns,
                use_rdrop=args.use_rdrop, rdrop_alpha=args.rdrop_alpha,
                grad_accum_steps=grad_accum_steps
            )

            # Evaluate with EMA weights
            ema.apply_shadow()
            eval_result = evaluate_model(model, val_loader, device)
            ema.restore()

            metrics = compute_combined_metric(eval_result)

            print(f"  Epoch {epoch+1}/{args.epochs} | Loss: {loss:.4f} | "
                  f"P_F1: {metrics['promise_f1']:.4f} | "
                  f"E_F1: {metrics['evidence_f1']:.4f} | "
                  f"Q_MacroF1: {metrics['quality_macro_f1']:.4f} | "
                  f"Combined: {metrics['combined']:.4f}")

            if metrics["combined"] > best_combined:
                best_combined = metrics["combined"]
                best_eval = eval_result
                patience_counter = 0

                # Save EMA weights
                ema.apply_shadow()
                save_dir = os.path.join(MODEL_DIR, short_name, f"fold{fold_idx}")
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
                tokenizer.save_pretrained(save_dir)
                model.encoder.config.save_pretrained(save_dir)
                ema.restore()

                # Save OOF probs
                oof_probs["promise"][val_idx] = eval_result["promise"]["probs"]
                ev_mask = masks_e[val_idx].astype(bool)
                oof_probs["evidence"][val_idx[ev_mask]] = eval_result["evidence"]["probs"][ev_mask]
                qu_mask = masks_q[val_idx].astype(bool)
                oof_probs["quality"][val_idx[qu_mask]] = eval_result["quality"]["probs"][qu_mask]
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        best_metrics = compute_combined_metric(best_eval)
        fold_metrics.append(best_metrics)
        print(f"  Best Combined: {best_metrics['combined']:.4f} | "
              f"P: {best_metrics['promise_f1']:.4f} (thr={best_metrics['promise_thr']:.2f}) | "
              f"E: {best_metrics['evidence_f1']:.4f} (thr={best_metrics['evidence_thr']:.2f}) | "
              f"Q: {best_metrics['quality_macro_f1']:.4f} (thr={best_metrics['quality_thr']:.2f})")

        del model, tokenizer, optimizer, scheduler, fgm, ema
        torch.cuda.empty_cache()

    # Summary
    mean_combined = np.mean([m["combined"] for m in fold_metrics])
    mean_p = np.mean([m["promise_f1"] for m in fold_metrics])
    mean_e = np.mean([m["evidence_f1"] for m in fold_metrics])
    mean_q = np.mean([m["quality_macro_f1"] for m in fold_metrics])

    print(f"\n{short_name} | Multi-task CV:")
    print(f"  Combined: {mean_combined:.4f} ± {np.std([m['combined'] for m in fold_metrics]):.4f}")
    print(f"  Promise F1:       {mean_p:.4f} ± {np.std([m['promise_f1'] for m in fold_metrics]):.4f}")
    print(f"  Evidence F1:      {mean_e:.4f} ± {np.std([m['evidence_f1'] for m in fold_metrics]):.4f}")
    print(f"  Quality MacroF1:  {mean_q:.4f} ± {np.std([m['quality_macro_f1'] for m in fold_metrics]):.4f}")

    # Save OOF probs
    model_dir = os.path.join(MODEL_DIR, short_name)
    for task in ["promise", "evidence", "quality"]:
        np.save(os.path.join(model_dir, f"oof_probs_{task}.npy"), oof_probs[task])

    # Save thresholds
    thresholds = {}
    for task_key, metric_key, thr_key in [
        ("promise", "promise_f1", "promise_thr"),
        ("evidence", "evidence_f1", "evidence_thr"),
        ("quality", "quality_macro_f1", "quality_thr"),
    ]:
        thresholds[task_key] = {
            "mean_threshold": float(np.mean([m[thr_key] for m in fold_metrics])),
            "fold_thresholds": [float(m[thr_key]) for m in fold_metrics],
            "mean_f1": float(np.mean([m[metric_key] for m in fold_metrics])),
            "fold_f1s": [float(m[metric_key]) for m in fold_metrics],
        }
    with open(os.path.join(model_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)

    # Log experiment
    try:
        from experiment_logger import log_experiment
        log_experiment(
            method="stage2_multitask",
            model=short_name,
            task="promise+evidence+quality",
            config={
                "epochs": args.epochs, "batch_size": args.batch_size,
                "micro_batch": args.micro_batch, "lr": args.lr,
                "n_folds": args.n_folds, "patience": args.patience,
                "fgm": True, "rdrop": args.use_rdrop,
                "rdrop_alpha": args.rdrop_alpha, "ema": True,
                "weighted_ce": True, "multi_sample_dropout_k": MULTI_SAMPLE_DROPOUT_K,
                "quality_classes": "Clear vs NotClear (Misleading merged)",
                "encoder": model_name,
            },
            metrics={
                "cv_combined": float(mean_combined),
                "cv_promise_f1": float(mean_p),
                "cv_evidence_f1": float(mean_e),
                "cv_quality_macro_f1": float(mean_q),
                "fold_combined": [float(m["combined"]) for m in fold_metrics],
            },
        )
    except Exception as e:
        print(f"  Warning: experiment logging failed: {e}")

    return mean_combined, fold_metrics, oof_probs


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model names to train (default: all 3)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--micro_batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use_rdrop", action="store_true", default=True)
    parser.add_argument("--no_rdrop", action="store_false", dest="use_rdrop")
    parser.add_argument("--rdrop_alpha", type=float, default=0.5)
    args = parser.parse_args()

    set_seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_list = args.models or ENCODER_MODELS

    print("Loading multi-task data...")
    (texts, labels_p, labels_e, labels_q,
     masks_e, masks_q, combined_labels) = load_multitask_data()

    print(f"Samples: {len(texts)}")
    print(f"  Promise:  {dict(zip(['No','Yes'], np.bincount(labels_p)))}")
    print(f"  Evidence: {dict(zip(['No','Yes'], np.bincount(labels_e[masks_e.astype(bool)])))} "
          f"(of {masks_e.sum():.0f} valid)")
    print(f"  Quality:  {dict(zip(['NotClear','Clear'], np.bincount(labels_q[masks_q.astype(bool)])))} "
          f"(of {masks_q.sum():.0f} valid)")

    all_results = {}
    all_oof = {}

    for model_name in model_list:
        mean_combined, fold_metrics, oof_probs = train_model(
            model_name, texts, labels_p, labels_e, labels_q,
            masks_e, masks_q, combined_labels, args
        )
        short = model_name.split("/")[-1]
        all_results[short] = {"combined": mean_combined, "fold_metrics": fold_metrics}
        all_oof[short] = oof_probs

    # Ensemble OOF evaluation
    if len(all_oof) > 1:
        print(f"\n{'='*60}")
        print("ENSEMBLE (soft-voting)")
        print(f"{'='*60}")

        for task in ["promise", "evidence", "quality"]:
            oof_list = [all_oof[m][task] for m in all_oof]
            # Average, handling NaN (only valid samples)
            stacked = np.stack(oof_list, axis=0)
            ensemble_probs = np.nanmean(stacked, axis=0)

            if task == "promise":
                labels = labels_p
                mask = np.ones(len(labels), dtype=bool)
                metric_type = "binary"
            elif task == "evidence":
                labels = labels_e
                mask = masks_e.astype(bool)
                metric_type = "binary"
            else:
                labels = labels_q
                mask = masks_q.astype(bool)
                metric_type = "macro"

            valid_probs = ensemble_probs[mask]
            valid_labels = labels[mask]
            thr, f1 = find_best_threshold(valid_probs, valid_labels, metric_type)
            preds = (valid_probs[:, 1] >= thr).astype(int)

            print(f"\n  {task}: F1={f1:.4f} (thr={thr:.2f})")
            label_names = ["No", "Yes"] if task != "quality" else ["Not Clear", "Clear"]
            print(classification_report(valid_labels, preds,
                                        target_names=label_names, digits=4))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, res in all_results.items():
        fm = res["fold_metrics"]
        print(f"\n{name}:")
        print(f"  Combined:        {res['combined']:.4f}")
        print(f"  Promise F1:      {np.mean([m['promise_f1'] for m in fm]):.4f}")
        print(f"  Evidence F1:     {np.mean([m['evidence_f1'] for m in fm]):.4f}")
        print(f"  Quality MacroF1: {np.mean([m['quality_macro_f1'] for m in fm]):.4f}")


if __name__ == "__main__":
    main()
