"""
DistilBERT Track Classifier — Fine-tuning Script (proposal §5.5, §12.6).

Fine-tunes distilbert-base-uncased on the 5 BMW internship tracks:
  AI, Backend, Frontend, Robotics, Simulation

Training data:
  - data/raw/bert_train_extra.jsonl    (100 samples, 20 per track)
  - data/raw/candidate_profiles_syn.jsonl (33 samples with bmw_track_label)

Output:
  models/track_classifier/        — saved model + tokenizer
  models/track_classifier/label_map.json
  evaluation/results/classifier_report.json — accuracy, F1, confusion matrix

Usage:
  python scripts/train_classifier.py
  python scripts/train_classifier.py --epochs 5 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models" / "track_classifier"
RESULTS_DIR = ROOT / "evaluation" / "results"

LABELS = ["AI", "Backend", "Frontend", "Robotics", "Simulation"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data() -> list[dict]:
    """Load and combine BERT training data + candidate profiles."""
    samples: list[dict] = []

    bert_path = DATA_DIR / "bert_train_extra.jsonl"
    if bert_path.exists():
        with open(bert_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                samples.append({"text": obj["text"], "label": obj["label"]})
        print(f"  Loaded {len(samples)} samples from bert_train_extra.jsonl")

    cand_path = DATA_DIR / "candidate_profiles_syn.jsonl"
    n_cand = 0
    if cand_path.exists():
        with open(cand_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                label = obj.get("bmw_track_label", "")
                if label not in LABELS:
                    continue
                text = (
                    f"{obj.get('summary', '')} "
                    f"Skills: {', '.join(obj.get('skills', []))}. "
                    f"{obj.get('raw_resume_snippet', '')}"
                )
                samples.append({"text": text.strip(), "label": label})
                n_cand += 1
        print(f"  Loaded {n_cand} samples from candidate_profiles_syn.jsonl")

    print(f"  Total: {len(samples)} samples across {len(LABELS)} classes")

    label_counts = {}
    for s in samples:
        label_counts[s["label"]] = label_counts.get(s["label"], 0) + 1
    for label in LABELS:
        print(f"    {label}: {label_counts.get(label, 0)}")

    return samples


class TrackDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int = 256):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def train_epoch(model, dataloader, optimizer, scheduler, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)


def evaluate_model(model, dataloader, device) -> tuple[list[int], list[int]]:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return all_preds, all_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DistilBERT track classifier (§5.5)")
    parser.add_argument("--epochs", type=int, default=8, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=256, help="Max token length")
    parser.add_argument("--folds", type=int, default=5, help="K-fold cross-validation folds")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\n[1/4] Loading data...")
    samples = load_data()
    texts = [s["text"] for s in samples]
    labels = [LABEL2ID[s["label"]] for s in samples]
    label_names = [s["label"] for s in samples]

    print(f"\n[2/4] Loading DistilBERT tokenizer and model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    print(f"\n[3/4] {args.folds}-fold cross-validation (epochs={args.epochs}, lr={args.lr})...")
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)

    all_fold_preds = np.zeros(len(samples), dtype=int)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels), 1):
        print(f"\n  --- Fold {fold}/{args.folds} ---")
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]

        train_ds = TrackDataset(train_texts, train_labels, tokenizer, args.max_length)
        val_ds = TrackDataset(val_texts, val_labels, tokenizer, args.max_length)

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size)

        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=len(LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, train_loader, optimizer, scheduler, device)
            if epoch % 2 == 0 or epoch == args.epochs:
                preds, true = evaluate_model(model, val_loader, device)
                acc = accuracy_score(true, preds)
                print(f"    Epoch {epoch}/{args.epochs} — loss: {loss:.4f}, val_acc: {acc:.4f}")

        preds, true = evaluate_model(model, val_loader, device)
        fold_acc = accuracy_score(true, preds)
        fold_accuracies.append(fold_acc)

        for i, idx in enumerate(val_idx):
            all_fold_preds[idx] = preds[i]

        print(f"  Fold {fold} accuracy: {fold_acc:.4f}")

    print(f"\n  Mean CV accuracy: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")

    print(f"\n[4/4] Training final model on ALL data and saving...")
    full_ds = TrackDataset(texts, labels, tokenizer, args.max_length)
    full_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=True)

    final_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    optimizer = torch.optim.AdamW(final_model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(full_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(final_model, full_loader, optimizer, scheduler, device)
        if epoch % 2 == 0 or epoch == args.epochs:
            print(f"    Epoch {epoch}/{args.epochs} — loss: {loss:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    final_model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    with open(MODEL_DIR / "label_map.json", "w", encoding="utf-8") as f:
        json.dump({"labels": LABELS, "label2id": LABEL2ID, "id2label": ID2LABEL}, f, indent=2)

    print(f"  Model saved to {MODEL_DIR}")

    cv_accuracy = float(np.mean(fold_accuracies))
    cv_preds = all_fold_preds.tolist()
    cv_true = labels

    report_dict = classification_report(
        cv_true, cv_preds, target_names=LABELS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(cv_true, cv_preds).tolist()
    macro_f1 = f1_score(cv_true, cv_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(cv_true, cv_preds, average="weighted", zero_division=0)

    print(f"\n{'='*60}")
    print("DISTILBERT CLASSIFIER — CROSS-VALIDATION RESULTS (§12.6)")
    print(f"{'='*60}")
    print(f"  Accuracy (CV mean): {cv_accuracy:.4f}")
    print(f"  Macro F1:           {macro_f1:.4f}")
    print(f"  Weighted F1:        {weighted_f1:.4f}")
    print(f"\n  Per-class report:")
    print(classification_report(cv_true, cv_preds, target_names=LABELS, zero_division=0))
    print(f"  Confusion Matrix (rows=true, cols=pred):")
    print(f"  Labels: {LABELS}")
    for row in cm:
        print(f"    {row}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "model": "distilbert-base-uncased",
        "num_classes": len(LABELS),
        "labels": LABELS,
        "training_samples": len(samples),
        "cv_folds": args.folds,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "cv_accuracy": round(cv_accuracy, 4),
        "cv_accuracy_std": round(float(np.std(fold_accuracies)), 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": report_dict,
        "confusion_matrix": cm,
        "fold_accuracies": [round(a, 4) for a in fold_accuracies],
    }

    with open(RESULTS_DIR / "classifier_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Report saved to {RESULTS_DIR / 'classifier_report.json'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
