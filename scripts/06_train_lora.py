#!/usr/bin/env python
"""
06_train_lora.py

LoRA fine-tuning of an ESM-2 classifier for a single protein sequence task.

This script uses:
- Hugging Face Transformers
- PEFT LoRA
- grouped CV by sequence clusters

Important:
This script operates on a metadata table containing:
- isolate_id
- sequence
- phenotype_binary
- group_id

Typical workflow:
1. run 01_prepare_metadata.py
2. merge in cluster IDs from 02_cluster_sequences_mmseqs.py
3. run this script

Output:
- fold metrics
- fold predictions
- trainable parameter summary
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, TaskType, get_peft_model

from utils import compute_binary_metrics, ensure_dir, summarize_trainable_parameters


class ProteinDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["sequence"],
            truncation=True,
            max_length=self.max_length,
        )
        enc["labels"] = int(row["phenotype_binary"])
        enc["isolate_id"] = row["isolate_id"]
        return enc


class KeepOnlyTensorCollator:
    """
    Wrap the HF padding collator, but retain isolate IDs outside the tensor batch.
    """
    def __init__(self, tokenizer):
        self.inner = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    def __call__(self, features):
        isolate_ids = [f.pop("isolate_id") for f in features]
        batch = self.inner(features)
        batch["isolate_id"] = isolate_ids
        return batch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True, help="Metadata TSV with sequence and phenotype_binary.")
    p.add_argument("--model-name", default="facebook/esm2_t12_35M_UR50D")
    p.add_argument("--group-col", default="group_id")
    p.add_argument("--sequence-col", default="sequence")
    p.add_argument("--label-col", default="phenotype_binary")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def build_model(model_name: str, lora_r: int, lora_alpha: int, lora_dropout: float):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="single_label_classification",
    )
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["query", "key", "value", "dense"],
        bias="none",
    )
    model = get_peft_model(model, config)
    return model


def make_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = np.asarray(logits)
        probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        return compute_binary_metrics(np.asarray(labels), probs)
    return compute_metrics


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    df = pd.read_csv(args.metadata, sep="\t")

    required = {"isolate_id", args.sequence_col, args.label_col, args.group_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {sorted(missing)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    cv = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    pred_rows = []
    metric_rows = []
    trainable_summary = []

    for fold, (tr, te) in enumerate(cv.split(df, df[args.label_col], groups=df[args.group_col]), start=1):
        train_df = df.iloc[tr].copy()
        test_df = df.iloc[te].copy()

        model = build_model(args.model_name, args.lora_r, args.lora_alpha, args.lora_dropout)
        trainable, total, pct = summarize_trainable_parameters(model)
        trainable_summary.append({"fold": fold, "trainable_params": trainable, "total_params": total, "trainable_pct": pct})

        train_ds = ProteinDataset(train_df, tokenizer, args.max_length)
        test_ds = ProteinDataset(test_df, tokenizer, args.max_length)

        fold_dir = ensure_dir(outdir / f"fold_{fold}")
        train_args = TrainingArguments(
            output_dir=str(fold_dir / "hf_output"),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="auroc",
            greater_is_better=True,
            logging_steps=10,
            save_total_limit=1,
            report_to=[],
            fp16=args.fp16,
            bf16=args.bf16,
            seed=args.random_state + fold,
        )

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            tokenizer=tokenizer,
            data_collator=KeepOnlyTensorCollator(tokenizer),
            compute_metrics=make_compute_metrics(),
        )
        trainer.train()

        preds = trainer.predict(test_ds)
        probs = torch.softmax(torch.tensor(preds.predictions), dim=1)[:, 1].numpy()
        labels = preds.label_ids
        metrics = compute_binary_metrics(labels, probs)
        metrics.update({"model_name": "esm2_lora", "fold": fold, "n_test": len(test_df)})
        metric_rows.append(metrics)

        fold_pred_df = pd.DataFrame(
            {
                "isolate_id": test_df["isolate_id"].values,
                "y_true": labels,
                "y_score": probs,
                "model_name": "esm2_lora",
                "fold": fold,
            }
        )
        pred_rows.append(fold_pred_df)

    pred_df = pd.concat(pred_rows, axis=0)
    metric_df = pd.DataFrame(metric_rows)
    summary = metric_df.groupby("model_name")[["auroc", "auprc", "f1", "balanced_accuracy", "mcc"]].agg(["mean", "std"])
    summary.columns = ["__".join(col) for col in summary.columns]
    summary = summary.reset_index()
    trainable_df = pd.DataFrame(trainable_summary)

    pred_df.to_csv(outdir / "lora_predictions_by_fold.tsv", sep="\t", index=False)
    metric_df.to_csv(outdir / "lora_metrics_by_fold.tsv", sep="\t", index=False)
    summary.to_csv(outdir / "lora_metrics_summary.tsv", sep="\t", index=False)
    trainable_df.to_csv(outdir / "lora_trainable_parameter_summary.tsv", sep="\t", index=False)

    print(f"Wrote LoRA results to: {outdir}")


if __name__ == "__main__":
    main()
