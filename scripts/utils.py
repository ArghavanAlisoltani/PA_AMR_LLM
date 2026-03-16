#!/usr/bin/env python
"""
Shared utilities for the AMR protein LLM pipeline.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from Bio import SeqIO
from sklearn.metrics import (
    auc,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

PHENO_POSITIVE = {"R", "RESISTANT", "1", 1, True}
PHENO_NEGATIVE = {"S", "SUSCEPTIBLE", "0", 0, False}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_isolate_id(header: str) -> str:
    """
    Extract a simple isolate identifier from a FASTA header.

    Supported examples:
    - iso_001
    - iso_001|oprD
    - iso_001 oprD something

    Rule:
    take the first token before whitespace, then split on '|'.
    """
    token = header.strip().split()[0]
    token = token.split("|")[0]
    return token


def sanitize_sequence(seq: str) -> str:
    """
    Keep only standard amino-acid letters that ESM-style tokenizers expect.
    Unknown residues are replaced with X.
    """
    seq = seq.strip().upper()
    allowed = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")
    return "".join([aa if aa in allowed else "X" for aa in seq])


def read_fasta_as_df(fasta_path: str | Path, protein_name: str) -> pd.DataFrame:
    rows = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        isolate_id = normalize_isolate_id(record.description)
        seq = sanitize_sequence(str(record.seq))
        rows.append(
            {
                "protein_name": protein_name,
                "fasta_path": str(fasta_path),
                "header": record.description,
                "isolate_id": isolate_id,
                "sequence": seq,
                "seq_len": len(seq),
            }
        )
    if not rows:
        raise ValueError(f"No sequences found in {fasta_path}")
    df = pd.DataFrame(rows)
    # Keep first sequence per isolate by default.
    df = df.sort_values(["isolate_id", "seq_len"], ascending=[True, False]).drop_duplicates(
        subset=["isolate_id"], keep="first"
    )
    return df


def read_phenotypes(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "isolate_id" not in df.columns or "phenotype" not in df.columns:
        raise ValueError("Phenotype TSV must contain isolate_id and phenotype columns.")
    out = df.copy()
    out["phenotype_str"] = out["phenotype"].astype(str).str.upper().str.strip()
    out["phenotype_binary"] = out["phenotype_str"].map(
        lambda x: 1 if x in {str(v).upper() for v in PHENO_POSITIVE} else 0
    )
    return out


def composite_group_id(df: pd.DataFrame, cluster_cols: Sequence[str]) -> pd.Series:
    return df[list(cluster_cols)].fillna("NA").astype(str).agg("|".join, axis=1)


def compute_binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    metrics = {
        "auroc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan,
        "auprc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
    }
    return metrics


def roc_pr_curves(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return {"roc": (fpr, tpr), "pr": (recall, precision)}


def save_json(obj: dict, path: str | Path) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)


def load_embedding_table(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def embedding_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("emb_")]


def fit_pca_projection(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    X_centered = X - X.mean(axis=0, keepdims=True)
    U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
    coords = U[:, :n_components] * S[:n_components]
    return coords, VT[:n_components]


def safe_name(items: Sequence[str]) -> str:
    return "__".join(items).replace("/", "_")


def summarize_trainable_parameters(model) -> Tuple[int, int, float]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    pct = 100.0 * trainable / total if total > 0 else 0.0
    return trainable, total, pct
