#!/usr/bin/env python
"""
05_train_frozen_baseline.py

Train classical supervised models on frozen ESM-2 embeddings using grouped CV.

Models:
- logistic regression
- random forest
- xgboost (optional)
- shallow MLP

Outputs:
- fold predictions
- fold metrics
- summary metrics
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import compute_binary_metrics, embedding_columns, ensure_dir

try:
    from xgboost import XGBClassifier
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--label-col", default="phenotype_binary")
    p.add_argument("--group-col", default="group_id")
    p.add_argument("--models", nargs="+", default=["logreg", "rf", "mlp"])
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def make_model(name: str, random_state: int):
    if name == "logreg":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=random_state)),
            ]
        )
    if name == "rf":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(
                    n_estimators=500,
                    random_state=random_state,
                    n_jobs=1,
                    class_weight="balanced_subsample",
                )),
            ]
        )
    if name == "mlp":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(256, 64),
                    activation="relu",
                    alpha=1e-4,
                    early_stopping=True,
                    max_iter=300,
                    random_state=random_state,
                )),
            ]
        )
    if name == "xgb":
        if not HAVE_XGB:
            raise ValueError("xgboost not installed.")
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("clf", XGBClassifier(
                    n_estimators=500,
                    max_depth=5,
                    learning_rate=0.03,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=1,
                )),
            ]
        )
    raise ValueError(f"Unsupported model: {name}")


def run_one_model(
    model_name: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    isolate_ids: np.ndarray,
    n_splits: int,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pred_rows = []
    metric_rows = []

    for fold, (tr, te) in enumerate(cv.split(X, y, groups), start=1):
        model = make_model(model_name, random_state + fold)
        model.fit(X[tr], y[tr])

        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X[te])[:, 1]
        else:
            score = model.decision_function(X[te])

        metrics = compute_binary_metrics(y[te], score)
        metrics.update({"model_name": model_name, "fold": fold, "n_test": len(te)})
        metric_rows.append(metrics)

        pred_rows.append(
            pd.DataFrame(
                {
                    "isolate_id": isolate_ids[te],
                    "y_true": y[te],
                    "y_score": score,
                    "model_name": model_name,
                    "fold": fold,
                }
            )
        )

    return pd.concat(pred_rows, axis=0), pd.DataFrame(metric_rows)


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    df = pd.read_csv(args.features, sep="\t")
    feat_cols = embedding_columns(df)
    if not feat_cols:
        feat_cols = [c for c in df.columns if "__emb_" in c]
    if not feat_cols:
        raise ValueError("No embedding columns found.")

    X = df[feat_cols].to_numpy(dtype=float)
    y = df[args.label_col].to_numpy(dtype=int)
    groups = df[args.group_col].astype(str).to_numpy()
    isolate_ids = df["isolate_id"].astype(str).to_numpy()

    results = Parallel(n_jobs=args.n_jobs)(
        delayed(run_one_model)(
            model_name=m,
            X=X,
            y=y,
            groups=groups,
            isolate_ids=isolate_ids,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        for m in args.models
    )

    preds = pd.concat([r[0] for r in results], axis=0)
    fold_metrics = pd.concat([r[1] for r in results], axis=0)
    summary = fold_metrics.groupby("model_name")[["auroc", "auprc", "f1", "balanced_accuracy", "mcc"]].agg(["mean", "std"])
    summary.columns = ["__".join(col) for col in summary.columns]
    summary = summary.reset_index()

    preds.to_csv(outdir / "baseline_predictions_by_fold.tsv", sep="\t", index=False)
    fold_metrics.to_csv(outdir / "baseline_metrics_by_fold.tsv", sep="\t", index=False)
    summary.to_csv(outdir / "baseline_metrics_summary.tsv", sep="\t", index=False)

    print(f"Wrote results to: {outdir}")


if __name__ == "__main__":
    main()
