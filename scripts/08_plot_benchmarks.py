#!/usr/bin/env python
"""
08_plot_benchmarks.py

Create publication-style benchmark plots from summary/fold prediction tables.

Outputs:
- benchmark_barplot.png/pdf
- cv_boxplot.png/pdf
- optional ROC / PR / PCA plots
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import ensure_dir, fit_pca_projection


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summary-tsv", required=True)
    p.add_argument("--fold-tsv", required=True)
    p.add_argument("--prediction-tsv", default=None)
    p.add_argument("--features-tsv", default=None)
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def savefig_all(fig, out_base: Path):
    fig.tight_layout()
    fig.savefig(str(out_base) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    summary = pd.read_csv(args.summary_tsv, sep="\t")
    folds = pd.read_csv(args.fold_tsv, sep="\t")

    # Bar plot on AUROC and AUPRC
    for metric in ["auroc", "auprc"]:
        mean_col = f"{metric}__mean" if f"{metric}__mean" in summary.columns else metric
        std_col = f"{metric}__std" if f"{metric}__std" in summary.columns else None

        fig = plt.figure(figsize=(10, 5))
        x = np.arange(len(summary))
        y = summary[mean_col].to_numpy()
        err = summary[std_col].to_numpy() if std_col and std_col in summary.columns else None
        labels = [
            f"{r.get('experiment_name', '')}\n{r['model_name']}".strip()
            for _, r in summary.iterrows()
        ]
        plt.bar(x, y, yerr=err, capsize=4)
        plt.xticks(x, labels, rotation=45, ha="right")
        plt.ylabel(metric.upper())
        plt.title(f"Benchmark comparison: {metric.upper()}")
        savefig_all(fig, outdir / f"benchmark_barplot_{metric}")

    # Boxplots by fold
    for metric in ["auroc", "auprc", "f1", "balanced_accuracy", "mcc"]:
        if metric not in folds.columns:
            continue
        fig = plt.figure(figsize=(10, 5))
        order = []
        arrays = []
        for (exp, model), sub in folds.groupby(["experiment_name", "model_name"], dropna=False):
            order.append(f"{exp}\n{model}".strip())
            arrays.append(sub[metric].values)
        plt.boxplot(arrays, labels=order, showfliers=True)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel(metric)
        plt.title(f"{metric} by CV fold")
        savefig_all(fig, outdir / f"cv_boxplot_{metric}")

    # Optional PCA plot of features
    if args.features_tsv:
        feat = pd.read_csv(args.features_tsv, sep="\t")
        feat_cols = [c for c in feat.columns if c.startswith("emb_") or "__emb_" in c]
        if len(feat_cols) >= 2 and "phenotype_binary" in feat.columns:
            X = feat[feat_cols].to_numpy(dtype=float)
            coords, _ = fit_pca_projection(X, n_components=2)
            fig = plt.figure(figsize=(6, 5))
            plt.scatter(coords[:, 0], coords[:, 1], c=feat["phenotype_binary"].values, alpha=0.8)
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("Embedding PCA")
            savefig_all(fig, outdir / "embedding_pca")

    print(f"Wrote plots to: {outdir}")


if __name__ == "__main__":
    main()
