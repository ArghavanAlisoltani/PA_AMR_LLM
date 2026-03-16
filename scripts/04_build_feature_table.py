#!/usr/bin/env python
"""
04_build_feature_table.py

Combine one or more embedding tables into a single feature matrix for phenotype prediction.

Supports:
- single proteins
- multi-protein combinations

Combine methods:
- concat : concatenate embedding vectors
- mean   : mean of matching-dimension embeddings across proteins

For grouped CV:
- also merges per-protein cluster TSV files
- creates a composite group_id across all supplied proteins
"""

from __future__ import annotations
import argparse
from pathlib import Path
from functools import reduce

import numpy as np
import pandas as pd

from utils import composite_group_id, embedding_columns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--embedding-tsv", action="append", required=True, help="May be repeated.")
    p.add_argument("--protein-name", action="append", required=True, help="Must match embedding-tsv order.")
    p.add_argument("--cluster-tsv", action="append", default=[], help="Optional, may be repeated.")
    p.add_argument("--phenotypes", required=True)
    p.add_argument("--combine-method", choices=["concat", "mean"], default="concat")
    p.add_argument("--out-prefix", required=True)
    return p.parse_args()


def load_embeddings(path: str, protein_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    emb_cols = embedding_columns(df)
    keep = ["isolate_id"] + emb_cols
    out = df[keep].copy()
    out = out.rename(columns={c: f"{protein_name}__{c}" for c in emb_cols})
    return out


def load_clusters(path: str, protein_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if "cluster_id" not in df.columns or "isolate_id" not in df.columns:
        raise ValueError(f"Cluster file missing required columns: {path}")
    return df[["isolate_id", "cluster_id"]].rename(columns={"cluster_id": f"{protein_name}__cluster_id"})


def main() -> None:
    args = parse_args()
    if len(args.embedding_tsv) != len(args.protein_name):
        raise ValueError("Number of --embedding-tsv and --protein-name entries must match.")
    if args.cluster_tsv and len(args.cluster_tsv) != len(args.protein_name):
        raise ValueError("If cluster files are provided, one --cluster-tsv is needed per protein.")

    phen = pd.read_csv(args.phenotypes, sep="\t")
    phen["phenotype_str"] = phen["phenotype"].astype(str).str.upper().str.strip()
    phen["phenotype_binary"] = phen["phenotype_str"].map(lambda x: 1 if x in {"R", "RESISTANT", "1"} else 0)

    dfs = []
    for emb_path, prot in zip(args.embedding_tsv, args.protein_name):
        dfs.append(load_embeddings(emb_path, prot))
    merged = reduce(lambda left, right: pd.merge(left, right, on="isolate_id", how="outer"), dfs)
    merged = phen.merge(merged, on="isolate_id", how="inner")

    cluster_cols = []
    if args.cluster_tsv:
        cluster_dfs = []
        for cpath, prot in zip(args.cluster_tsv, args.protein_name):
            cdf = load_clusters(cpath, prot)
            cluster_dfs.append(cdf)
            cluster_cols.append(f"{prot}__cluster_id")
        cmerged = reduce(lambda left, right: pd.merge(left, right, on="isolate_id", how="outer"), cluster_dfs)
        merged = merged.merge(cmerged, on="isolate_id", how="left")
        merged["group_id"] = composite_group_id(merged, cluster_cols)
    else:
        merged["group_id"] = merged["isolate_id"]

    if args.combine_method == "mean" and len(args.protein_name) > 1:
        # only valid if all embedding dimensions match, which is true if they used the same model
        per_prot_cols = []
        for prot in args.protein_name:
            cols = [c for c in merged.columns if c.startswith(f"{prot}__emb_")]
            per_prot_cols.append(cols)
        emb_dim = len(per_prot_cols[0])
        if not all(len(c) == emb_dim for c in per_prot_cols):
            raise ValueError("Mean combine requires equal embedding dimensions across proteins.")
        feat = pd.DataFrame(index=merged.index)
        for i in range(emb_dim):
            vals = np.column_stack([merged[cols[i]].values for cols in per_prot_cols])
            feat[f"emb_{i:04d}"] = np.nanmean(vals, axis=1)
        base_cols = ["isolate_id", "phenotype", "phenotype_str", "phenotype_binary", "group_id"]
        merged = pd.concat([merged[base_cols], feat], axis=1)

    combo_name = "__".join(args.protein_name)
    out_path = Path(f"{args.out_prefix}_features.tsv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, sep="\t", index=False)

    manifest = pd.DataFrame(
        {
            "combo_name": [combo_name],
            "n_isolates": [len(merged)],
            "n_positive": [int(merged["phenotype_binary"].sum())],
            "n_negative": [int((1 - merged["phenotype_binary"]).sum())],
            "combine_method": [args.combine_method],
        }
    )
    manifest.to_csv(Path(f"{args.out_prefix}_manifest.tsv"), sep="\t", index=False)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
