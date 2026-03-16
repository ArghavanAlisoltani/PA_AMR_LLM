#!/usr/bin/env python
"""
01_prepare_metadata.py

Read a protein FASTA and a phenotype table, normalize isolate IDs,
merge both, and write a clean metadata table for downstream steps.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

from utils import ensure_dir, read_fasta_as_df, read_phenotypes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--protein-fasta", required=True, help="Input FASTA for one protein.")
    p.add_argument("--protein-name", required=True, help="Protein name, e.g. oprD.")
    p.add_argument("--phenotypes", required=True, help="TSV with isolate_id and phenotype.")
    p.add_argument("--outdir", required=True, help="Output directory.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    seq_df = read_fasta_as_df(args.protein_fasta, args.protein_name)
    pheno_df = read_phenotypes(args.phenotypes)

    merged = seq_df.merge(pheno_df, on="isolate_id", how="left", validate="one_to_one")
    merged["has_phenotype"] = ~merged["phenotype"].isna()

    merged.to_csv(outdir / f"{args.protein_name}_metadata.tsv", sep="\t", index=False)

    summary = pd.DataFrame(
        {
            "protein_name": [args.protein_name],
            "n_sequences": [len(seq_df)],
            "n_with_phenotype": [merged["has_phenotype"].sum()],
            "n_without_phenotype": [(~merged["has_phenotype"]).sum()],
            "median_length": [float(seq_df["seq_len"].median())],
        }
    )
    summary.to_csv(outdir / f"{args.protein_name}_summary.tsv", sep="\t", index=False)

    print(f"Wrote metadata to: {outdir}")


if __name__ == "__main__":
    main()
