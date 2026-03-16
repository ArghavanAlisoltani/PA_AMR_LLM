#!/usr/bin/env python
"""
02_cluster_sequences_mmseqs.py

Cluster sequences with MMseqs2 to create grouping variables for grouped CV.

This script:
1. converts the FASTA into an MMseqs DB
2. runs easy-cluster
3. emits a TSV with isolate_id -> cluster_id

Requires:
- mmseqs on PATH

If MMseqs2 is unavailable, you can substitute CD-HIT externally and adapt the cluster TSV format.
"""

from __future__ import annotations
import argparse
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from Bio import SeqIO

from utils import ensure_dir, normalize_isolate_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--protein-fasta", required=True)
    p.add_argument("--protein-name", required=True)
    p.add_argument("--min-seq-id", type=float, default=0.9, help="MMseqs2 identity threshold.")
    p.add_argument("--cov-mode", type=int, default=0)
    p.add_argument("--coverage", type=float, default=0.8)
    p.add_argument("--tmp-dir", required=True)
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    tmpdir = ensure_dir(args.tmp_dir)

    if shutil.which("mmseqs") is None:
        raise SystemExit("mmseqs not found on PATH. Install MMseqs2 first.")

    cluster_prefix = outdir / f"{args.protein_name}_mmseqs"

    cmd = [
        "mmseqs", "easy-cluster",
        args.protein_fasta,
        str(cluster_prefix),
        str(tmpdir),
        "--min-seq-id", str(args.min_seq_id),
        "-c", str(args.coverage),
        "--cov-mode", str(args.cov_mode),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    rep_fasta = outdir / f"{args.protein_name}_mmseqs_rep_seq.fasta"
    all_seqs = outdir / f"{args.protein_name}_mmseqs_all_seqs.fasta"
    cluster_tsv = outdir / f"{args.protein_name}_mmseqs_cluster.tsv"

    if not cluster_tsv.exists():
        raise FileNotFoundError(f"Expected cluster output not found: {cluster_tsv}")

    mm = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["cluster_rep", "member_header"])
    rep_to_cluster = {rep: f"{args.protein_name}_cluster_{i:05d}" for i, rep in enumerate(mm["cluster_rep"].unique(), start=1)}
    mm["cluster_id"] = mm["cluster_rep"].map(rep_to_cluster)
    mm["isolate_id"] = mm["member_header"].map(normalize_isolate_id)
    mm["protein_name"] = args.protein_name

    out = mm[["protein_name", "isolate_id", "cluster_id", "cluster_rep", "member_header"]].drop_duplicates()
    out.to_csv(outdir / f"{args.protein_name}_clusters.tsv", sep="\t", index=False)
    print(f"Wrote: {outdir / f'{args.protein_name}_clusters.tsv'}")


if __name__ == "__main__":
    main()
