#!/usr/bin/env python
"""
03_extract_embeddings.py

Extract ESM-2 embeddings for one protein FASTA file.

Default model:
- facebook/esm2_t12_35M_UR50D

Outputs:
- per-isolate embedding TSV
- metadata TSV

Notes:
- Uses mean pooling over amino-acid token embeddings, excluding BOS/EOS tokens.
- Supports CPU or GPU.
- Batching is the main source of acceleration.

"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from utils import ensure_dir, read_fasta_as_df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--protein-fasta", required=True)
    p.add_argument("--protein-name", required=True)
    p.add_argument("--model-name", default="facebook/esm2_t12_35M_UR50D")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def mean_pool_hidden(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Pool token embeddings with attention-mask-aware mean pooling.
    """
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    masked = hidden * mask
    summed = masked.sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.outdir)

    df = read_fasta_as_df(args.protein_fasta, args.protein_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)
    model.eval()
    model.to(args.device)

    all_rows = []
    with torch.no_grad():
        for start in tqdm(range(0, len(df), args.batch_size), desc="embedding batches"):
            sub = df.iloc[start:start + args.batch_size]
            seqs = sub["sequence"].tolist()

            enc = tokenizer(
                seqs,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(args.device) for k, v in enc.items()}

            outputs = model(**enc)
            pooled = mean_pool_hidden(outputs.last_hidden_state, enc["attention_mask"])
            pooled = pooled.detach().cpu().numpy()

            for row_dict, emb in zip(sub.to_dict(orient="records"), pooled):
                out = {**row_dict}
                for i, val in enumerate(emb):
                    out[f"emb_{i:04d}"] = float(val)
                all_rows.append(out)

    emb_df = pd.DataFrame(all_rows)
    emb_df.to_csv(outdir / f"{args.protein_name}_embeddings.tsv", sep="\t", index=False)

    meta = pd.DataFrame(
        {
            "protein_name": [args.protein_name],
            "model_name": [args.model_name],
            "n_sequences": [len(emb_df)],
            "embedding_dim": [len([c for c in emb_df.columns if c.startswith('emb_')])],
            "device": [args.device],
        }
    )
    meta.to_csv(outdir / f"{args.protein_name}_embedding_run.tsv", sep="\t", index=False)

    print(f"Wrote embeddings to: {outdir / f'{args.protein_name}_embeddings.tsv'}")


if __name__ == "__main__":
    main()
