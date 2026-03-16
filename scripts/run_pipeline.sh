#!/usr/bin/env bash
set -euo pipefail

# run_pipeline.sh
#
# Convenience wrapper for a single-protein end-to-end run:
# prepare metadata -> cluster -> embeddings -> feature table -> frozen baseline
#
# Example:
# bash scripts/run_pipeline.sh \
#   --phenotypes data/example/phenotypes.tsv \
#   --protein-fasta data/example/oprD.faa \
#   --protein-name oprD \
#   --outdir results/oprD_demo

PHENOTYPES=""
PROTEIN_FASTA=""
PROTEIN_NAME=""
OUTDIR=""
MODEL_NAME="facebook/esm2_t12_35M_UR50D"
BATCH_SIZE="8"
N_SPLITS="5"
N_JOBS="4"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --phenotypes) PHENOTYPES="$2"; shift 2 ;;
    --protein-fasta) PROTEIN_FASTA="$2"; shift 2 ;;
    --protein-name) PROTEIN_NAME="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --n-splits) N_SPLITS="$2"; shift 2 ;;
    --n-jobs) N_JOBS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${PHENOTYPES}" || -z "${PROTEIN_FASTA}" || -z "${PROTEIN_NAME}" || -z "${OUTDIR}" ]]; then
  echo "Missing required arguments." >&2
  exit 1
fi

mkdir -p "${OUTDIR}"/{metadata,clusters,embeddings,features,baseline,plots,tmp}

python scripts/01_prepare_metadata.py \
  --protein-fasta "${PROTEIN_FASTA}" \
  --protein-name "${PROTEIN_NAME}" \
  --phenotypes "${PHENOTYPES}" \
  --outdir "${OUTDIR}/metadata"

python scripts/02_cluster_sequences_mmseqs.py \
  --protein-fasta "${PROTEIN_FASTA}" \
  --protein-name "${PROTEIN_NAME}" \
  --min-seq-id 0.90 \
  --coverage 0.80 \
  --tmp-dir "${OUTDIR}/tmp/mmseqs" \
  --outdir "${OUTDIR}/clusters"

python scripts/03_extract_embeddings.py \
  --protein-fasta "${PROTEIN_FASTA}" \
  --protein-name "${PROTEIN_NAME}" \
  --model-name "${MODEL_NAME}" \
  --batch-size "${BATCH_SIZE}" \
  --outdir "${OUTDIR}/embeddings"

python scripts/04_build_feature_table.py \
  --embedding-tsv "${OUTDIR}/embeddings/${PROTEIN_NAME}_embeddings.tsv" \
  --protein-name "${PROTEIN_NAME}" \
  --cluster-tsv "${OUTDIR}/clusters/${PROTEIN_NAME}_clusters.tsv" \
  --phenotypes "${PHENOTYPES}" \
  --combine-method concat \
  --out-prefix "${OUTDIR}/features/${PROTEIN_NAME}"

python scripts/05_train_frozen_baseline.py \
  --features "${OUTDIR}/features/${PROTEIN_NAME}_features.tsv" \
  --label-col phenotype_binary \
  --group-col group_id \
  --models logreg rf mlp \
  --n-splits "${N_SPLITS}" \
  --n-jobs "${N_JOBS}" \
  --outdir "${OUTDIR}/baseline"

python scripts/08_plot_benchmarks.py \
  --summary-tsv "${OUTDIR}/baseline/baseline_metrics_summary.tsv" \
  --fold-tsv "${OUTDIR}/baseline/baseline_metrics_by_fold.tsv" \
  --features-tsv "${OUTDIR}/features/${PROTEIN_NAME}_features.tsv" \
  --outdir "${OUTDIR}/plots"

echo "Pipeline completed: ${OUTDIR}"
