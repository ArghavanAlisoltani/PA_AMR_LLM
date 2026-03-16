# AMR Protein LLM Pipeline

This repository provides an end-to-end pipeline for **protein-level AMR phenotype prediction**
using **ESM-2 embeddings**, a **frozen baseline**, **LoRA fine-tuning**, **grouped cross-validation
by sequence cluster**, and **publication-quality benchmark figures**.

It is designed for the case where you have:

- one or more **target proteins** collected from many isolates
- a phenotype table such as **resistant / susceptible**
- a need to compare:
  - which **single protein** predicts best
  - which **protein combinations** predict best
  - whether **LoRA fine-tuning** outperforms a **non-fine-tuned baseline**

The code is written to be practical for bacterial AMR use cases, especially
for *Pseudomonas aeruginosa*.

---

## 1) Suggested proteins and combinations to try first

These suggestions are especially relevant for *P. aeruginosa* because:
loss or alteration of **OprD** is strongly linked to carbapenem resistance,
efflux systems such as **MexAB-OprM** and **MexXY-OprM** contribute to multidrug resistance,
AmpC contributes to beta-lactam resistance, and **GyrA/ParC** mutations are classic
fluoroquinolone resistance determinants. ESM-2 is available through Meta's ESM repository,
and LoRA is a standard PEFT method supported by Hugging Face PEFT. citeturn530694search8turn530694search19turn530694search2turn530694search18

### Carbapenem-focused panel
- `oprD`
- `ampC`
- `mexB`
- `oprM`
- `mexR`
- `nalC`
- `nalD`
- `ftsI` (optional)

Recommended combinations:
- `oprD`
- `oprD + ampC`
- `oprD + mexB + oprM`
- `oprD + ampC + mexB + oprM`
- `oprD + mexR + nalC + nalD`

### Fluoroquinolone-focused panel
- `gyrA`
- `parC`
- `parE`
- `gyrB`

Recommended combinations:
- `gyrA`
- `gyrA + parC`
- `gyrA + parC + parE`
- `gyrA + parC + gyrB + parE`

### Aminoglycoside / broad MDR panel
- `mexX`
- `mexY`
- `oprM`
- `fusA1` (if available)
- `amgS` (if available)

Recommended combinations:
- `mexY`
- `mexX + mexY`
- `mexX + mexY + oprM`

### Beta-lactam / broad MDR mixed panel
- `oprD`
- `ampC`
- `mexB`
- `mexY`
- `gyrA`
- `parC`

Recommended combinations:
- `oprD + ampC`
- `oprD + mexB + mexY`
- `oprD + ampC + mexB + gyrA + parC`

### Good experimental rule
Start with:
1. one high-confidence focal protein
2. one biologically motivated 2-protein combination
3. one 4–6 protein panel

That gives you interpretable results before trying very large panels.

---

## 2) Directory layout

```text
amr_protein_llm_pipeline/
├── README.md
├── environment.yml
├── requirements.txt
├── configs/
│   └── protein_panels_p_aeruginosa.yaml
├── data/
│   ├── README_data_format.md
│   └── example/
│       ├── phenotypes.tsv
│       ├── oprD.faa
│       └── ampC.faa
├── notebooks/
│   └── amr_protein_llm_pipeline.ipynb
├── scripts/
│   ├── 01_prepare_metadata.py
│   ├── 02_cluster_sequences_mmseqs.py
│   ├── 03_extract_embeddings.py
│   ├── 04_build_feature_table.py
│   ├── 05_train_frozen_baseline.py
│   ├── 06_train_lora.py
│   ├── 07_compare_models.py
│   ├── 08_plot_benchmarks.py
│   ├── utils.py
│   └── run_pipeline.sh
└── results/
```

---

## 3) Input format

### 3.1 Phenotype table

A TSV file with at least:

| isolate_id | phenotype |
|---|---|
| iso_001 | R |
| iso_002 | S |

Optional columns:
- `antibiotic`
- `study`
- `lineage`
- `country`
- `year`
- `split_hint`

Binary phenotypes can be:
- `R/S`
- `1/0`
- `resistant/susceptible`

### 3.2 Protein FASTA files

One FASTA per protein. Example:
- `oprD.faa`
- `ampC.faa`
- `gyrA.faa`

Headers should contain an isolate identifier that can be matched back to the phenotype table.

Example:
```fasta
>iso_001
MTPA...
>iso_002
MTPA...
```

or

```fasta
>iso_001|oprD
MTPA...
```

The scripts normalize headers to infer `isolate_id`.

---

## 4) Why grouped CV by sequence cluster matters

If highly similar sequences appear in both train and test, your evaluation can be overly optimistic.
This pipeline supports **grouped cross-validation using sequence clusters**, so sequences from the same
cluster stay in the same fold.

Recommended clustering:
- identity threshold: `0.90` for near-duplicate control
- stricter test: `0.70` or `0.80`

For single-protein models:
- group by that protein's cluster

For multi-protein combinations:
- group by a composite of cluster IDs across proteins

---

## 5) What the pipeline does

### Step A
Prepare metadata and normalize protein headers.

### Step B
Cluster each protein with MMseqs2.

### Step C
Extract ESM-2 embeddings.

### Step D
Build feature tables for:
- each single protein
- each protein combination

### Step E
Train a frozen baseline:
- logistic regression
- random forest
- xgboost (optional)
- shallow MLP (optional)

### Step F
Train LoRA fine-tuned ESM-2 models per protein.

### Step G
Compare frozen vs LoRA models with grouped CV.

### Step H
Generate publication-quality figures:
- ROC curves
- PR curves
- boxplots of CV metrics
- grouped benchmark barplots
- calibration plots
- embedding PCA/UMAP

---

## 6) Environment setup

### Conda

```bash
conda env create -f environment.yml
conda activate amr-protein-llm
```

If you want GPU:
- install a CUDA-compatible PyTorch build for your system

### MMseqs2
The clustering script expects `mmseqs` on your PATH.

Example install:
```bash
conda install -c conda-forge -c bioconda mmseqs2
```

---

## 7) Quick start

### Example run with single protein: OprD

```bash
bash scripts/run_pipeline.sh \
  --phenotypes data/example/phenotypes.tsv \
  --protein-fasta data/example/oprD.faa \
  --protein-name oprD \
  --outdir results/oprD_demo
```

### Example run with two proteins: OprD + AmpC

```bash
python scripts/04_build_feature_table.py \
  --embedding-tsv results/oprD_demo/embeddings/oprD_embeddings.tsv \
  --embedding-tsv results/ampC_demo/embeddings/ampC_embeddings.tsv \
  --protein-name oprD \
  --protein-name ampC \
  --phenotypes data/example/phenotypes.tsv \
  --cluster-tsv results/oprD_demo/clusters/oprD_clusters.tsv \
  --cluster-tsv results/ampC_demo/clusters/ampC_clusters.tsv \
  --combine-method concat \
  --out-prefix results/combos/oprD_ampC
```

Then train the frozen baseline:
```bash
python scripts/05_train_frozen_baseline.py \
  --features results/combos/oprD_ampC_features.tsv \
  --label-col phenotype_binary \
  --group-col group_id \
  --models logreg rf mlp \
  --n-jobs 8 \
  --outdir results/combos/oprD_ampC_baseline
```

---

## 8) Recommended experiment plan

### Phase 1: single proteins
Run:
- `oprD`
- `ampC`
- `mexB`
- `gyrA`
- `parC`
- `mexY`

This identifies the strongest single predictor.

### Phase 2: pairs
Run:
- `oprD + ampC`
- `oprD + mexB`
- `gyrA + parC`
- `mexX + mexY`

### Phase 3: biologically coherent panels
Run:
- carbapenem panel
- fluoroquinolone panel
- broad MDR panel

### Phase 4: fine-tuning
Apply LoRA first to the best:
- 2 single proteins
- 2 top combinations by frozen baseline

Note:
this repository fine-tunes at the **single-protein sequence** level.
For multi-protein panels, the pipeline currently recommends:
- frozen embeddings + classical model, or
- future extension to multi-branch LoRA

---

## 9) Fair comparison: how to prove fine-tuning helps

To show that LoRA helps beyond a frozen model:

1. keep the **same backbone family**
2. keep the **same grouped CV folds**
3. compare **fold-wise metrics**
4. report:
   - AUROC
   - AUPRC
   - F1
   - balanced accuracy
   - MCC
5. run paired tests or bootstrap confidence intervals on fold-level differences

This repository provides fold-level metric tables to support those comparisons.

---

## 10) Main output files

### Per run
- embeddings TSV
- clusters TSV
- feature table TSV
- fold predictions TSV
- metric summary TSV
- plots PDF/PNG

### Important comparison files
- `benchmark_summary.tsv`
- `benchmark_by_fold.tsv`
- `model_comparison_stats.tsv`

---

## 11) Compute notes

### Embedding extraction
Parallelism happens through:
- GPU batching
- multiprocessing data loading

### Frozen baselines
Parallelism happens through:
- `n_jobs`
- cross-validation loop parallelization

### LoRA
Parallelism happens through:
- GPU
- gradient accumulation
- mixed precision if available

---

## 12) Practical advice

### If dataset is small
Use:
- `esm2_t6_8M_UR50D`
- frozen baseline first
- LoRA only if baseline shows real signal

### If dataset is medium
Use:
- `facebook/esm2_t12_35M_UR50D`
- LoRA on attention layers only

### If dataset is large
Try:
- `facebook/esm2_t30_150M_UR50D`
- stronger regularization
- external holdout

---

## 13) Publication-quality figure set

Recommended minimum figure panel:

1. **Benchmark barplot**
   - AUROC/AUPRC for each protein and model

2. **Boxplot by fold**
   - one box per model

3. **ROC curves**
   - for best single protein

4. **PR curves**
   - especially important if classes are imbalanced

5. **Calibration plot**
   - frozen vs LoRA

6. **Embedding PCA/UMAP**
   - resistant vs susceptible

7. **Protein-combination heatmap**
   - rows = proteins/combinations
   - columns = metrics

---

## 14) Limitations

- LoRA training is implemented per single protein sequence, not yet as a joint multi-protein sequence model.
- Multi-protein comparison is currently done by combining frozen embeddings.
- Protein-based prediction may miss resistance driven mainly by:
  - gene regulation
  - promoter variants
  - accessory genes
  - copy number
  - non-target mechanisms

---

## 15) Next extension ideas

- multi-branch neural network for protein combinations
- nested CV
- temporal holdout
- lineage holdout
- SHAP / attribution maps
- external validation cohort
