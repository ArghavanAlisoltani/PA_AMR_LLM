# Data format

## Phenotype file

Required columns:
- `isolate_id`
- `phenotype`

Accepted phenotype values:
- `R`, `S`
- `1`, `0`
- `resistant`, `susceptible`

Optional:
- `antibiotic`
- `lineage`
- `site`
- `year`

Example:

```tsv
isolate_id	phenotype	antibiotic
iso_001	R	imipenem
iso_002	S	imipenem
iso_003	R	imipenem
```

## FASTA files

One FASTA per protein.

Example:
```fasta
>iso_001
MKK...
>iso_002
MKK...
```

or
```fasta
>iso_001|oprD
MKK...
```

## Important assumptions

- one sequence per isolate per protein
- the isolate identifier can be parsed from the FASTA header
- if multiple sequences exist for one isolate and protein, keep only the best curated representative

## Example files included

- `example/phenotypes.tsv`
- `example/oprD.faa`
- `example/ampC.faa`
