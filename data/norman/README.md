# Norman Perturb-seq analysis

The input is `NormanWeissman2019_filtered.h5ad` from [scPerturb](https://scperturb.org/), [Zenodo record 7041849](https://doi.org/10.5281/zenodo.7041849). The underlying experiment is Norman et al. (2019), *Exploring Genetic Interaction Manifolds Constructed from Rich Single-Cell Phenotypes*, GEO GSE133344. The distribution is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Input and preprocessing

- 111,445 cells and 33,694 genes; 11,855 controls, 57,831 single-perturbation cells, and 41,759 double-perturbation cells.
- Counts are in `adata.X`. Perturbations are in `obs['perturbation']`, with `control` for controls and an underscore between two targets.
- All cells in the supplied filtered file are used. Counts are normalized to 10,000 per cell and log1p transformed.
- The 2,000 genes with greatest variance are selected using controls and singles. Variances use a random sample of 50,000 of those cells, with seed 0. Processing runs in gene blocks to limit memory use.
- Eligibility requires at least 200 control cells, 50 cells per single perturbation, and 30 cells per double perturbation. Both constituent singles must be available.
- The precision diagnostic uses 20 centered PCs fitted to all control and single cells, followed by Ledoit-Wolf precision estimation in each context. A control-only split uses the same coordinates and estimator (seed 0; 5,928 and 5,927 cells). Its rho2 is 0.501; the single-perturbation median is 0.570. The group sizes differ, so this is a descriptive reference.

The additive predictor and its direction/magnitude metrics use the same genes and context means. Double-perturbation cells enter evaluation only. The optional top-100 metric selects genes by their observed double-versus-control change.

## Reproduce

From the repository root:

```bash
python -m pip install -r requirements-data.txt
python scripts/download_norman_data.py
python scripts/run_norman_realdata_experiment.py --h5ad external_data/NormanWeissman2019_filtered.h5ad --label-column perturbation --target-separator _ --n-hvgs 2000 --n-pcs 20 --min-cells-control 200 --min-cells-single 50 --min-cells-double 30 --seed 0 --outdir outputs/norman
```

The downloader checks the file against Zenodo's checksum and saves its source metadata. The cell-level file is not included here.

## Included files

| File | Contents |
| --- | --- |
| `source.json` | Dataset record, download URL, size, and checksums |
| `config.json`, `metadata.json` | Analysis settings, package versions, and sample counts |
| `selected_genes.csv` | The 2,000 genes in variance order |
| `context_counts.csv` | Cell counts by perturbation context |
| `context_means.csv` | Mean normalized log expression on the selected genes |
| `pairs.csv` | Double perturbations and their constituent singles |
| `double_prediction_metrics.csv` | Additive R2, raw correlation, RMSE, and top-100 R2 |
| `direction_magnitude_metrics.csv` | Direction cosine, norm ratio, and decomposed R2 for each pair |
| `direction_magnitude_summary.json` | Summaries across pairs |
| `lowrank_audit_metrics.csv` | Precision-difference concentration for each single perturbation |
| `control_split_baseline.json` | Seeded control-control comparison in the same PCA coordinates |

`python run_reproduce.py` also writes the cohort-level failure counts and Spearman correlations to `outputs/norman_descriptive_summary.json`. It recomputes the pair metrics from `context_means.csv` and redraws the report figures. The results in this directory come from the scPerturb input above; they replace the course submission's smaller subset analysis.
