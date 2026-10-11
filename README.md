# Benchmarking feature selection for circulating miRNA lung-cancer biomarkers

**Manuscript:** *Beyond cross-validated accuracy: a multi-criteria benchmark of ten feature-selection algorithms for circulating microRNA lung-cancer biomarkers across specimens and platforms*

This repository contains the analysis notebook and scripts for a multi-criteria benchmark of ten feature-selection algorithms, Borda rank aggregation, and a random-selection negative control. It assesses nested cross-validated discrimination, selection stability, overlap-aware external transportability, biological concordance, redundancy, global-signal capture, and runtime.

## Contents

- `PaperLu3_analysis.ipynb` — executed analysis notebook.
- `PaperLu3_analysis.py` — script version of the analysis.
- `fs_methods.py` — implementations of ten feature-selection methods and Borda aggregation.
- `lung_mirna_utils.py` — shared cohort-loading, harmonization and analysis utilities.
- `mirbase_hsa_alias_all_versions.csv` — miRBase alias reference used in harmonization.
- `results_summary.json` — supplied analysis summary.
- `cache/rankings_D1.pkl`, `rankings_D2.pkl`, `rankings_D3.pkl` — precomputed full, fold and subsample rankings.

## Data

The preprocessed input datasets are distributed separately from the code. Download the data archive from the companion dataset record, and extract it so that `Lung_miRNA_Preprocessed_Datasets/` is in the repository directory or a parent directory. The shared loader searches the current directory and parent directories for this folder. Verify the required input files before running.

**Companion data:** https://www.kaggle.com/datasets/ntknhuong/lung-cancer-mirna-preprocessed-datasets

The seven blood cohorts are E-MTAB-8026, GSE137140, GSE106817, GSE112264, GSE113486, GSE31568 and GSE61741. A TCGA lung tissue dataset is used for biological concordance. Cite the original data providers when reusing the data.

## Reproduce

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook PaperLu3_analysis.ipynb
```

The notebook writes generated figures to `figures/` and tables to `tables/`. These output folders are ignored by Git by default; remove the corresponding `.gitignore` entries if you want to publish the outputs. `results_summary.json` is supplied as an analysis artifact.

### Cached rankings

The three `.pkl` files are included to avoid rerunning the most computationally expensive feature-ranking stage. To recompute rankings from scratch, **delete or rename `cache/`** before executing the notebook. The notebook regenerates the cache automatically. Runtimes depend on hardware and software versions.

**Security:** Python pickle files can execute code when loaded. Only unpickle files from a source you trust. These files are provided as supplied and have not been independently verified against a fresh full run.

## Reproducibility and methodological notes

- Discovery regimes: D1 E-MTAB-8026 (whole blood), D2 GSE137140 (serum), D3 GSE61741 (whole blood).
- Common feature universe: 816 miRNAs according to the supplied notebook.
- Feature panel sizes: 5, 10 and 20; downstream classifiers: L2 logistic regression and random forest.
- The code computes feature rankings separately on training folds for cross-validated evaluation; readers should inspect the full notebook and any remaining preprocessing or hyperparameter-selection steps before making a claim of complete leakage-free nested validation.
- Dataset reuse permissions and source attribution should be checked independently before republishing redistributed inputs under a new license.

## Citation and release

Please cite the associated manuscript and original datasets. Add the final manuscript citation and a Zenodo version DOI when available. A GitHub release can be archived in Zenodo after repository review.
