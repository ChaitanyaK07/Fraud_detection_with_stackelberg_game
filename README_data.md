# Dataset Instructions

## Where to place the file

Download the dataset and place it at:

```
HAD-Stack/data/fraud_dataset.csv
```

The file must be named exactly `fraud_dataset.csv`.

---

## Why the dataset is not in the repo

The raw CSV is **not committed to the repository** for two reasons:

1. **File size** — CSV files with tens of thousands of rows and 65 columns
   are too large for standard Git hosting. GitHub has a 100 MB file limit
   and the repo will become slow to clone.

2. **Data sharing policy** — The dataset may be subject to usage restrictions.
   Distributing it inside a public repo without explicit permission is not appropriate.

---

## How to share the dataset with collaborators

You have three good options depending on your situation:

### Option A — Google Drive (simplest for academic use)

1. Upload `fraud_dataset.csv` to a shared Google Drive folder
2. Add the shareable link to this README under "Download Link" below
3. Collaborators download it and place it at `data/fraud_dataset.csv`

### Option B — Kaggle Dataset (best for public release)

1. Go to [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Create a new dataset, upload the CSV
3. Add the Kaggle dataset URL here
4. Collaborators run: `kaggle datasets download <your-username>/<dataset-name>`

### Option C — Zenodo (best for academic citation)

1. Go to [zenodo.org](https://zenodo.org) and log in with your ORCID or GitHub
2. Upload the CSV, fill in metadata (title, description, license)
3. Zenodo gives you a permanent DOI (e.g. `10.5281/zenodo.XXXXXXX`)
4. Add the DOI here — this is citable in the paper

---

## Download Link

```
[FILL IN AFTER UPLOADING]
```

---

## Dataset Description

Once placed correctly, `1_data_prep.py` will load it automatically.

| Property | Value |
|----------|-------|
| Filename | `fraud_dataset.csv` |
| Total rows | 26,393 |
| Raw features | 65 |
| Legitimate transactions | 21,848 (82.8%) |
| Fraudulent transactions | 4,545 (17.2%) |
| Class ratio | 4.8 : 1 |

**Note on leakage:** The dataset is synthetically generated and contains 25+
features that directly encode the class label. `1_data_prep.py` removes all of
them automatically. Do not use the raw 65-feature version for modelling — it
produces trivial AUC = 1.0.

---

## .gitignore entry

The root `.gitignore` already excludes the CSV:

```
data/*.csv
data/*.parquet
data/*.json
models/*.pt
```
