# PubMed Relevance Classifier

A **pandas + scikit-learn**. It downloads biomedical papers from PubMed, trains a text classifier to mark papers as **Relevant** or **Not Relevant**, and writes filtered results to **Parquet**.

Results have been put into the .gitignore file due to size. if you would like to see the results, clone the repo and follow the below steps.


## What it does

| Step | Description |
|------|-------------|
| **Download** | Fetches PubMed records as XML via [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/) |
| **Parse** | Converts Medline XML to a table using [`pubmed_parser`](https://github.com/titipata/pubmed_parser) |
| **Train** | Builds a TF-IDF + Random Forest model on labeled papers |
| **Setup pipeline** | Downloads papers for your query, keeps only `Relevant` rows, saves Parquet |
| **Update pipeline** | Fetches new/changed papers since the last run and merges into the main Parquet file |

The original library expected you to supply a pre-trained `joblib` classifier. This version includes training so you can go end-to-end from search queries to a saved model.

## Project layout

```
pubmed/
├── README.md                 # this file
├── requirements.txt
├── config.example.yaml       # copy to config.yaml (optional)
├── pubmed-classifier.ipynb   # walkthrough notebook (Kaggle or local)
├── src/
│   ├── download.py           # E-utilities esearch + efetch
│   ├── parse_papers.py       # XML → DataFrame
│   ├── features.py           # text fields for the model
│   ├── model.py              # train / predict / joblib I/O
│   └── pipeline.py           # setup + update pipelines
├── scripts/
│   ├── train_classifier.py   # train and save classifier.joblib
│   ├── run_setup.py          # initial download + classify
│   └── run_update.py         # incremental update
├── data/                     # created at runtime (gitignored)
└── artifacts/                # model + last-run pickle (gitignored)
```

## Requirements

- Python 3.10+
- Internet access (PubMed API)
- Optional: [NCBI API key](https://www.ncbi.nlm.nih.gov/account/) for higher rate limits

### Install locally

From the `pubmed` folder:

```bash
cd pubmed
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data source

All paper metadata comes from **PubMed** (`db=pubmed`) at:

`https://eutils.ncbi.nlm.nih.gov/entrez/eutils/`

You provide **PubMed search queries** (same syntax as on the PubMed website). The classifier itself is trained from **your labels**—either a CSV you create or weak labels from a positive vs negative query pair.

## Quick start

### 1. Train a classifier (weak supervision)

Use one query for papers you want (`Relevant`) and one for papers you do not (`Not Relevant`):

```bash
python scripts/train_classifier.py \
  --positive-query "machine learning[Title/Abstract] AND oncology[MeSH Terms]" \
  --negative-query "cardiology[MeSH Terms] NOT machine learning[Title/Abstract]" \
  --max-per-query 500
```

Model is saved to `artifacts/classifier.joblib`.

### 2. Run the setup pipeline

Download papers matching your topic, classify them, save Parquet:

```bash
python scripts/run_setup.py \
  --queries "machine learning[Title/Abstract] AND oncology[MeSH Terms]" \
  --max-records 1000
```

Output: `data/relevant_papers.parquet`

### 3. Run an update (next day or later)

PubMed updates once per day; wait at least one day between runs.

```bash
python scripts/run_update.py \
  --queries "machine learning[Title/Abstract] AND oncology[MeSH Terms]"
```

Outputs:

- `data/relevant_papers.parquet` — merged main dataset  
- `data/new_and_updated.parquet` — only the latest batch  

## Training with your own labels

Create `labels.csv`:

```csv
python scripts/run_setup.py \
  --queries "machine learning[Title/Abstract] AND oncology[MeSH Terms]" \
  --max-records 1000
```

Option A — labels include everything you need:

```bash
python scripts/train_classifier.py --labels-csv labels.csv
```

Option B — labels + XML you already downloaded:

```bash
python scripts/train_classifier.py --labels-csv labels.csv --xml-dir data/xml
```

Labels must be exactly `Relevant` or `Not Relevant` (matches the original Pubmed-Pipeline filter).

## Python API

Same class names as the original library, without Spark:

```python
from pathlib import Path
from src.pipeline import PubmedPipelineSetup, PubmedPipelineUpdate

setup = PubmedPipelineSetup(
    xml_dir="data/xml",
    classifier_path="artifacts/classifier.joblib",
    dataframe_output_path="data/relevant_papers.parquet",
    last_run_path="artifacts/last_run.pkl",
)

setup.download_xml_from_pubmed(
    ["machine learning[Title/Abstract] AND oncology[MeSH Terms]"],
    api_key=None,
    max_records=500,
)
df = setup.run_pipeline()
```

Update pipeline:

```python
update = PubmedPipelineUpdate(
    xml_dir="data/xml_update",
    classifier_path="artifacts/classifier.joblib",
    main_dataframe_path="data/relevant_papers.parquet",
    last_run_path="artifacts/last_run.pkl",
    new_and_updated_path="data/new_and_updated.parquet",
)
update.download_xml_from_pubmed(["your query"], api_key=None)
main_df, new_df = update.run_pipeline()
```

## Notebook (Kaggle or local)

Open `pubmed-classifier.ipynb`. It:

1. Checks how many PubMed hits your queries return  
2. Trains a classifier from positive/negative queries  
3. Runs the setup pipeline and previews results  

**On Kaggle:** upload this `pubmed` folder (or clone the repo into the notebook), enable **Internet** in notebook settings, and set `ROOT` to `/kaggle/working/pubmed` if needed. No GPU required—this is CPU + API bound.

**Locally:** run Jupyter from the `pubmed` directory so imports resolve.

## Model details

The classifier mirrors the original pipeline’s input fields:

- `abstract`, `title`, `medline_ta`, `keywords`, `publication_types`, `chemical_list`, `country`, `author`, `mesh_terms`

Each field gets its own **TF-IDF** vectorizer; features are combined and passed to a **Random Forest**. Only rows predicted as **`Relevant`** are kept in the output Parquet files.

To customize features or algorithms, edit `src/features.py` and `src/model.py`.

## Configuration file (optional)

```bash
cp config.example.yaml config.yaml
# edit queries and paths
```

Scripts use CLI flags by default; you can load `config.yaml` in your own wrapper if you prefer YAML-driven runs.

## Comparison with Pubmed-Pipeline

| | Original Pubmed-Pipeline | This project |
|---|--------------------------|--------------|
| Runtime | PySpark | pandas |
| Download | bash + wget + parallel | Python `requests` |
| Classifier | You provide `joblib` | Train included |
| Output | Parquet | Parquet |
| Update logic | mdat + edat union | Same E-utilities pattern |

## NCBI usage

This software uses data from NLM® PubMed. Respect [NCBI E-utilities policies](https://www.ncbi.nlm.nih.gov/home/about/policies/) (rate limits, citation, no excessive automated re-querying). An API key is recommended for regular use.

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| HTTP 429 | Add `--api-key`, reduce `--max-records`, wait and retry |
| No XML files | Check queries on pubmed.ncbi.nlm.nih.gov first |
| Update fails “less than 1 day” | Wait 24h between runs (PubMed daily refresh) |
| Empty Parquet | Classifier may be too strict; retrain with better labels |
| `pubmed_parser` import error | `pip install -r requirements.txt` |
| `BadGzipFile: Not a gzipped file` | Fixed in this repo — re-run after pulling latest `parse_papers.py` |

## License note

Inspired by [Pubmed-Pipeline](https://github.com/nicford/Pubmed-Pipeline) (GPL). This folder is new code for the `kaggle-ML` repo; check repo-level license for distribution terms.
