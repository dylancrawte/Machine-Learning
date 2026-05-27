# PubMed Relevance Classifier

Train a PubMed text classifier and keep only papers predicted as `Relevant`.

This project:
- downloads PubMed XML via NCBI E-utilities
- parses records into pandas
- trains a TF-IDF + Random Forest classifier
- writes filtered outputs to Parquet

Generated outputs are gitignored.

## Requirements

- Python 3.10+
- Internet access
- Optional NCBI API key for higher rate limits

## Setup

```bash
cd pubmed
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run on Kaggle

In a Kaggle notebook, run shell commands with `!` (no local venv needed):

```bash
!pip install -r pubmed/requirements.txt
!python pubmed/scripts/train_classifier.py --positive-query "machine learning[Title/Abstract] AND oncology[MeSH Terms]" --negative-query "cardiology[MeSH Terms] NOT machine learning[Title/Abstract]" --max-per-query 500
!python pubmed/scripts/run_setup.py --queries "machine learning[Title/Abstract] AND oncology[MeSH Terms]" --max-records 1000
```

For updates (on a later day):

```bash
!python pubmed/scripts/run_update.py --queries "machine learning[Title/Abstract] AND oncology[MeSH Terms]"
```

## Quick Start

### 1) Train classifier

```bash
python scripts/train_classifier.py \
  --positive-query "machine learning[Title/Abstract] AND oncology[MeSH Terms]" \
  --negative-query "cardiology[MeSH Terms] NOT machine learning[Title/Abstract]" \
  --max-per-query 500
```

Output: `artifacts/classifier.joblib`

### 2) Run initial setup pipeline

```bash
python scripts/run_setup.py \
  --queries "machine learning[Title/Abstract] AND oncology[MeSH Terms]" \
  --max-records 1000
```

Output: `data/relevant_papers.parquet`

### 3) Run incremental update (later)

Run at most once per day:

```bash
python scripts/run_update.py \
  --queries "machine learning[Title/Abstract] AND oncology[MeSH Terms]"
```

Outputs:
- `data/relevant_papers.parquet` (merged main dataset)
- `data/new_and_updated.parquet` (latest batch only)

## Train With Your Own Labels

Create `labels.csv`:

```csv
pmid,label
12345678,Relevant
23456789,Not Relevant
```

Then run:

```bash
python scripts/train_classifier.py --labels-csv labels.csv
```

If your CSV only has labels and PMIDs, and you already have XML:

```bash
python scripts/train_classifier.py --labels-csv labels.csv --xml-dir data/xml
```

## Common Flags

- `--api-key`: optional NCBI API key
- `--max-records`: cap downloaded records for setup/update
- `--skip-download`: reuse existing XML for setup/update/train

## Troubleshooting

- HTTP 429: add `--api-key`, reduce `--max-records`, retry later
- No XML files: validate query first on PubMed website
- Update says less than 1 day: wait 24h between update runs
- Empty output: retrain classifier with better labels
