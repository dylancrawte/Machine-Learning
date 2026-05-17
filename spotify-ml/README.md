# Spotify Machine Learning

This folder has a couple of small ML workflows built around Spotify-style audio features. One uses **K-means** to group songs into two playlist-style clusters. The other trains a classifier to predict mood labels — **Calm**, **Energetic**, **Happy**, or **Sad** — from the same kind of feature data.

If you are just getting started, use **Path A** below. You do not need a Spotify Developer account or API keys; it runs on the CSV data already in this repo.

**Running on Kaggle:** This project does not need a GPU, but Kaggle Notebooks are still a handy way to run everything in the browser without setting up Python locally. Create a notebook at [kaggle.com/code](https://www.kaggle.com/code), upload `spotify-ml.ipynb` (or clone this repo into the notebook session), then install dependencies with `!pip install -r spotify-ml/requirements.txt` in a cell. The CSV files under `spotify-ml/data/` are already in the repo, so you do not need to add an extra Kaggle dataset for Path A. Run the script commands from the notebook (`!python spotify-ml/scripts/run_clustering.py`, and so on) or work through the notebook cells if you prefer.

---

## Install (both paths)

```bash
cd kaggle-ML/spotify-ml
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

- Python 3.10+
- **No GPU** required (fine on an 8GB laptop)
- Path A needs **no internet** after install

---

## Path A — No Spotify app (recommended)

Use this if you do **not** want a Spotify Developer account or API keys. Everything uses CSV files in `data/`.

### What you get

| Step | Command | Output |
|------|---------|--------|
| Cluster ~400 songs | `python scripts/run_clustering.py` | `data/cluster0.csv`, `cluster1.csv` |
| Train mood model | `python scripts/train_mood_classifier.py` | `artifacts/mood_model.joblib` |
| Predict a mood | `python scripts/predict_mood.py --from-csv "1999"` | Printed prediction |

### Step-by-step

**1. Cluster songs into two groups**

```bash
python scripts/run_clustering.py
```

Open `data/cluster0.csv` and `data/cluster1.csv`. Each file is a list of tracks with `danceability`, `energy`, `valence`, `loudness`, and a cluster label. Build playlists manually in Spotify if you like.

**2. Train the mood classifier**

```bash
python scripts/train_mood_classifier.py
```

Uses `data/data_moods.csv` (686 labeled songs). You should see cross-validation and test accuracy in the terminal.

**3. Predict mood for a song in the dataset**

```bash
python scripts/predict_mood.py --from-csv "1999"
```

Try other titles from `data/data_moods.csv` (column `name`). Partial matches work:

```bash
python scripts/predict_mood.py --from-csv "Radiohead"
```

**4. (Optional) Notebook walkthrough**

```bash
jupyter notebook spotify-ml.ipynb
```

Same workflow with plots (3D cluster view, confusion matrix).

### Path A checklist

- [ ] `pip install -r requirements.txt`
- [ ] `python scripts/run_clustering.py`
- [ ] `python scripts/train_mood_classifier.py`
- [ ] `python scripts/predict_mood.py --from-csv "1999"`

No `.env` file needed. No Spotify login.

---

## Path B — With Spotify Developer app (optional)

Use this only if you want to fetch **live** track metadata and audio features from Spotify (e.g. predict mood for any track ID).

### Limitations (read first)

- You must register an app at [Spotify Developer Dashboard](https://developer.spotify.com/dashboard).
- **Redirect URI is not required** for mood prediction (`--track-id`) — only Client ID + Client Secret.
- Since [Nov 2024](https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api), Spotify often returns **403 Forbidden** on `/v1/audio-features` for **new apps** in Development Mode. If that happens, **use Path A** — your credentials can still be correct.
- Never commit real secrets to git. Use `.env` only (see `.gitignore`).

### Step-by-step

**1. Create a Spotify app**

1. Log in at [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard).
2. Click **Create app**.
3. Name it (e.g. `kaggle-ml-mood`), accept terms, create.
4. Open the app → **Settings**.
5. Copy **Client ID** and **Client secret** (click *View client secret*).

**Redirect URI (optional for this project):**  
You do **not** need one for `predict_mood.py --track-id`. If the dashboard requires at least one URI, add:

```text
http://127.0.0.1:8888/callback
```

Save settings. These scripts do not use user login or playlist creation.

**2. Configure credentials locally**

```bash
cd kaggle-ML/spotify-ml
cp config.example.env .env
```

Edit `.env` (no quotes around values):

```env
SPOTIFY_CLIENT_ID=paste_your_client_id_here
SPOTIFY_CLIENT_SECRET=paste_your_client_secret_here
```

**3. Train the model (same as Path A)**

You still train on bundled CSVs — the API does not replace `data_moods.csv`:

```bash
python scripts/train_mood_classifier.py
```

**4. Predict from a Spotify track ID**

From a Spotify link like `https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b`, the ID is `0VjIjW4GlUZAMYd2vXMi3b`.

```bash
python scripts/predict_mood.py --track-id 0VjIjW4GlUZAMYd2vXMi3b
```

**If you get 403:** Spotify blocked audio features for your app. Fall back to Path A:

```bash
python scripts/predict_mood.py --from-csv "Song Name"
```

**5. (Advanced) Build your own CSV**

Only if your app has audio-features access: use `src/spotify_client.py` to download tracks into a DataFrame (same columns as `data_moods.csv`). Most new apps cannot do this without [extended API access](https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api).

### Path B checklist

- [ ] Spotify Developer app created
- [ ] Client ID + Secret in `.env` (not in `config.example.env`)
- [ ] `python scripts/train_mood_classifier.py`
- [ ] `python scripts/predict_mood.py --track-id <id>` **or** use Path A if 403

---

## Which path should I use?

| Goal | Use |
|------|-----|
| Learn clustering + mood ML on a laptop | **Path A** |
| Kaggle / offline / no signup | **Path A** |
| Predict mood for songs already in `data_moods.csv` | **Path A** (`--from-csv`) |
| Fetch live features for arbitrary Spotify tracks | **Path B** (may hit 403 on new apps) |
| Create playlists via API (original repo) | Not implemented here; needs OAuth + redirect URI |

---

## How the ML works

### Clustering

**Features:** `danceability`, `energy`, `valence`, `loudness`  
**Steps:** merge `df1.csv` + `df2.csv` → MinMaxScaler → K-means (`k=2`) → save cluster CSVs.

### Mood classification

**Features:** 10 audio columns (see `src/features.py`)  
**Labels:** `Calm`, `Energetic`, `Happy`, `Sad`  
**Model:** MinMaxScaler + MLPClassifier (sklearn replacement for the original Keras net).

---

## Project layout

```
spotify-ml/
├── README.md
├── requirements.txt
├── config.example.env      # template only — copy to .env for Path B
├── spotify-ml.ipynb
├── data/                   # Path A: all you need
│   ├── data_moods.csv
│   ├── df1.csv
│   └── df2.csv
├── src/
├── scripts/
└── artifacts/              # created when you train
```

---

## Python API

```python
from src.clustering import load_clustering_data, run_clustering, save_cluster_outputs
from src.mood_model import load_mood_data, train_mood_classifier, predict_mood_from_features

df = load_clustering_data("data/df1.csv", "data/df2.csv")
clustered, _, _ = run_clustering(df)
save_cluster_outputs(clustered, "data")

moods = load_mood_data("data/data_moods.csv")
pipeline, encoder, metrics = train_mood_classifier(moods)
print(predict_mood_from_features(moods.iloc[0], pipeline, encoder))
```

---

## Troubleshooting

| Issue | Path | Fix |
|-------|------|-----|
| `No module named ...` | A or B | Activate venv; `pip install -r requirements.txt` |
| `invalid_client` | B | Real Client ID/Secret in `.env`; reset secret in dashboard if needed |
| `403` on `audio-features` | B | Use **Path A** (`--from-csv`); see [API changes](https://developer.spotify.com/blog/2024-11-27-changes-to-the-web-api) |
| Song not found with `--from-csv` | A | Pick a `name` from `data/data_moods.csv` |
| Low accuracy | A or B | Expected on small data; more labels would help |
| Secrets in git | B | Only `.env` (gitignored); never put real keys in `config.example.env` |

---

## Credits

Based on [Spotify-Machine-Learning](https://github.com/alejandrolucchesi/Spotify-Machine-Learning) and the Towards Data Science articles linked in that repo.
