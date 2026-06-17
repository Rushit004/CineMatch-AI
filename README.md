# 🎬 CineMatch-AI

<div align = 'center'>

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Deployed on Streamlit Cloud](https://img.shields.io/badge/Deployed-Streamlit%20Cloud-FF4B4B?style=for-the-badge&logo=streamlit)](https://cinematch-ai-mrs.streamlit.app/)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)](https://cinematch-ai-mrs.streamlit.app/)

</div>

<br/>

### *Stop scrolling for something to watch. Let the model pick.*

A content-based movie recommender that reads a movie's plot, genres, keywords, cast,
and director — turns all of it into a single feature vector — and uses cosine
similarity to surface the 5 closest matches, posters included.

🌐 **[Open Live Demo →](https://cinematch-ai-mrs.streamlit.app/)**

## Project Preview

> 📸 *Add a screenshot or short GIF of the deployed app (e.g. `screenshots/01_app_overview.png`) here.*

---

## Live demo

🌐 **[cinematch-ai-mrs.streamlit.app](https://cinematch-ai-mrs.streamlit.app/)**

Pick any movie from the dropdown and hit **Recommend** — no login, no setup, works instantly in the browser.

## How to run this project locally

### Prerequisites
- Python 3.10+
- [Git LFS](https://git-lfs.com/) — `movies.pkl` and `similarity.pkl` are tracked with Git LFS
- A free [TMDB API key](https://www.themoviedb.org/settings/api) (for fetching live posters)

### Steps
```bash
# Clone the repository (Git LFS required to pull the .pkl model files)
git lfs install
git clone https://github.com/Rushit004/CineMatch-AI.git
cd CineMatch-AI

# Create a virtual environment & install dependencies
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt

# Add your TMDB API key as a Streamlit secret
mkdir .streamlit
echo TMDB_API_KEY = "your_tmdb_api_key_here" > .streamlit/secrets.toml

# Run the app
streamlit run main.py
```

> Want to retrain on a different dataset or tweak the feature engineering? Open `Movie-Recommender-System.ipynb`, run all cells, and it will regenerate `movies.pkl` + `similarity.pkl`.

---

## ✨ Features

- 🎯 **Content-based recommendations** — top 5 movies most similar to your pick, based on plot, genres, keywords, top cast, and director
- ⚡ **Instant results** — similarity matrix is precomputed offline and pickled, so the app does zero model inference at request time
- 🖼️ **Live poster fetching** straight from the TMDB API for every recommendation
- 🛡️ **Resilient networking** — automatic retries with exponential backoff, explicit handling for TMDB rate limits (HTTP 429) and connection resets, graceful placeholder fallback if a poster can't be fetched
- 🔍 **Searchable dropdown** of ~4,800 movies from the TMDB 5000 dataset
- 📐 **Dynamic layout** — result columns scale to however many recommendations come back, instead of a hardcoded grid

---

## 🎮 How to Use

### Step 1 — Pick a Movie
Choose any title from the **Choose a movie** dropdown.

### Step 2 — Hit Recommend
Click **`Recommend`**. The app looks up the movie's row in the precomputed similarity matrix, sorts every other movie by similarity score, and takes the top 5 (skipping the movie itself).

### Step 3 — Browse the Results
Each recommendation renders as a poster + title, fetched live from TMDB:

```
Selected: "The Dark Knight"
        ↓
[Poster] [Poster] [Poster] [Poster] [Poster]
 Movie A  Movie B  Movie C  Movie D  Movie E
```

If a poster fails to load (missing `poster_path`, rate limit, timeout), a placeholder image is shown instead so the UI never breaks.

---

## 🧠 Behind the Scenes: Recommendation Pipeline

```
TMDB 5000 Movies + Credits (CSV)
        │  merge on 'title'
        ▼
movie_id · title · overview · genres · keywords · cast · crew
        │  parse JSON-like columns (ast.literal_eval)
        ▼
genres   → list of genre names
keywords → list of keyword tags
cast     → top 3 billed actors
crew     → director only
        │  strip spaces, lowercase
        ▼
tags = overview + genres + keywords + cast + crew
        │  CountVectorizer(max_features=5000, stop_words='english')
        │  PorterStemmer for text normalization
        ▼
5000-dimension Bag-of-Words vectors (one per movie)
        │  cosine_similarity(vectors)
        ▼
4806 × 4806 similarity matrix
        │  pickle.dump
        ▼
movies.pkl + similarity.pkl  →  loaded once by main.py at app startup
        │
        ▼
User selects a movie → top-5 nearest neighbors + live TMDB posters
```

---

## 🧪 ML Concepts Demonstrated

This project covers concepts typically taught in an intro Data Science / ML course:

- Content-based filtering (as opposed to collaborative filtering)
- Feature engineering from nested/JSON-like dataframe columns
- Bag-of-Words text vectorization (`CountVectorizer`)
- Text normalization via stemming (`PorterStemmer`, NLTK)
- Cosine similarity for nearest-neighbor search
- Train-once, serve-via-pickle architecture — heavy computation happens offline in a notebook, the app only loads precomputed artifacts
- Defensive API integration — retries, exponential backoff, rate-limit handling, and graceful degradation

---

## 📊 Dataset

[**TMDB 5000 Movie Dataset**](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) (Kaggle) — two CSVs joined on `title`:

| File | Rows | Key columns used |
|---|---|---|
| `tmdb_5000_movies.csv` | 4,803 | `movie_id`, `title`, `overview`, `genres`, `keywords` |
| `tmdb_5000_credits.csv` | 4,803 | `cast`, `crew` |

After merging and dropping rows with missing overviews, **~4,806 movies** remain in the final recommendation set.

---

## 🗂️ Project Structure

```
CineMatch-AI/
│
├── Movie-Recommender-System.ipynb   # EDA, feature engineering & model-building notebook
├── main.py                          # Streamlit app — UI, recommendation lookup, poster fetching
├── movies.pkl                       # Serialized cleaned movie metadata (Git LFS)
├── similarity.pkl                   # Precomputed cosine similarity matrix (Git LFS)
├── tmdb_5000_movies.csv             # Raw TMDB movies dataset
├── tmdb_5000_credits.csv            # Raw TMDB credits dataset (cast & crew)
├── requirements.txt                 # Python dependencies
├── .gitattributes                   # Git LFS tracking rules for .pkl files
└── README.md
```

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| Python 3 | Core language |
| Pandas / NumPy | Data loading, cleaning, merging |
| scikit-learn (`CountVectorizer`, `cosine_similarity`) | Vectorization & similarity scoring |
| NLTK (`PorterStemmer`) | Text normalization / stemming |
| Streamlit | Web app UI, deployed on Streamlit Community Cloud |
| TMDB API | Live poster fetching |
| Pickle + Git LFS | Model artifact serialization & version-controlled storage |

---

## About the Author

**Rushit Tholiya**  
[LinkedIn Profile](https://www.linkedin.com/in/rushit-tholiya-605341311/)

[GitHub Profile](https://github.com/Rushit004)
