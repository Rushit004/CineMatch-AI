import streamlit as st
import pickle
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import logging
import os


from dotenv import load_dotenv


# ---------------- CONFIG ---------------- #
load_dotenv()

api_key = os.getenv("TMDB_API_KEY")



PLACEHOLDER = "https://via.placeholder.com/500x750?text=No+Image"
MAX_RETRIES = 3          # How many times to retry a failed request
RETRY_BACKOFF = 1.5      # Seconds to wait: 1.5s → 3s → 6s between retries
DELAY_BETWEEN_CALLS = 0.4  # Seconds between each poster fetch to avoid rate limiting

logging.basicConfig(level=logging.INFO)

# Proper browser-like headers so TMDB doesn't reject the request
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


def _build_session() -> requests.Session:
    """
    Create a requests Session with automatic retry logic.
    Retries on connection reset (10054), timeouts, and 5xx server errors.
    Uses exponential backoff so we don't hammer TMDB immediately.
    """
    session = requests.Session()
    session.headers.update(HEADERS)

    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF,         # waits 1.5s, 3s, 6s between attempts
        status_forcelist=[429, 500, 502, 503, 504],  # retry on these HTTP status codes
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# Single shared session for all requests (more efficient, respects keep-alive)
_session = _build_session()


# ---------------- FETCH POSTER ---------------- #
def fetch_poster(movie_id):
    """
    Fetch movie poster URL from TMDB.
    - Uses a persistent session with browser-like headers
    - Automatically retries up to 3 times with exponential backoff
    - Returns a placeholder image URL on any failure
    """
    url = (
        f"https://api.themoviedb.org/3/movie/{movie_id}"
        f"?api_key={api_key}&language=en-US"
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _session.get(url, timeout=8)

            if response.status_code == 429:
                # TMDB explicitly told us to slow down — wait longer
                wait = RETRY_BACKOFF * (2 ** attempt)
                logging.warning(f"Rate limited by TMDB (429). Waiting {wait}s before retry...")
                time.sleep(wait)
                continue

            if response.status_code != 200:
                logging.warning(
                    f"TMDB HTTP {response.status_code} for movie_id {movie_id} "
                    f"(attempt {attempt}/{MAX_RETRIES})"
                )
                return PLACEHOLDER

            data = response.json()
            poster_path = data.get('poster_path')

            if not poster_path:
                logging.info(f"No poster_path in TMDB response for movie_id {movie_id}")
                return PLACEHOLDER

            return "https://image.tmdb.org/t/p/w500/" + poster_path

        except requests.exceptions.ConnectionError as e:
            # This catches ConnectionResetError(10054) — TMDB closed the connection
            wait = RETRY_BACKOFF * (2 ** attempt)
            logging.warning(
                f"Connection reset for movie_id {movie_id} "
                f"(attempt {attempt}/{MAX_RETRIES}). Retrying in {wait}s... [{e}]"
            )
            time.sleep(wait)

        except requests.exceptions.Timeout:
            logging.warning(f"Timeout for movie_id {movie_id} (attempt {attempt}/{MAX_RETRIES})")
            time.sleep(RETRY_BACKOFF * attempt)

        except Exception as e:
            logging.error(f"Unexpected error fetching poster for movie_id {movie_id}: {e}")
            return PLACEHOLDER

    # All retries exhausted
    logging.error(f"All {MAX_RETRIES} retries failed for movie_id {movie_id}. Using placeholder.")
    return PLACEHOLDER


# ---------------- RECOMMEND FUNCTION ---------------- #
def recommend(movie):
    """Return top 5 recommended movie titles and their poster URLs."""

    # Look up index of the selected movie
    matched = movies[movies['title'] == movie].index

    if len(matched) == 0:
        return [], []

    movie_index = matched[0]
    distances = similarity[movie_index]

    # Sort by similarity score descending, skip index 0 (the movie itself)
    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []
    recommended_movies_poster = []

    for i in movies_list:
        try:
            movie_id = movies.iloc[i[0]]['movie_id']
            title = movies.iloc[i[0]]['title']

            recommended_movies.append(title)

            # FIX 3: Add delay to respect TMDB rate limits (you imported time — use it!)
            time.sleep(0.25)
            recommended_movies_poster.append(fetch_poster(movie_id))

        except Exception as e:
            # FIX 4: Log the actual error instead of swallowing it silently
            logging.error(f"Error processing recommendation at index {i[0]}: {e}")
            recommended_movies.append("Unknown")
            recommended_movies_poster.append(PLACEHOLDER)

    return recommended_movies, recommended_movies_poster


# ---------------- LOAD DATA ---------------- #
# FIX 5: Wrap pickle loads in try/except + use context managers (with open)
try:
    with open('movies.pkl', 'rb') as f:
        movies = pickle.load(f)
    with open('similarity.pkl', 'rb') as f:
        similarity = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"❌ Required data file not found: {e}")
    st.info("Make sure 'movies.pkl' and 'similarity.pkl' are in the same folder as app.py")
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to load data: {e}")
    st.stop()


# ---------------- UI ---------------- #
st.title('🎬 Movie Recommender System')

selected_movie_name = st.selectbox(
    'Choose a movie',
    movies['title'].values
)

if st.button('Recommend'):
    with st.spinner('Fetching recommendations...'):
        try:
            names, posters = recommend(selected_movie_name)

            if len(names) == 0:
                st.error("Movie not found in the dataset!")
            else:
                cols = st.columns(len(names))   # FIX 6: Dynamic columns, not hardcoded 5

                for i in range(len(names)):     # FIX 7: Iterate actual length, not range(5)
                    with cols[i]:
                        st.text(names[i])
                        # FIX 8: Never pass None to st.image()
                        st.image(posters[i] if posters[i] else PLACEHOLDER)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            logging.error(f"Recommendation error: {e}")