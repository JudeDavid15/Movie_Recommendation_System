# Movie Recommender

A content-based movie recommendation system built with TF-IDF and cosine similarity. Select a movie and get 5 similar recommendations based on plot, genre, and crew.

## Live Demo
[Movie Recommender](https://movierecommendationsystem-p3muw2rldv8zv4sqgaqnhq.streamlit.app/)

## How It Works

1. Each movie's overview, genre, and crew are combined into a single text field.
2. `TfidfVectorizer` converts that text into a sparse feature matrix.
3. When a movie is selected, cosine similarity is computed between that movie's vector and all others, on the fly.
4. The top 5 most similar movies (excluding the selected movie itself) are shown.

## Dataset

[IMDB movies dataset](imdb_movies.csv) — includes title, genre, overview, and crew for each film.

## Key Findings

During development, two feature-engineering questions were tested:

- **Does removing `crew` from the feature text help or hurt?**
  Removing it hurt, recommendations were noticeably less relevant. Crew acts as a disambiguating signal (shared actors/directors across franchise entries) that plot and genre text alone don't fully capture.

- **CountVectorizer vs TF-IDF?**
  No meaningful difference in relevance, both surface nearly identical recommendations. TF-IDF was chosen as the more standard approach for text similarity tasks.

A notable engineering bug along the way: an early version of the "without crew" comparison used a dataset built by cross-referencing two separately-filtered DataFrames with mismatched indices, which silently paired the wrong plot text with the wrong movies. This initially made "without crew" look *better*, until the alignment bug was found and fixed, a good reminder to sanity-check intermediate data, not just final metrics.

## Running Locally

```bash
pip install -r requirements.txt
streamlit run movie.py
```

## Files

- `movie.py` : Streamlit app
- `movie.pkl` : preprocessed movie DataFrame (title, genre, overview, crew, combined details)
- `tfidf_matrix.pkl` : precomputed sparse TF-IDF matrix (similarity is computed on demand, not precomputed, to keep file size small)
- `movie_rec.ipynb` : full development notebook, including the CountVectorizer vs TF-IDF and with/without-crew comparisons

## Tech Stack

Python, pandas, scikit-learn (TfidfVectorizer, cosine_similarity), Streamlit
