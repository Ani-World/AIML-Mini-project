"""
recommend_by_content.py
Robust content-based recommender:
 - prefers Genre_* boolean columns (if present)
 - if not found, will create Genre_* from a 'Genre' text column (comma-separated)
 - if still nothing, will fallback to using 'Plot' or 'Description' text with TF-IDF
Saves CSV + bar plot of top recommendations in ./plots
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from pathlib import Path

def safe_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r'[\\/:"*?<>|]+', '_', s)
    s = re.sub(r'\s+', '_', s)
    return s[:200]

def build_genres_from_text(df: pd.DataFrame) -> pd.DataFrame:
    if 'Genre' not in df.columns:
        return df
    genre_set = set()
    for v in df['Genre'].dropna().astype(str):
        genre_set.update([g.strip() for g in v.split(',') if g.strip()])
    genre_list = sorted(genre_set)
    for g in genre_list:
        col = f"Genre_{g}"
        df[col] = df['Genre'].fillna('').apply(lambda x: int(g in [gg.strip() for gg in x.split(',')]) if x else 0)
    return df

def load_and_prepare(data_path: str):
    df = pd.read_csv(data_path)
    if df.empty:
        raise ValueError("Loaded dataframe is empty. Check the CSV.")
    if 'Name' not in df.columns:
        raise ValueError("CSV must have a 'Name' column.")

    genre_cols = [col for col in df.columns if col.startswith('Genre_')]
    if not genre_cols:
        if 'Genre' in df.columns:
            print("No Genre_* columns found — building them from 'Genre' column.")
            df = build_genres_from_text(df)
            genre_cols = [col for col in df.columns if col.startswith('Genre_')]
        else:
            print("No Genre_* or 'Genre' column found. Will try text-based fallback (Plot/Description).")

    if genre_cols:
        df_genres = df[genre_cols].fillna(0).astype(int)
        if df_genres.shape[1] == 0:
            raise ValueError("After processing, no genre features available.")
        feature_matrix = df_genres.values
        feature_type = 'genres'
        feature_cols_used = genre_cols
    else:
        text_col = next((c for c in ('Plot','Description','Overview','Story') if c in df.columns), None)
        if text_col is None:
            raise ValueError("No genre columns and no text column (Plot/Description) available.")
        print(f"Using text column '{text_col}' for content similarity (TF-IDF).")
        texts = df[text_col].fillna('').astype(str).tolist()
        vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        feature_matrix = vectorizer.fit_transform(texts).toarray()
        feature_type = 'text'
        feature_cols_used = [f"tfidf_{i}" for i in range(feature_matrix.shape[1])]

    return df, feature_matrix, feature_type, feature_cols_used

def compute_similarity_matrix(feature_matrix: np.ndarray):
    if feature_matrix is None or feature_matrix.shape[1] == 0:
        raise ValueError("No features available to compute similarity.")
    return cosine_similarity(feature_matrix)

def recommend(df: pd.DataFrame, cos_sim: np.ndarray, movie_name: str, top_n=5):
    names = df['Name'].astype(str).tolist()
    lowered = [n.lower() for n in names]
    movie_name_clean = movie_name.strip().lower()
    matches = [i for i, n in enumerate(lowered) if n == movie_name_clean]
    if not matches:
        matches = [i for i, n in enumerate(lowered) if movie_name_clean in n]
    if not matches:
        print(f"Movie '{movie_name}' not found (exact or substring match).")
        return None, None
    idx = matches[0]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i,_ in sim_scores if i != idx][:top_n]
    recs = df.iloc[top_indices][['Name', 'Year'] + ([c for c in ['Rating'] if c in df.columns])]
    recs = recs.reset_index(drop=True)
    return idx, recs

def main():
    data_path = input("Enter the full path of your preprocessed movies CSV: ").strip()
    if not os.path.isfile(data_path):
        print(f"No file found at '{data_path}'.")
        return

    try:
        df, feature_matrix, feature_type, feature_cols_used = load_and_prepare(data_path)
    except Exception as e:
        print("Error preparing data:", e)
        return

    try:
        cos_sim = compute_similarity_matrix(feature_matrix)
    except Exception as e:
        print("Error computing similarity:", e)
        return

    movie_input = input("Enter the movie name for recommendations: ").strip()
    idx, recs = recommend(df, cos_sim, movie_input, top_n=5)
    if recs is None:
        return

    out_dir = Path('plots')
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = safe_filename(movie_input)
    csv_path = out_dir / f"recommendations_for_{safe_name}.csv"
    recs.to_csv(csv_path, index=False)
    print(f"Recommendations saved to {csv_path}")

    if 'Rating' in recs.columns and not recs['Rating'].isna().all():
        plt.figure(figsize=(8, max(4, len(recs)*0.6)))
        plt.barh(recs['Name'], recs['Rating'], color='skyblue')
        plt.xlabel('Rating')
        plt.title(f'Top {len(recs)} Recommendations for \"{movie_input}\"')
        plt.gca().invert_yaxis()
        png_path = out_dir / f"recommendations_for_{safe_name}.png"
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()
        print(f"Bar plot saved to {png_path}")
    else:
        print("Ratings column not available for plotting; saved CSV only.")

if __name__ == "__main__":
    main()
