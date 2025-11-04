# '''
# File : recommend_by_content.py
# Works: This code provides content-based movie recommendations using cosine similarity on genres: it asks the user for the dataset path and movie name, finds the most similar movies based on genre overlap, and saves both a CSV and a bar plot of the recommended movies in the plots folder.
# '''


# import pandas as pd
# import os
# from sklearn.metrics.pairwise import cosine_similarity
# import matplotlib.pyplot as plt

# # Ask user for dataset path
# DATA_PATH = input("Enter the full path of your preprocessed movies CSV: ")

# # Check if file exists
# if not os.path.isfile(DATA_PATH):
#     raise FileNotFoundError(f"No file found at '{DATA_PATH}'")

# # Load data
# df = pd.read_csv(DATA_PATH)

# # Select only genre columns (boolean columns)
# genre_cols = [col for col in df.columns if col.startswith('Genre_')]
# df_genres = df[genre_cols].astype(int)  # Ensure numeric

# # Compute cosine similarity between movies
# cos_sim = cosine_similarity(df_genres)

# # Function to get top N recommendations
# def recommend(movie_name, top_n=5):
#     if movie_name not in df['Name'].values:
#         print(f"Movie '{movie_name}' not found in dataset.")
#         return
#     idx = df.index[df['Name'] == movie_name][0]
#     sim_scores = list(enumerate(cos_sim[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     top_indices = [i[0] for i in sim_scores[1:top_n+1]]  # skip the movie itself
#     recommendations = df.iloc[top_indices][['Name', 'Year', 'Rating']]
#     return recommendations

# # Ask user for movie input
# movie_input = input("Enter the movie name for recommendations: ")
# recs = recommend(movie_input)

# if recs is not None:
#     # Ensure plots folder exists
#     os.makedirs('plots', exist_ok=True)
    
#     # Save recommendations to CSV
#     csv_path = os.path.join('plots', f'recommendations_for_{movie_input}.csv')
#     recs.to_csv(csv_path, index=False)
#     print(f"Recommendations for '{movie_input}' saved in {csv_path}.")

#     # Save a simple bar plot of Ratings
#     plt.figure(figsize=(8,4))
#     plt.barh(recs['Name'], recs['Rating'], color='skyblue')
#     plt.xlabel('Rating')
#     plt.title(f'Top {len(recs)} Recommendations for "{movie_input}"')
#     plt.gca().invert_yaxis()
#     png_path = os.path.join('plots', f'recommendations_for_{movie_input}.png')
#     plt.tight_layout()
#     plt.savefig(png_path)
#     plt.close()
#     print(f"Bar plot saved at {png_path}.")


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
    # create a filename-safe version of s
    s = s.strip()
    s = re.sub(r'[\\/:"*?<>|]+', '_', s)
    s = re.sub(r'\s+', '_', s)
    return s[:200]

def build_genres_from_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    If there's a 'Genre' column with comma separated genres, convert to Genre_<name> columns.
    """
    if 'Genre' not in df.columns:
        return df
    # build genre set
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
    # find genre_* columns
    genre_cols = [col for col in df.columns if col.startswith('Genre_')]
    if not genre_cols:
        # try to create from 'Genre' text column
        if 'Genre' in df.columns:
            print("No Genre_* columns found — building them from 'Genre' column.")
            df = build_genres_from_text(df)
            genre_cols = [col for col in df.columns if col.startswith('Genre_')]
        else:
            print("No Genre_* or 'Genre' column found. Will try text-based fallback (Plot/Description).")

    # If genre columns are available, use them
    if genre_cols:
        df_genres = df[genre_cols].fillna(0).astype(int)
        if df_genres.shape[1] == 0:
            raise ValueError("After processing, no genre features available.")
        feature_matrix = df_genres.values
        feature_type = 'genres'
        feature_cols_used = genre_cols
    else:
        # fallback to text features (Plot or Description)
        text_col = None
        for candidate in ('Plot', 'Description', 'Overview', 'Story'):
            if candidate in df.columns:
                text_col = candidate
                break
        if text_col is None:
            raise ValueError("No genre columns and no text column (Plot/Description) available to compute similarity.")
        print(f"Using text column '{text_col}' for content similarity (TF-IDF).")
        texts = df[text_col].fillna('').astype(str).tolist()
        vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        feature_matrix = vectorizer.fit_transform(texts).toarray()
        feature_type = 'text'
        feature_cols_used = [f"tfidf_{i}" for i in range(feature_matrix.shape[1])]
        # store vectorizer in df for future persisted use? (not saved here)
    return df, feature_matrix, feature_type, feature_cols_used

def compute_similarity_matrix(feature_matrix: np.ndarray):
    if feature_matrix is None or feature_matrix.shape[1] == 0:
        raise ValueError("No features available to compute similarity.")
    return cosine_similarity(feature_matrix)

def recommend(df: pd.DataFrame, cos_sim: np.ndarray, movie_name: str, top_n=5):
    # case-insensitive matching
    names = df['Name'].astype(str).tolist()
    # exact case-insensitive match first
    lowered = [n.lower() for n in names]
    movie_name_clean = movie_name.strip().lower()
    matches = [i for i, n in enumerate(lowered) if n == movie_name_clean]
    if not matches:
        # try contains
        matches = [i for i, n in enumerate(lowered) if movie_name_clean in n]
    if not matches:
        print(f"Movie '{movie_name}' not found (tried exact and substring match).")
        return None, None
    idx = matches[0]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # skip itself
    top_indices = [i for i,score in sim_scores if i != idx][:top_n]
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
        print("Error while preparing data:", e)
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

    # Prepare output directory
    out_dir = Path('plots')
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_name = safe_filename(movie_input)
    csv_path = out_dir / f"recommendations_for_{safe_name}.csv"
    recs.to_csv(csv_path, index=False)
    print(f"Recommendations saved to {csv_path}")

    # Bar plot if Rating exists
    if 'Rating' in recs.columns and not recs['Rating'].isna().all():
        plt.figure(figsize=(8, max(4, len(recs)*0.6)))
        plt.barh(recs['Name'], recs['Rating'])
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
