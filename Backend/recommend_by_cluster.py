# recommend_by_cluster.py (rank by distance to centroid)
import pandas as pd
import numpy as np
import joblib
import json
import os
import re

def safe_lower(s): 
    return s.strip().lower()

def recommend_movies_by_cluster(movie_query, top_n=5):
    # Load artifacts
    kmeans_path = 'person2/models/kmeans.joblib'
    scaler_path = 'person2/models/scaler.joblib'
    feature_cols_path = 'person2/models/feature_cols.json'
    name_feat_path = 'person2/models/name_features_mapping.csv'

    for path in [kmeans_path, scaler_path, feature_cols_path, name_feat_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")

    kmeans = joblib.load(kmeans_path)
    scaler = joblib.load(scaler_path)
    with open(feature_cols_path, 'r') as f:
        feature_cols = json.load(f)
    name_feat = pd.read_csv(name_feat_path)

    # Find matches (case-insensitive substring)
    name_series = name_feat['Name'].astype(str)
    lowered = name_series.str.lower()
    q = movie_query.strip().lower()
    matches_idx = lowered[lowered.str.contains(q)].index.tolist()

    if not matches_idx:
        print(f"No movies found matching '{movie_query}'")
        return None, []

    # take first match
    idx = matches_idx[0]
    movie_name_exact = name_feat.loc[idx, 'Name']

    # Extract feature vector for all movies
    X_all = name_feat[feature_cols].fillna(0).values.astype(float)
    X_all_scaled = scaler.transform(X_all)

    # Compute distances to all centroids
    dists_to_centroids = kmeans.transform(X_all_scaled)

    # Cluster ID for the query movie
    cluster_id = kmeans.predict(X_all_scaled[[idx]])[0]

    # Distance to the cluster centroid of interest
    dist_to_our_centroid = dists_to_centroids[:, cluster_id]

    # Build DataFrame with distances
    dfc = pd.DataFrame({
        'Name': name_feat['Name'],
        'Cluster': kmeans.predict(X_all_scaled),
        'DistanceToCentroid': dist_to_our_centroid
    })

    same_cluster = dfc[dfc['Cluster'] == cluster_id].copy()
    same_cluster = same_cluster[same_cluster['Name'] != movie_name_exact]
    same_cluster = same_cluster.sort_values('DistanceToCentroid', ascending=True)

    top = same_cluster.head(top_n)['Name'].tolist()
    return movie_name_exact, top

if __name__ == "__main__":
    q = input("Enter a movie name (partial allowed): ").strip()
    try:
        n = int(input("How many recommendations do you want? (default 5): ").strip() or 5)
    except:
        n = 5

    exact, recs = recommend_movies_by_cluster(q, top_n=n)
    if recs:
        print(f"\nRecommendations based on '{exact}':")
        for i, r in enumerate(recs, 1):
            print(f"{i}. {r}")
    else:
        print("No recommendations found.")
