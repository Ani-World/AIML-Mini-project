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
    kmeans = joblib.load('person2/models/kmeans.joblib')
    scaler = joblib.load('person2/models/scaler.joblib')
    with open('person2/models/feature_cols.json', 'r') as f:
        feature_cols = json.load(f)
    name_feat = pd.read_csv('person2/models/name_features_mapping.csv')

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

    # Extract feature vector for the query movie and all movies
    X_all = name_feat[feature_cols].fillna(0).values.astype(float)
    X_all_scaled = scaler.transform(X_all)

    # compute distances of all movies to each centroid
    # kmeans.transform returns distances to each cluster center (shape: n_samples x n_clusters)
    dists_to_centroids = kmeans.transform(X_all_scaled)

    # cluster id for the query movie
    cluster_id = kmeans.predict(X_all_scaled[[idx]])[0]

    # distance of every movie to the cluster centroid we care about
    dist_to_our_centroid = dists_to_centroids[:, cluster_id]

    # build DataFrame with distances and filter to same cluster
    dfc = pd.DataFrame({
        'Name': name_feat['Name'],
        'Cluster': kmeans.predict(X_all_scaled),
        'DistanceToCentroid': dist_to_our_centroid
    })

    same_cluster = dfc[dfc['Cluster'] == cluster_id].copy()
    # exclude the movie itself
    same_cluster = same_cluster[same_cluster['Name'] != movie_name_exact]

    # sort ascending by distance (closest to centroid first)
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


r'''
Enter a movie name (partial allowed): dabang
How many recommendations do you want? (default 5): 3

Recommendations based on 'Dabangg':
1. Jurmana
2. Rakhwale
3. Aag Aur Chingari
PS C:\Users\VAISHNAVI AHIRE\OneDrive\Desktop\AIML-Mini-project> 







'''