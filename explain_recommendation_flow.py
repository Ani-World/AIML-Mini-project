#!/usr/bin/env python3
"""
explain_recommendation_flow.py

Concise demo mode (new): prints a short, attractive summary for the top-N
recommendations and generates two small graphs saved to ./artifacts:
 - bar chart showing hybrid score components per recommended movie
 - pie chart showing relative contribution of regression/knn/apriori to each top recommendation

Usage examples:
  python explain_recommendation_flow.py            # default user 1, top 5
  python explain_recommendation_flow.py --user 1 --top 5
  python explain_recommendation_flow.py --user 1 --top 10 --verbose

This file keeps compatibility with your models when available but falls back to
lightweight demo mode if models or heavy packages (numpy/pandas) are missing.
"""

import os
import json
import ast
import argparse
import textwrap
from collections import defaultdict
from math import log1p
import math
from datetime import datetime

# plotting (matplotlib) - optional but included in lightweight manner
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# Try numpy/pandas; fall back if missing
try:
    import numpy as np
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False
    class SimpleNP:
        @staticmethod
        def zeros(n):
            return [0.0]*n
        @staticmethod
        def mean(arr, axis=0):
            return [sum(col)/len(col) for col in zip(*arr)] if arr else [0.0]
    np = SimpleNP()

# Paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(ROOT_DIR, "Backend")
MODEL_DIR = os.path.join(BACKEND_DIR, "models")
DATASETS_DIR = os.path.join(ROOT_DIR, "datasets")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Load lightweight demo movies (keeps consistent structure)
_movies_map = {}
movie_features = None
movie_id_to_index = {}

# Minimal demo set
DEMO = [
    (1, "The First Dawn", 2018, 110, 4.1, 1200, "Dir A", "Actor A1", "Actor A2", "Actor A3", 120),
    (2, "Silent Echoes", 2019, 95, 3.9, 950, "Dir B", "Actor B1", "Actor B2", "Actor B3", 95),
    (3, "Midnight Run", 2020, 125, 4.5, 2000, "Dir C", "Actor C1", "Actor C2", "Actor C3", 200),
    (4, "River of Stars", 2017, 105, 4.2, 1500, "Dir D", "Actor D1", "Actor D2", "Actor D3", 150),
    (5, "Lonely Planet", 2016, 100, 3.7, 800, "Dir E", "Actor E1", "Actor E2", "Actor E3", 80),
]
for t in DEMO:
    movie_id = t[0]
    _movies_map[movie_id] = {
        "movie_id": movie_id,
        "Name": t[1],
        "Year": t[2],
        "Duration": t[3],
        "Rating": float(t[4]),
        "Votes": int(t[5]),
        "Director": t[6],
        "Actor 1": t[7],
        "Actor 2": t[8],
        "Actor 3": t[9],
        "poster": None,
        "popularity": t[10] if len(t) > 10 else int(t[5])
    }

# synthetic features
genre_cols = ["Genre_Action", "Genre_Drama"]
movie_features_list = []
for i, mid in enumerate(sorted(_movies_map.keys())):
    movie_id_to_index[mid] = i
    v = [0.0]*len(genre_cols)
    v[i % len(genre_cols)] = 1.0
    year_norm = (_movies_map[mid]["Year"] - 2000) / 25.0
    dur_norm = _movies_map[mid]["Duration"] / 150.0
    rating_norm = _movies_map[mid]["Rating"] / 5.0
    votes_log = math.log1p(_movies_map[mid]["Votes"])
    fv = v + [year_norm, dur_norm, rating_norm, votes_log]
    movie_features_list.append(fv)
if PANDAS_AVAILABLE:
    movie_features = np.vstack(movie_features_list)
else:
    movie_features = movie_features_list

# Try to load lightweight models (but non-fatal)
kmeans_model = None
regressor_model = None
apriori_rules = {}
model_metadata = {}
try:
    import joblib
    if os.path.exists(os.path.join(MODEL_DIR, "kmeans_model.pkl")):
        kmeans_model = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    if os.path.exists(os.path.join(MODEL_DIR, "regressor_model.pkl")):
        regressor_model = joblib.load(os.path.join(MODEL_DIR, "regressor_model.pkl"))
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            model_metadata = json.load(f)
    apr_path = os.path.join(MODEL_DIR, "apriori_rules.json")
    if os.path.exists(apr_path):
        with open(apr_path, 'r', encoding='utf-8') as f:
            apriori_rules = json.load(f)
except Exception:
    # silent fallback — models are optional for attractive demo
    pass

# In-memory user ratings
rating_matrix = defaultdict(dict)
rating_matrix[1] = {1: 5.0, 3: 4.0}  # sample user who liked movie 1 and 3

# Utilities
def cosine_sim(a, b):
    try:
        if PANDAS_AVAILABLE:
            if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
                return 0.0
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        else:
            # simple vector dot/norm
            dot = sum(x*y for x,y in zip(a,b))
            na = math.sqrt(sum(x*x for x in a))
            nb = math.sqrt(sum(x*x for x in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot/(na*nb)
    except Exception:
        return 0.0

def get_user_profile(user_id):
    user_ratings = rating_matrix.get(user_id, {})
    liked = [mid for mid, r in user_ratings.items() if r >= 4]
    if not liked:
        return [0.0]*len(movie_features[0])
    idxs = [movie_id_to_index[mid] for mid in liked if mid in movie_id_to_index]
    vecs = [movie_features[i] for i in idxs]
    if PANDAS_AVAILABLE:
        return np.mean(np.vstack(vecs), axis=0)
    else:
        return list(map(lambda x: sum(x)/len(x), zip(*vecs)))

def get_knn_scores_for_user(user_id):
    up = get_user_profile(user_id)
    scores = {}
    for mid, idx in movie_id_to_index.items():
        mv = movie_features[idx]
        scores[mid] = cosine_sim(up, mv)
    return scores

def get_regressor_preds_for_user(user_id, mids):
    """Predict ratings for given movie ids. Handles model/feature-size mismatches gracefully.

    - If a trained regressor model is available and the feature dimension matches the model,
      predictions are returned.
    - If the regressor model is missing or the feature size doesn't match, the function
      falls back to returning the movie's average rating from _movies_map.
    """
    preds = {}
    # Fast fallback: if no model, use movie avg rating
    if regressor_model is None:
        for mid in mids:
            preds[mid] = float(_movies_map[mid]["Rating"])
        return preds

    # Build feature matrix for mids
    X = []
    valid_mids = []
    for mid in mids:
        if mid in movie_id_to_index:
            X.append(movie_features[movie_id_to_index[mid]])
            valid_mids.append(mid)
    if not X:
        return preds

    # Ensure X is a numpy array when possible
    try:
        if PANDAS_AVAILABLE:
            X = np.vstack(X)
        else:
            # convert to simple 2D list (scikit-learn accepts lists too)
            X = [list(x) for x in X]
    except Exception:
        # If stacking fails, fallback to avg ratings
        for mid in valid_mids:
            preds[mid] = float(_movies_map[mid]["Rating"])
        return preds

    # Check model expected feature size (n_features_in_ or coef_ shape)
    expected = None
    try:
        if hasattr(regressor_model, 'n_features_in_'):
            expected = int(regressor_model.n_features_in_)
        elif hasattr(regressor_model, 'coef_'):
            coef = getattr(regressor_model, 'coef_')
            # coef_ may be 1D or 2D
            if hasattr(coef, 'shape') and len(coef.shape) > 0:
                expected = int(coef.shape[-1])
    except Exception:
        expected = None

    if expected is not None:
        current = X.shape[1] if PANDAS_AVAILABLE else len(X[0])
        if current != expected:
            print(f"[WARN] regressor expects {expected} features but movie features have {current}. Falling back to average ratings.")
            for mid in valid_mids:
                preds[mid] = float(_movies_map[mid]["Rating"])
            return preds

    # Run prediction, with safety try/except
    try:
        preds_arr = regressor_model.predict(X)
        for i, mid in enumerate(valid_mids):
            preds[mid] = float(preds_arr[i])
    except Exception as e:
        print(f"[WARN] regressor prediction failed: {e}. Falling back to average ratings.")
        for mid in valid_mids:
            preds[mid] = float(_movies_map[mid]["Rating"])
    return preds


def apriori_boost_for_user(user_id, mids):
    boosts = {mid: 0.0 for mid in mids}
    reasons = defaultdict(list)
    user_ratings = rating_matrix.get(user_id, {})
    liked_mids = [mid for mid, r in user_ratings.items() if r >= 4]
    if not liked_mids or not apriori_rules:
        return boosts, reasons
    liked_items = set()
    for movie_id in liked_mids:
        m = _movies_map.get(movie_id)
        if not m: continue
        for a in ("Actor 1","Actor 2","Actor 3"):
            if m.get(a): liked_items.add(m[a].strip())
        if m.get("Director"): liked_items.add(m["Director"].strip())
    for rec_mid in mids:
        rec_key = str(rec_mid)
        for rule in apriori_rules.get(rec_key, []):
            for item in liked_items:
                if item in rule:
                    boosts[rec_mid] += 1.0
                    reasons[rec_mid].append(rule)
                    break
    maxb = max(boosts.values()) if boosts else 0.0
    if maxb>0:
        for k in boosts: boosts[k] = boosts[k]/maxb
    return boosts, reasons

# Compact, attractive terminal output + charts

def pretty_summary(user_id=1, top_n=5, verbose=False):
    print("--- Recommendation snapshot (concise) ---")
    user_ratings = rating_matrix.get(user_id, {})
    print(f"User {user_id}: rated {len(user_ratings)} movies; liked: {[mid for mid,r in user_ratings.items() if r>=4]}")

    rated = set(user_ratings.keys())
    candidate_mids = [mid for mid in movie_id_to_index.keys() if mid not in rated]
    if not candidate_mids:
        print("No candidates found.")
        return

    reg_preds = get_regressor_preds_for_user(user_id, candidate_mids)
    knn_scores = get_knn_scores_for_user(user_id)
    apr_boosts, apr_reasons = apriori_boost_for_user(user_id, candidate_mids)

    hybrid_w = model_metadata.get("hybrid_weights", {"regression":0.5, "knn":0.3, "apriori":0.2})

    rows = []
    for mid in candidate_mids:
        reg_score = reg_preds.get(mid, 0.0)/5.0
        knn_score = knn_scores.get(mid, 0.0)
        apr_score = apr_boosts.get(mid, 0.0)
        hybrid = hybrid_w.get("regression",0.5)*reg_score + hybrid_w.get("knn",0.3)*knn_score + hybrid_w.get("apriori",0.2)*apr_score
        rows.append((mid, _movies_map[mid]["Name"], hybrid, reg_score, knn_score, apr_score, apr_reasons.get(mid, []), _movies_map[mid].get("popularity",0)))

    rows_sorted = sorted(rows, key=lambda x: (x[2], x[7]), reverse=True)[:top_n]

    # Print compact table
    print("Top recommendations:")
    print(f"{'#':<3} {'Movie':<20} {'Hybrid':>7} {'Reg':>6} {'KNN':>6} {'APR':>6}")
    print('-'*60)
    for i, r in enumerate(rows_sorted, start=1):
        print(f"{i:<3} {r[1][:20]:<20} {r[2]:7.3f} {r[3]:6.3f} {r[4]:6.3f} {r[5]:6.3f}")
        if verbose and r[6]:
            print(f"     reasons: {r[6]}")
    print('')

    # Create charts if matplotlib available
    if MATPLOTLIB_AVAILABLE and rows_sorted:
        names = [r[1] for r in rows_sorted]
        regs = [r[3] for r in rows_sorted]
        knns = [r[4] for r in rows_sorted]
        aprs = [r[5] for r in rows_sorted]
        hybrids = [r[2] for r in rows_sorted]

        # stacked bar of components (regression, knn, apriori)
        fig, ax = plt.subplots(figsize=(8,4))
        x = range(len(names))
        ax.bar(x, regs, label='Regression')
        ax.bar(x, knns, bottom=regs, label='KNN')
        bottom = [r+ k for r,k in zip(regs, knns)]
        ax.bar(x, aprs, bottom=bottom, label='Apriori')
        ax.set_xticks(x)
        ax.set_xticklabels([n[:12] for n in names], rotation=25)
        ax.set_title('Hybrid score components (top {})'.format(len(names)))
        ax.legend()
        chart_path = os.path.join(ARTIFACTS_DIR, f"hybrid_components_{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
        fig.tight_layout()
        fig.savefig(chart_path)
        plt.close(fig)

        # Pie chart for first recommendation showing contribution
        fig2, ax2 = plt.subplots(figsize=(4,4))
        first = rows_sorted[0]
        parts = [first[3], first[4], first[5]]
        labels = ['Regression','KNN','Apriori']
        # normalize parts to sum>0
        s = sum(parts) if sum(parts)>0 else 1
        ax2.pie([p/s for p in parts], labels=labels, autopct='%1.1f%%')
        ax2.set_title(f'Contribution to: {first[1][:18]}')
        pie_path = os.path.join(ARTIFACTS_DIR, f"contrib_{user_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png")
        fig2.savefig(pie_path)
        plt.close(fig2)

        print(f"Charts saved to: {chart_path}")
        print(f"Pie chart saved to: {pie_path}")
    else:
        if not MATPLOTLIB_AVAILABLE:
            print("[INFO] matplotlib not available — skipping charts. Install matplotlib to enable charts.")

    # show reasons for top1 if available
    top1 = rows_sorted[0] if rows_sorted else None
    if top1 and top1[6]:
        print("Top-1 explanation:")
        for reason in top1[6][:3]:
            print(" - ", reason)
    print('--- End snapshot ---')

# CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--user', type=int, default=1)
    parser.add_argument('--top', type=int, default=5)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    pretty_summary(user_id=args.user, top_n=args.top, verbose=args.verbose)
