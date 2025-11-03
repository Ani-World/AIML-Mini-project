# Backend/app.py
"""
Integrated Flask backend:
 - loads ML models from Backend/models/
 - uses database.py helpers if available (fallback to in-memory)
 - JWT auth for register/login
 - onboarding with dedupe
 - hybrid recommendations (regressor + knn + apriori + kmeans)
 - endpoints:
    POST /api/auth/register
    POST /api/auth/login
    GET  /api/movies/onboarding
    POST /api/onboarding
    POST /api/rate
    GET  /api/recommendations
    GET  /api/predict
    GET  /api/movie/<id>
    GET  /api/user/archive
    POST /api/feedback
    GET  /api/health
"""

import os
import time
import hashlib
import json
import ast
import requests
from functools import wraps
from collections import defaultdict
from math import log1p
from datetime import datetime, timedelta
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import numpy as np
import joblib

# --- Config ---
PORT = int(os.environ.get("PORT", 5000))
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# TMDB API Configuration
# Get your free API key from: https://www.themoviedb.org/settings/api
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
TMDB_API_BASE = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"

template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
app = Flask(__name__, template_folder=template_dir, static_folder="static")
CORS(app)

# Simple password hashing helper
def hash_password(password):
    """Simple password hashing (for demo - use bcrypt in production)"""
    return hashlib.sha256(password.encode()).hexdigest()

# -----------------------------
# Try to import database helpers (optional)
# -----------------------------
use_db = False
try:
    from database import (
        get_all_movies, get_movie, get_user_ratings, save_rating,
        get_all_movie_features, get_cluster_assignments, get_movies_for_onboarding,
        create_user, get_user_by_email
    )
    use_db = True
    app.logger.info("[DB] database.py loaded — using DB-backed storage")
except Exception as e:
    app.logger.warning(f"[DB] database.py not available or missing functions, falling back to in-memory demo. Error: {e}")
    use_db = False

# -----------------------------
# Load ML models and metadata
# -----------------------------
try:
    kmeans_model = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
    regressor_model = joblib.load(os.path.join(MODEL_DIR, "regressor_model.pkl"))
    # optional: apriori_rules.json (map movie_id(str) -> [rule strings])
    apriori_rules = {}
    apr_path = os.path.join(MODEL_DIR, "apriori_rules.json")
    if os.path.exists(apr_path):
        with open(apr_path, "r", encoding="utf-8") as f:
            apriori_rules = json.load(f)
    # model metadata (genre_cols, etc.)
    model_metadata = {}
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            model_metadata = json.load(f)
    app.logger.info("[MODELS] Loaded models successfully.")
except Exception as e:
    app.logger.error(f"[MODELS] Error loading models from {MODEL_DIR}: {e}")
    raise

# -----------------------------
# Load movies + features either from DB or fallback to demo dataset
# -----------------------------
if use_db:
    _movies_map = get_all_movies()
    cluster_assignments = get_cluster_assignments()
    # movie_features: ndarray (n_movies, f); movie_id_to_index: {movie_id: idx}
    movie_features, movie_id_to_index, normalization_params = get_all_movie_features(model_metadata.get("genre_cols", []))
else:
    # Demo fallback — small hardcoded dataset (keeps compatibility with earlier demo)
    _movies = [
        (1, "The First Dawn", 2018, 110, 4.1, 1200, "Dir A", "Actor A1", "Actor A2", "Actor A3", 120),
        (2, "Silent Echoes", 2019, 95, 3.9, 950, "Dir B", "Actor B1", "Actor B2", "Actor B3", 95),
        (3, "Midnight Run", 2020, 125, 4.5, 2000, "Dir C", "Actor C1", "Actor C2", "Actor C3", 200),
        (4, "River of Stars", 2017, 105, 4.2, 1500, "Dir D", "Actor D1", "Actor D2", "Actor D3", 150),
        (5, "Lonely Planet", 2016, 100, 3.7, 800, "Dir E", "Actor E1", "Actor E2", "Actor E3", 80),
    ]
    _movies_map = {}
    for t in _movies:
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
            "poster": f"https://via.placeholder.com/300x420?text={movie_id}",
            "popularity": t[10] if len(t) > 10 else int(t[5])
        }
    # build fake features (small)
    genre_cols = model_metadata.get("genre_cols", ["Genre_Action", "Genre_Drama"])
    movie_features_list = []
    movie_id_to_index = {}
    for i, mid in enumerate(sorted(_movies_map.keys())):
        movie_id_to_index[mid] = i
        v = np.zeros(len(genre_cols))
        v[i % len(genre_cols)] = 1.0
        year_norm = (_movies_map[mid]["Year"] - 2000) / 25.0
        dur_norm = _movies_map[mid]["Duration"] / 150.0
        rating_norm = _movies_map[mid]["Rating"] / 5.0
        votes_log = np.log1p(_movies_map[mid]["Votes"])
        fv = np.concatenate([v, [year_norm, dur_norm, rating_norm, votes_log]])
        movie_features_list.append(fv)
    movie_features = np.vstack(movie_features_list)
    normalization_params = {}

# -----------------------------
# In-memory caches and simple storage
# -----------------------------
rating_matrix = defaultdict(dict)  # user_id -> {movie_id: rating}

# Support DB-backed users if DB provides create_user/get_user_by_email
_users = {}           # email -> {user_id,email,name,password}  (fallback)
_next_user_id = 1     # fallback auto-increment

# Hybrid weights
hybrid_weights = model_metadata.get("hybrid_weights", {"regression": 0.5, "knn": 0.3, "apriori": 0.2})

# Onboarding dedupe cache
_last_onboarding_submission = {}
_ONBOARDING_DEDUPE_WINDOW = 8

# -----------------------------
# Utilities
# -----------------------------

def build_tmdb_poster_url(poster_path, size='w500'):
    """
    Build TMDB poster URL from poster path.
    Reference: https://developer.themoviedb.org/docs/image-basics
    
    Args:
        poster_path: TMDB poster path (e.g., '/1E5baAaEse26fej7uHcjOgEE2t2.jpg') or full URL
        size: Image size - 'w500', 'w780', or 'original'
    
    Returns:
        Full TMDB image URL or original URL if already a full URL
    """
    if not poster_path:
        return None
    
    # If it's already a full URL, return as is
    if poster_path.startswith('http://') or poster_path.startswith('https://'):
        return poster_path
    
    # If it's a TMDB path (starts with /), build the URL
    if poster_path.startswith('/'):
        return f"https://image.tmdb.org/t/p/{size}{poster_path}"
    
    # Otherwise, treat as relative path
    return f"https://image.tmdb.org/t/p/{size}/{poster_path}"

def movie_to_output(m):
    """Standardized movie JSON output"""
    return {
        "movie_id": m["movie_id"],
        "Name": m.get("Name"),
        "Year": int(m.get("Year", 0)),
        "Duration": int(m.get("Duration", 0)),
        "avg_rating": float(m.get("Rating", 0.0)),
        "votes": int(m.get("Votes", 0)),
        "director": m.get("Director"),
        "actor1": m.get("Actor 1"),
        "actor2": m.get("Actor 2"),
        "actor3": m.get("Actor 3"),
        "genre_vector": m.get("genre_vector") or m.get("Genre") or "{}",
        "poster": build_tmdb_poster_url(m.get("poster") or m.get("poster_url"), size='w500') or f"https://via.placeholder.com/300x420?text={m.get('movie_id')}",
        "popularity_score": round(m.get("popularity", 0) * (m.get("Rating", 0) / 5.0) * log1p(m.get("Votes", 0)), 3)
    }

def cosine_sim(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def get_user_profile(user_id):
    """Average feature vector of liked movies (rating >= 4)"""
    # Prefer DB-backed ratings if available
    if str(user_id) in rating_matrix:
        user_ratings = rating_matrix[user_id]
    else:
        try:
            user_ratings = get_user_ratings(user_id) if use_db else {}
        except Exception:
            user_ratings = {}
        rating_matrix[user_id] = user_ratings

    liked = [mid for mid, r in user_ratings.items() if r >= 4]
    if not liked:
        return np.zeros(movie_features.shape[1])
    idxs = [movie_id_to_index[mid] for mid in liked if mid in movie_id_to_index]
    if not idxs:
        return np.zeros(movie_features.shape[1])
    return np.mean(movie_features[idxs, :], axis=0)

def get_knn_scores_for_user(user_id):
    up = get_user_profile(user_id)
    scores = {}
    for mid, idx in movie_id_to_index.items():
        mv = movie_features[idx]
        scores[mid] = cosine_sim(up, mv)
    return scores

def get_regressor_preds_for_user(user_id, mids):
    """
    Predict ratings using the regressor model.
    The regressor was trained on movie features only (not user profile).
    """
    X = []
    valid_mids = []
    for mid in mids:
        if mid in movie_id_to_index:
            idx = movie_id_to_index[mid]
            mv = movie_features[idx]  # Use only movie features (26 features)
            X.append(mv)
            valid_mids.append(mid)
    if not X:
        return {}
    X = np.vstack(X)
    preds = regressor_model.predict(X)
    return {mid: float(preds[i]) for i, mid in enumerate(valid_mids)}

def apriori_boost_for_user(user_id, mids):
    boosts = {mid: 0.0 for mid in mids}
    reasons = defaultdict(list)
    # get liked items
    try:
        user_ratings = rating_matrix[user_id] if user_id in rating_matrix else (get_user_ratings(user_id) if use_db else {})
    except:
        user_ratings = {}
    liked_mids = [mid for mid, r in (user_ratings or {}).items() if r >= 4]
    if not liked_mids:
        return boosts, reasons
    liked_items = set()
    for movie_id in liked_mids:
        m = _movies_map.get(movie_id)
        if not m:
            continue
        for a in ("Actor 1", "Actor 2", "Actor 3"):
            if m.get(a):
                liked_items.add(m[a].strip())
        if m.get("Director"):
            liked_items.add(m["Director"].strip())
        # genre_vector might be stored as JSON string or dict
        gv = m.get("genre_vector") or m.get("Genre")
        if gv:
            try:
                gdict = ast.literal_eval(gv) if isinstance(gv, str) else gv
                if isinstance(gdict, dict):
                    liked_items.update(gdict.keys())
            except:
                pass
    # apply apriori rules (rules are keyed by str(movie_id) -> list[str rules])
    for rec_mid in mids:
        rec_key = str(rec_mid)
        rules = apriori_rules.get(rec_key, [])
        for rule in rules:
            for item in liked_items:
                if item in rule:
                    boosts[rec_mid] += 1.0
                    reasons[rec_mid].append(rule)
                    break
    maxb = max(boosts.values()) if boosts else 0.0
    if maxb > 0:
        for k in boosts:
            boosts[k] = boosts[k] / maxb
    return boosts, reasons

# -----------------------------
# Auth endpoints (register/login)
# -----------------------------
@app.route('/api/auth/register', methods=['POST'])
def register():
    global _next_user_id
    data = request.get_json(force=True) or {}
    email = (data.get('email') or '').strip().lower()
    pw = data.get('password') or ''
    name = data.get('name') or ''
    
    if not email or not pw:
        return jsonify({"error": "email and password required"}), 400

    # Use database if available
    if use_db:
        try:
            existing = get_user_by_email(email)
            if existing:
                return jsonify({"error": "user already exists"}), 409
            # Hash password before storing
            password_hash = hash_password(pw)
            user_id = create_user(name, email, password_hash)
            return jsonify({"message": "registered", "user_id": user_id}), 201
        except Exception as e:
            app.logger.error(f"Registration error: {e}")
            return jsonify({"error": "registration failed"}), 500

    # Fallback in-memory
    if email in _users:
        return jsonify({"error": "user already exists"}), 409
    uid = _next_user_id
    _next_user_id += 1
    _users[email] = {"user_id": uid, "email": email, "name": name, "password": pw}
    return jsonify({"message": "registered", "user_id": uid}), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.get_json(force=True) or {}
    email = (data.get('email') or '').strip().lower()
    pw = data.get('password') or ''
    
    if not email or not pw:
        return jsonify({"error": "email and password required"}), 400

    # Use database if available
    if use_db:
        try:
            user = get_user_by_email(email)
            if not user:
                return jsonify({"error": "invalid credentials"}), 401
            # Compare hashed password
            password_hash = hash_password(pw)
            if user.get("password_hash") != password_hash:
                return jsonify({"error": "invalid credentials"}), 401
            return jsonify({"message": "ok", "user_id": user["user_id"]}), 200
        except Exception as e:
            app.logger.error(f"Login error: {e}")
            return jsonify({"error": "login failed"}), 500

    # Fallback in-memory
    user = _users.get(email)
    if not user or user.get("password") != pw:
        return jsonify({"error": "invalid credentials"}), 401
    return jsonify({"message": "ok", "user_id": user["user_id"]}), 200

# -----------------------------
# Movies for onboarding
# -----------------------------
@app.route('/api/movies/onboarding', methods=['GET'])
def movies_onboarding():
    # use DB helper if available to fetch curated list
    try:
        if use_db and callable(globals().get("get_movies_for_onboarding", None)):
            movies = get_movies_for_onboarding(limit=50)
            return jsonify({"movies": movies}), 200
    except Exception:
        pass
    movies = [{"movie_id": m["movie_id"], "name": m["Name"], "year": m["Year"], "poster": m.get("poster")} for m in _movies_map.values()]
    return jsonify({"movies": movies}), 200

# -----------------------------
# Onboarding with dedupe
# -----------------------------
@app.route('/api/onboarding', methods=['POST'])
def onboarding():
    global _last_onboarding_submission
    data = request.get_json(force=True) or {}
    user_id = data.get('user_id')
    responses = data.get('responses') or []
    if not user_id or not isinstance(responses, list):
        return jsonify({"error": "user_id and responses required"}), 400

    # deterministic hash
    try:
        normalized = {"user_id": int(user_id), "responses": sorted(
            [{"movie_id": int(r.get("movie_id")), "like": int(r.get("like"))} for r in responses],
            key=lambda x: (x["movie_id"], x["like"])
        )}
        payload_bytes = json.dumps(normalized, separators=(',', ':'), sort_keys=True).encode('utf-8')
    except Exception:
        payload_bytes = json.dumps(data, separators=(',', ':'), sort_keys=True).encode('utf-8')

    payload_hash = hashlib.sha1(payload_bytes).hexdigest()
    now = time.time()
    last = _last_onboarding_submission.get(str(user_id))
    if last:
        last_hash, last_ts, last_resp = last
        if payload_hash == last_hash and (now - last_ts) <= _ONBOARDING_DEDUPE_WINDOW:
            app.logger.info(f"onboarding: duplicate submission detected for user {user_id}, returning cached response")
            return jsonify(last_resp), 200

    saved = 0
    for r in responses:
        try:
            movie_id = int(r.get('movie_id'))
            like_val = int(r.get('like'))
            if like_val not in (-1, 0, 1):
                continue
        except Exception:
            continue
        pseudo_rating = 5.0 if like_val == 1 else (3.0 if like_val == 0 else 1.0)
        # DB-backed save if available
        try:
            if use_db and callable(globals().get("save_rating", None)):
                save_rating(user_id, movie_id, pseudo_rating)
            rating_matrix[user_id][movie_id] = pseudo_rating
        except Exception:
            rating_matrix[user_id][movie_id] = pseudo_rating
        saved += 1

    # Simple popularity-based fallback recommendations
    rated_movie_ids = set(rating_matrix[user_id].keys())
    all_movies = list(_movies_map.values())
    recs = [m for m in sorted(all_movies, key=lambda x: (x.get("popularity",0), x.get("Rating",0)), reverse=True) if m["movie_id"] not in rated_movie_ids][:10]
    out = [{"movie_id": m["movie_id"], "name": m["Name"], "year": m["Year"], "poster": m.get("poster")} for m in recs]
    response_body = {"message": f"saved {saved} responses", "recommendations": out}
    _last_onboarding_submission[str(user_id)] = (payload_hash, now, response_body)
    app.logger.info(f"onboarding: saved {saved} responses for user {user_id}")
    return jsonify(response_body), 200

# -----------------------------
# Rate
# -----------------------------
@app.route('/api/rate', methods=['POST'])
def api_rate():
    data = request.get_json(force=True) or {}
    user_id = data.get('user_id'); movie_id = data.get('movie_id'); rating = data.get('rating')
    if user_id is None or movie_id is None or rating is None:
        return jsonify({"error":"user_id, movie_id, rating required"}), 400
    try:
        user_id = int(user_id); movie_id = int(movie_id); rating = float(rating)
    except:
        return jsonify({"error":"invalid types"}), 400
    try:
        if use_db and callable(globals().get("save_rating", None)):
            save_rating(user_id, movie_id, rating)
        rating_matrix[user_id][movie_id] = rating
    except Exception as e:
        rating_matrix[user_id][movie_id] = rating
    return jsonify({"message":"rating saved"}), 201

# -----------------------------
# Recommendations (hybrid)
# -----------------------------
@app.route('/api/recommendations', methods=['GET'])
def api_recommendations():
    user_id = request.args.get('user_id'); n = int(request.args.get('n', 20))
    force_refresh = request.args.get('refresh', 'false').lower() == 'true'
    
    if user_id is None:
        return jsonify({"error":"user_id required"}), 400
    try:
        user_id = int(user_id)
    except:
        return jsonify({"error":"user_id must be int-like"}), 400

    # Force refresh from DB if requested (useful after onboarding)
    if force_refresh or user_id not in rating_matrix:
        try:
            rating_matrix[user_id] = get_user_ratings(user_id) if use_db else {}
        except:
            rating_matrix[user_id] = {}

    rated = set(rating_matrix[user_id].keys())
    candidate_mids = [mid for mid in movie_id_to_index.keys() if mid not in rated]
    if not candidate_mids:
        return jsonify({"recommendations": []}), 200

    app.logger.info("Generating recommendations using: KMeans, Regressor, KNN, Apriori")
    reg_preds = get_regressor_preds_for_user(user_id, candidate_mids)
    knn_scores = get_knn_scores_for_user(user_id)
    apr_boosts, apr_reasons = apriori_boost_for_user(user_id, candidate_mids)

    items = []
    for mid in candidate_mids:
        reg_score = reg_preds.get(mid, 0.0) / 5.0
        knn_score = knn_scores.get(mid, 0.0)
        apr_score = apr_boosts.get(mid, 0.0)
        hybrid_score = (hybrid_weights.get("regression",0.5) * reg_score +
                        hybrid_weights.get("knn",0.3) * knn_score +
                        hybrid_weights.get("apriori",0.2) * apr_score)
        m = _movies_map.get(mid, {})
        out = movie_to_output(m)
        out["cluster_id"] = cluster_assignments.get(mid) if 'cluster_assignments' in globals() else None
        out["hybrid_score"] = round(float(hybrid_score), 4)
        out["reg_score"] = round(float(reg_score),4)
        out["knn_score"] = round(float(knn_score),4)
        out["apr_score"] = round(float(apr_score),4)
        out["apriori_reasons"] = apr_reasons.get(mid, [])
        items.append(out)

    items_sorted = sorted(items, key=lambda x: (x["hybrid_score"], x["popularity_score"]), reverse=True)
    return jsonify({"recommendations": items_sorted[:n]}), 200

# -----------------------------
# Predict
# -----------------------------
@app.route('/api/predict', methods=['GET'])
def api_predict():
    user_id = request.args.get('user_id'); movie_id = request.args.get('movie_id')
    if user_id is None or movie_id is None:
        return jsonify({"error":"user_id and movie_id required"}), 400
    try:
        user_id = int(user_id); movie_id = int(movie_id)
    except:
        return jsonify({"error":"user_id and movie_id must be ints"}), 400
    if movie_id not in _movies_map:
        return jsonify({"error":"movie_id not found"}), 404
    if user_id not in rating_matrix:
        try:
            rating_matrix[user_id] = get_user_ratings(user_id) if use_db else {}
        except:
            rating_matrix[user_id] = {}
    preds = get_regressor_preds_for_user(user_id, [movie_id])
    pred = preds.get(movie_id, _movies_map[movie_id]["Rating"])
    num_user_ratings = len(rating_matrix[user_id])
    confidence = min(0.95, 0.2 + 0.05 * num_user_ratings)
    return jsonify({
        "Name": _movies_map[movie_id]["Name"],
        "movie_id": movie_id,
        "predicted_rating": round(float(pred), 3),
        "confidence": round(float(confidence), 3),
        "source": "hybrid_regressor"
    }), 200

# -----------------------------
# Movie metadata
# -----------------------------
@app.route('/api/movie/<int:movie_id>', methods=['GET'])
def api_movie(movie_id):
    if movie_id not in _movies_map:
        return jsonify({"error":"movie not found"}), 404
    m = _movies_map[movie_id]
    out = movie_to_output(m)
    
    # Use larger poster size for detail page (w780 instead of w500)
    # Get raw poster_url from database to build larger URL
    if use_db:
        try:
            from database import get_connection
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT poster_url FROM movies WHERE movie_id = ?", (movie_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row and 'poster_url' in row.keys() and row['poster_url']:
                # Build larger poster URL from original path
                larger_poster = build_tmdb_poster_url(row['poster_url'], size='w780')
                if larger_poster:
                    out["poster"] = larger_poster
            elif out.get("poster") and "image.tmdb.org" in out.get("poster", ""):
                # Upgrade existing TMDB URL to larger size
                existing_poster = out["poster"]
                if "/w500/" in existing_poster:
                    out["poster"] = existing_poster.replace("/w500/", "/w780/")
        except Exception as e:
            app.logger.debug(f"Could not fetch poster_path for movie {movie_id}: {e}")
    
    out["cluster_id"] = cluster_assignments.get(movie_id) if 'cluster_assignments' in globals() else None
    out["apriori_reasons"] = apriori_rules.get(str(movie_id), [])
    return jsonify(out), 200

# -----------------------------
# Archive (user history)
# -----------------------------
@app.route('/api/user/archive', methods=['GET'])
def api_archive():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error":"user_id required"}), 400
    try:
        user_id = int(user_id)
    except:
        return jsonify({"error":"user_id must be int-like"}), 400
    if user_id in rating_matrix:
        user_ratings = rating_matrix[user_id]
    else:
        try:
            user_ratings = get_user_ratings(user_id) if use_db else {}
        except:
            user_ratings = {}
        rating_matrix[user_id] = user_ratings
    out = []
    for mid, r in user_ratings.items():
        m = _movies_map.get(mid)
        if m:
            o = movie_to_output(m)
            o["user_rating"] = r
            out.append(o)
    return jsonify({"archive": out}), 200

# -----------------------------
# Analytics endpoint (watch time, genre distribution, etc.)
# -----------------------------
@app.route('/api/user/analytics', methods=['GET'])
def api_analytics():
    """Get user analytics: watch time, genre distribution, etc."""
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id required"}), 400
    try:
        user_id = int(user_id)
    except:
        return jsonify({"error": "user_id must be int-like"}), 400
    
    # Get user ratings
    if user_id in rating_matrix:
        user_ratings = rating_matrix[user_id]
    else:
        try:
            user_ratings = get_user_ratings(user_id) if use_db else {}
        except:
            user_ratings = {}
        rating_matrix[user_id] = user_ratings
    
    # Calculate analytics
    total_watch_time = 0
    genre_counts = defaultdict(int)
    year_counts = defaultdict(int)
    rating_distribution = defaultdict(int)
    
    for mid, rating in user_ratings.items():
        m = _movies_map.get(mid)
        if m:
            # Watch time (only for rated >= 4 as "watched")
            if rating >= 4:
                total_watch_time += m.get("Duration", 0)
            
            # Genre distribution
            try:
                genre_dict = ast.literal_eval(m.get("Genre", "{}"))
                if isinstance(genre_dict, dict):
                    for genre in genre_dict.keys():
                        genre_counts[genre] += 1
            except:
                pass
            
            # Year distribution
            year = m.get("Year")
            if year:
                year_counts[int(year)] += 1
            
            # Rating distribution
            rating_bucket = int(rating)
            rating_distribution[rating_bucket] += 1
    
    # Format genre data for charts
    genre_list = [{"genre": k, "count": v} for k, v in sorted(genre_counts.items(), key=lambda x: -x[1])]
    
    return jsonify({
        "total_watch_time": total_watch_time,  # minutes
        "total_watch_time_hours": round(total_watch_time / 60, 1),
        "movies_rated": len(user_ratings),
        "genre_distribution": genre_list,
        "year_distribution": dict(year_counts),
        "rating_distribution": dict(rating_distribution),
        "average_rating": round(sum(user_ratings.values()) / len(user_ratings), 2) if user_ratings else 0
    }), 200

# -----------------------------
# Feedback
# -----------------------------
@app.route('/api/feedback', methods=['POST'])
def api_feedback():
    data = request.get_json(force=True) or {}
    user_id = data.get('user_id'); movie_id = data.get('movie_id'); feedback_type = data.get('feedback_type','unknown')
    if not user_id:
        return jsonify({"error":"user_id required"}), 400
    try:
        user_id = int(user_id)
    except:
        return jsonify({"error":"user_id must be int-like"}), 400
    app.logger.info(f"Feedback from user {user_id} on movie {movie_id}: {feedback_type}")
    # TODO: store feedback in DB table user_feedback if desired
    return jsonify({"message":"feedback saved"}), 201

# -----------------------------
# Health
# -----------------------------
@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status":"ok", "movies_loaded": len(_movies_map)}), 200

# -----------------------------
# Optionally render UI pages if templates exist
# -----------------------------
# @app.route('/')
# def index():
#     # If you have templates/dashboard.html etc, this will render - otherwise it's ok to keep as JSON-only API
#     try:
#         return render_template('index.html')
#     except Exception:
#         return jsonify({"message":"Archivr backend running"}), 200
# ------------------- Template Routes -------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/login")
def login_ui():
    return render_template("login.html")

@app.route("/register")
def register_ui():
    return render_template("register.html")

@app.route("/dashboard")
def dashboard_ui():
    return render_template("dashboard.html")

@app.route("/archive")
def archive_ui():
    return render_template("archive.html")

@app.route("/movie/<int:movie_id>")
def movie_detail_ui(movie_id):
    """Render movie detail page"""
    # Verify movie exists
    if movie_id not in _movies_map:
        return render_template("movie_detail.html", movie_id=None, error="Movie not found"), 404
    return render_template("movie_detail.html", movie_id=movie_id)

@app.route("/onboarding")
def onboarding_ui():
    return render_template("onboarding.html")

# -----------------------------
# Run
# -----------------------------
if __name__ == '__main__':
    app.logger.info("Starting backend on port %d", PORT)
    app.run(host='127.0.0.1', port=PORT, debug=True)
