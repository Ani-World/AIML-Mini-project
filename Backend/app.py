"""
Updated Flask backend with better ML model integration
"""
import os, time, hashlib, json
from collections import defaultdict
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS


# Import database and ML models
import db
from ml_models import HybridRecommender

app = Flask(__name__)
CORS(app)

# -----------------------------
# Global variables
# -----------------------------
genre_cols = []
movies = {}
rating_matrix = defaultdict(dict)
hybrid_recommender = HybridRecommender()

# -----------------------------
# Load dataset and initialize models
# -----------------------------
def load_dataset_and_initialize():
    """Load dataset and initialize ML models"""
    global genre_cols, movies, hybrid_recommender
    
    print("Loading dataset and initializing models...")
    
    # Load movies from database
    movies = db.get_all_movies()
    if not movies:
        print("Warning: No movies found in database")
        return
    
    print(f"Loaded {len(movies)} movies from database")
    
    # Extract genre columns from the first movie's genre vector
    if movies:
        sample_movie = next(iter(movies.values()))
        try:
            genre_dict = json.loads(sample_movie.get('Genre', '{}'))
            genre_cols = list(genre_dict.keys())
            print(f"Found {len(genre_cols)} genre columns")
        except Exception as e:
            print(f"Error parsing genre columns: {e}")
            genre_cols = []
    
    # Get feature matrix for all movies
    feature_matrix, movie_id_to_index, norms = db.get_all_movie_features(genre_cols)
    movie_ids = sorted(movies.keys())
    
    print(f"Feature matrix shape: {feature_matrix.shape}")
    
    # Initialize hybrid recommender
    try:
        hybrid_recommender.initialize_models(
            movies=movies,
            ratings=rating_matrix,
            genre_cols=genre_cols,
            feature_matrix=feature_matrix,
            movie_ids=movie_ids
        )
    except Exception as e:
        print(f"Error initializing ML models: {e}")

# -----------------------------
# Authentication
# -----------------------------
_users = {}
_next_user_id = 1

@app.route('/api/auth/register', methods=['POST'])
def register():
    global _next_user_id
    data = request.get_json(force=True) or {}
    email = (data.get('email') or '').strip().lower()
    pw = data.get('password') or ''
    name = data.get('name') or ''
    
    if not email or not pw: 
        return jsonify({"error": "email and password required"}), 400
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
    
    user = _users.get(email)
    if not user or user.get("password") != pw: 
        return jsonify({"error": "invalid credentials"}), 401
    
    return jsonify({"message": "ok", "user_id": user["user_id"]}), 200

# -----------------------------
# Movies for onboarding
# -----------------------------
@app.route('/api/movies/onboarding', methods=['GET'])
def movies_onboarding():
    movies_list = db.get_movies_for_onboarding(limit=50)
    return jsonify({"movies": movies_list}), 200

# -----------------------------
# Onboarding
# -----------------------------
_last_onboarding_submission = {}
_ONBOARDING_DEDUPE_WINDOW = 8

@app.route('/api/onboarding', methods=['POST'])
def onboarding():
    global _last_onboarding_submission, rating_matrix
    
    data = request.get_json(force=True) or {}
    user_id = data.get('user_id')
    responses = data.get('responses') or []

    if not user_id or not isinstance(responses, list):
        return jsonify({"error": "user_id and responses required"}), 400

    # Deduplication check
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

    last = _last_onboarding_submission.get(user_id)
    if last:
        last_hash, last_ts, last_resp = last
        if payload_hash == last_hash and (now - last_ts) <= _ONBOARDING_DEDUPE_WINDOW:
            return jsonify(last_resp), 200

    # Save ratings
    saved = 0
    for r in responses:
        try:
            movie_id = int(r.get('movie_id'))
            like_val = int(r.get('like'))
            if like_val not in (-1, 0, 1):
                continue
        except Exception:
            continue
        
        # Map like to pseudo-rating
        pseudo_rating = 5.0 if like_val == 1 else (3.0 if like_val == 0 else 1.0)
        rating_matrix[user_id][movie_id] = pseudo_rating
        saved += 1

    # Get recommendations
    try:
        feature_matrix, movie_id_to_index, norms = db.get_all_movie_features(genre_cols)
        movie_ids = sorted(movies.keys())
        
        recommendations = hybrid_recommender.get_recommendations(
            user_id=int(user_id),
            ratings=rating_matrix,
            movies=movies,
            feature_matrix=feature_matrix,
            movie_ids=movie_ids,
            top_k=10
        )
        
        # Format recommendations for frontend
        recs_formatted = []
        for rec in recommendations:
            recs_formatted.append({
                "movie_id": rec["movie_id"],
                "name": rec["Name"],
                "year": rec["Year"],
                "poster": rec.get("poster", f"https://via.placeholder.com/300x420?text={rec['movie_id']}"),
                "hybrid_score": rec.get("hybrid_score", 0)
            })
        
        response_body = {
            "message": f"saved {saved} responses", 
            "recommendations": recs_formatted
        }
        
    except Exception as e:
        print(f"Error getting ML recommendations: {e}")
        # Fallback to popularity-based recommendations
        rated_movie_ids = set(rating_matrix[user_id].keys())
        recs = [m for m in sorted(movies.values(), 
                                key=lambda x: x.get("Votes", 0) * x.get("Rating", 0), 
                                reverse=True) 
                if m["movie_id"] not in rated_movie_ids][:10]
        recs_formatted = [{
            "movie_id": m["movie_id"],
            "name": m["Name"],
            "year": m["Year"],
            "poster": m.get("poster")
        } for m in recs]
        
        response_body = {
            "message": f"saved {saved} responses (fallback)", 
            "recommendations": recs_formatted
        }

    # Store dedupe cache
    _last_onboarding_submission[user_id] = (payload_hash, now, response_body)
    
    return jsonify(response_body), 200

# -----------------------------
# Rate endpoint
# -----------------------------
@app.route('/api/rate', methods=['POST'])
def api_rate():
    global rating_matrix
    
    data = request.get_json(force=True) or {}
    user_id = data.get('user_id')
    movie_id = data.get('movie_id')
    rating = data.get('rating')
    
    if user_id is None or movie_id is None or rating is None:
        return jsonify({"error": "user_id, movie_id, rating required"}), 400
    
    try:
        user_id = int(user_id)
        movie_id = int(movie_id)
        rating = float(rating)
    except:
        return jsonify({"error": "invalid types"}), 400
    
    rating_matrix[user_id][movie_id] = rating
    
    return jsonify({"message": "rating saved"}), 201

# -----------------------------
# ML-Powered Recommendations
# -----------------------------
@app.route('/api/recommendations', methods=['GET'])
def api_recommendations():
    user_id = request.args.get('user_id')
    n = int(request.args.get('n', 20))
    
    if user_id is None:
        return jsonify({"error": "user_id required"}), 400
    
    try:
        user_id = int(user_id)
    except:
        return jsonify({"error": "user_id must be int-like"}), 400

    print(f"Getting recommendations for user {user_id}")
    
    try:
        # Get feature matrix
        feature_matrix, movie_id_to_index, norms = db.get_all_movie_features(genre_cols)
        movie_ids = sorted(movies.keys())
        
        # Get ML-powered recommendations
        recommendations = hybrid_recommender.get_recommendations(
            user_id=user_id,
            ratings=rating_matrix,
            movies=movies,
            feature_matrix=feature_matrix,
            movie_ids=movie_ids,
            top_k=n
        )
        
        print(f"Returning {len(recommendations)} recommendations")
        return jsonify({"recommendations": recommendations}), 200
        
    except Exception as e:
        print(f"Error in recommendations: {e}")
        # Fallback to popularity-based
        user_ratings = rating_matrix.get(user_id, {})
        rated_movies = set(user_ratings.keys())
        fallback_recs = [
            movie for movie_id, movie in movies.items() 
            if movie_id not in rated_movies
        ]
        fallback_recs.sort(key=lambda x: (x.get("Votes", 0) * x.get("Rating", 0)), reverse=True)
        
        result = fallback_recs[:n]
        print(f"Fallback returning {len(result)} recommendations")
        return jsonify({"recommendations": result}), 200

# -----------------------------
# ML Prediction
# -----------------------------
@app.route('/api/predict', methods=['GET'])
def api_predict():
    user_id = request.args.get('user_id')
    movie_id = request.args.get('movie_id')
    
    if user_id is None or movie_id is None:
        return jsonify({"error": "user_id and movie_id required"}), 400
    
    try:
        user_id = int(user_id)
        movie_id = int(movie_id)
    except:
        return jsonify({"error": "user_id and movie_id must be ints"}), 400
    
    if movie_id not in movies:
        return jsonify({"error": "movie_id not found"}), 404

    try:
        # Get feature matrix
        feature_matrix, movie_id_to_index, norms = db.get_all_movie_features(genre_cols)
        movie_ids = sorted(movies.keys())
        
        if movie_id not in movie_ids:
            return jsonify({"error": "movie not in feature matrix"}), 404
        
        # Get user profile
        user_profile = hybrid_recommender._get_user_profile(
            user_id, rating_matrix, feature_matrix, movie_ids
        )
        
        # Get movie features
        movie_idx = movie_ids.index(movie_id)
        movie_features = feature_matrix[movie_idx]
        
        # Predict rating
        prediction = hybrid_recommender.regression.predict(
            user_profile, movie_features.reshape(1, -1)
        )[0]
        
        # Calculate confidence based on user history
        num_user_ratings = len(rating_matrix.get(user_id, {}))
        confidence = min(0.95, 0.2 + 0.05 * num_user_ratings)
        
        return jsonify({
            "Name": movies[movie_id]["Name"],
            "movie_id": movie_id,
            "predicted_rating": round(float(prediction), 3),
            "confidence": round(float(confidence), 3),
            "source": "hybrid_ml_model"
        }), 200
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        # Fallback to average rating
        return jsonify({
            "Name": movies[movie_id]["Name"],
            "movie_id": movie_id,
            "predicted_rating": round(movies[movie_id]["Rating"], 3),
            "confidence": 0.5,
            "source": "fallback_average"
        }), 200

# -----------------------------
# Other endpoints (unchanged)
# -----------------------------
@app.route('/api/movie/<int:movie_id>', methods=['GET'])
def api_movie(movie_id):
    if movie_id not in movies:
        return jsonify({"error": "movie not found"}), 404
    
    movie = movies[movie_id].copy()
    cluster_assignments = db.get_cluster_assignments()
    movie["cluster_id"] = cluster_assignments.get(movie_id)
    
    return jsonify(movie), 200

@app.route('/api/user/archive', methods=['GET'])
def api_archive():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "user_id required"}), 400
    
    try:
        user_id = int(user_id)
    except:
        return jsonify({"error": "user_id must be int-like"}), 400
    
    user_ratings = rating_matrix.get(user_id, {})
    archive = []
    
    for movie_id, rating in user_ratings.items():
        if movie_id in movies:
            movie_data = movies[movie_id].copy()
            movie_data["user_rating"] = rating
            archive.append(movie_data)
    
    return jsonify({"archive": archive}), 200

@app.route('/api/health', methods=['GET'])
def health():
    status = {
        "status": "ok",
        "movies_loaded": len(movies),
        "genre_columns": len(genre_cols),
        "models_initialized": hybrid_recommender.is_initialized,
        "users_registered": len(_users)
    }
    return jsonify(status), 200

# -----------------------------
# Initialize on startup
# -----------------------------
if __name__ == '__main__':
    # Load dataset and initialize models
    load_dataset_and_initialize()
    
    port = int(os.environ.get("PORT", 5000))
    app.logger.info("Starting ML-powered backend on port %d", port)
    app.run(host='127.0.0.1', port=port, debug=True)