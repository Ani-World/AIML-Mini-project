"""
Database module for SQLite operations
Provides clean interface for all database operations
"""
import sqlite3
import json
import os
import numpy as np
from typing import List, Dict, Optional, Tuple


def build_tmdb_poster_url(poster_path, size='w500'):
    """
    Build TMDB poster URL from poster path.
    Reference: https://developer.themoviedb.org/docs/image-basics
    
    Args:
        poster_path: TMDB poster path (e.g., '/1E5baAaEse26fej7uHcjOgEE2t2.jpg') or full URL
        size: Image size - 'w500', 'w780', or 'original'
    
    Returns:
        Full TMDB image URL or original URL if already a full URL, None if empty
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

# Database path
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "instance", "movies.db")


def get_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn


def get_all_movies() -> Dict[int, Dict]:
    """
    Fetch all movies from database and return as dict
    Returns: {movie_id: movie_dict}
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM movies ORDER BY movie_id")
    rows = cursor.fetchall()
    conn.close()
    
    movies = {}
    for row in rows:
        movie_id = row['movie_id']
        movies[movie_id] = {
            "movie_id": movie_id,
            "Name": row['name'],
            "Year": row['year'],
            "Duration": row['duration'],
            "Rating": row['avg_rating'],
            "Votes": row['votes'],
            "Director": row['director'],
            "Actor 1": row['actor1'],
            "Actor 2": row['actor2'],
            "Actor 3": row['actor3'],
            "Genre": row['genre_vector'] if 'genre_vector' in row.keys() else '{}',
            "cluster_id": row['cluster_id'],
            "poster": build_tmdb_poster_url(row['poster_url'] if 'poster_url' in row.keys() else None, size='w500') or f"https://via.placeholder.com/300x420?text={movie_id}",
        }
    
    return movies


def get_movie(movie_id: int) -> Optional[Dict]:
    """Fetch single movie by ID"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM movies WHERE movie_id = ?", (movie_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return {
        "movie_id": row['movie_id'],
        "Name": row['name'],
        "Year": row['year'],
        "Duration": row['duration'],
        "Rating": row['avg_rating'],
        "Votes": row['votes'],
        "Director": row['director'],
        "Actor 1": row['actor1'],
        "Actor 2": row['actor2'],
        "Actor 3": row['actor3'],
        "Genre": row['genre_vector'] if 'genre_vector' in row.keys() else '{}',
        "cluster_id": row['cluster_id'],
        "poster": build_tmdb_poster_url(row['poster_url'] if 'poster_url' in row.keys() else None, size='w500') or f"https://via.placeholder.com/300x420?text={movie_id}",
    }


def get_user_ratings(user_id: int) -> Dict[int, float]:
    """
    Fetch all ratings for a user
    Returns: {movie_id: rating}
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT movie_id, rating FROM ratings WHERE user_id = ?", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    
    return {row['movie_id']: row['rating'] for row in rows}


def save_rating(user_id: int, movie_id: int, rating: float):
    """Save or update user rating"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT OR REPLACE INTO ratings (user_id, movie_id, rating, timestamp)
        VALUES (?, ?, ?, datetime('now'))
    """, (user_id, movie_id, rating))
    
    conn.commit()
    conn.close()


def get_movie_features(movie_id: int, genre_cols: List[str], year_min: float, year_max: float, 
                       duration_max: float) -> Optional[np.ndarray]:
    """
    Generate feature vector for a movie
    Returns normalized feature vector with genre flags + numeric features
    """
    movie = get_movie(movie_id)
    if not movie:
        return None
    
    # Parse genre vector
    try:
        genre_dict = json.loads(movie['Genre'])
    except:
        genre_dict = {}
    
    # Build genre vector (one-hot for each genre)
    genre_vector = np.array([genre_dict.get(g, 0) for g in genre_cols], dtype=float)
    
    # Normalize numeric features
    norm_year = (movie['Year'] - year_min) / (year_max - year_min) if year_max > year_min else 0.0
    norm_duration = movie['Duration'] / duration_max if duration_max > 0 else 0.0
    norm_rating = movie['Rating'] / 5.0  # Scale to [0,1]
    log_votes = np.log1p(movie['Votes'])
    
    # Concatenate all features
    features = np.concatenate([genre_vector, [norm_year, norm_duration, norm_rating, log_votes]])
    
    return features


def get_cluster_assignments() -> Dict[int, int]:
    """Fetch cluster assignments for all movies"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT movie_id, cluster_id FROM movies WHERE cluster_id IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()
    
    return {row['movie_id']: row['cluster_id'] for row in rows}


def get_all_movie_features(genre_cols: List[str]) -> Tuple[np.ndarray, Dict[int, int], Dict[str, float]]:
    """
    Compute feature matrix for all movies
    Returns: (feature_matrix, movie_id_to_index, normalizations)
    """
    movies = get_all_movies()
    if not movies:
        return np.empty((0, len(genre_cols) + 4)), {}, {}
    
    # Compute normalization constants
    years = [m['Year'] for m in movies.values()]
    durations = [m['Duration'] for m in movies.values()]
    
    year_min, year_max = min(years), max(years)
    duration_max = max(durations) if durations else 1.0
    
    norms = {
        'year_min': year_min,
        'year_max': year_max,
        'duration_max': duration_max
    }
    
    # Build feature matrix
    features_list = []
    movie_id_to_index = {}
    
    for idx, (movie_id, movie) in enumerate(sorted(movies.items())):
        try:
            genre_dict = json.loads(movie['Genre'])
        except:
            genre_dict = {}
        
        genre_vector = np.array([genre_dict.get(g, 0) for g in genre_cols], dtype=float)
        norm_year = (movie['Year'] - year_min) / (year_max - year_min) if year_max > year_min else 0.0
        norm_duration = movie['Duration'] / duration_max if duration_max > 0 else 0.0
        norm_rating = movie['Rating'] / 5.0
        log_votes = np.log1p(movie['Votes'])
        
        feature = np.concatenate([genre_vector, [norm_year, norm_duration, norm_rating, log_votes]])
        features_list.append(feature)
        movie_id_to_index[movie_id] = idx
    
    feature_matrix = np.vstack(features_list) if features_list else np.empty((0, len(genre_cols) + 4))
    
    return feature_matrix, movie_id_to_index, norms


def update_cluster_assignments(movie_id: int, cluster_id: int):
    """Update cluster assignment for a movie"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("UPDATE movies SET cluster_id = ? WHERE movie_id = ?", (cluster_id, movie_id))
    conn.commit()
    conn.close()


def get_movies_for_onboarding(limit: int = 50) -> List[Dict]:
    """Fetch movies for onboarding page"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT movie_id, name, year, poster_url 
        FROM movies 
        ORDER BY votes DESC, avg_rating DESC 
        LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    movies = []
    for row in rows:
        movies.append({
            "movie_id": row['movie_id'],
            "name": row['name'],
            "year": row['year'],
            "poster": build_tmdb_poster_url(row['poster_url'], size='w500') or f"https://via.placeholder.com/300x420?text={row['movie_id']}"
        })
    
    return movies

def get_user_by_email(email: str) -> Optional[Dict]:
    """Fetch a user by email"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        return None
    
    return {
        "user_id": row['user_id'],
        "email": row['email'],
        "name": row['name'],
        "password_hash": row['password_hash'],
        # add other fields if needed
    }


def create_user(name: str, email: str, password_hash: str) -> int:
    """
    Create a new user
    Returns: user_id of the created user
    """
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO users (name, email, password_hash)
        VALUES (?, ?, ?)
    """, (name, email, password_hash))
    
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    
    return user_id
