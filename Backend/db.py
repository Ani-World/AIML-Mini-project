"""
Database module for SQLite operations
Provides clean interface for all database operations
"""
import sqlite3
import json
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple


def get_db_path():
    """Get database path and ensure instance directory exists"""
    instance_dir = os.path.join(os.path.dirname(__file__), "..", "instance")
    os.makedirs(instance_dir, exist_ok=True)
    return os.path.join(instance_dir, "movies.db")


def init_database_if_needed():
    """Initialize database if it doesn't exist"""
    db_path = get_db_path()
    
    if os.path.exists(db_path):
        return True  # Database already exists
    
    print(f"Database not found. Creating new database at: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create movies table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS movies (
                movie_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                year INTEGER,
                duration INTEGER,
                avg_rating REAL,
                votes INTEGER,
                director TEXT,
                actor1 TEXT,
                actor2 TEXT,
                actor3 TEXT,
                genre_vector TEXT,
                cluster_id INTEGER,
                poster_url TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create ratings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ratings (
                rating_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                movie_id INTEGER NOT NULL,
                rating REAL NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (user_id),
                FOREIGN KEY (movie_id) REFERENCES movies (movie_id),
                UNIQUE(user_id, movie_id)
            )
        ''')
        
        # Try to load data from CSV
        csv_path = os.path.join(os.path.dirname(__file__), "..", "datasets", "preprocessed_movies_data.csv")
        
        if os.path.exists(csv_path):
            print("Loading data from CSV...")
            load_movies_from_csv(cursor, csv_path)
        else:
            print("CSV file not found. Creating sample movies...")
            insert_sample_movies(cursor)
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ratings_user_id ON ratings(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ratings_movie_id ON ratings(movie_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_movies_cluster_id ON movies(cluster_id)')
        
        conn.commit()
        conn.close()
        print("Database initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        return False


def load_movies_from_csv(cursor, csv_path):
    """Load movies from CSV file into database"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} rows")
        
        df = df.fillna('')
        inserted_count = 0
        
        for index, row in df.iterrows():
            try:
                name = str(row['Name'])[:200] if 'Name' in row else f"Movie {index}"
                year = int(row['Year']) if 'Year' in row and pd.notna(row['Year']) else 2000
                duration = int(row['Duration']) if 'Duration' in row and pd.notna(row['Duration']) else 120
                rating = float(row['Rating']) if 'Rating' in row and pd.notna(row['Rating']) else 5.0
                votes = int(row['Votes']) if 'Votes' in row and pd.notna(row['Votes']) else 100
                director = str(row['Director'])[:100] if 'Director' in row else ''
                actor1 = str(row['Actor 1'])[:100] if 'Actor 1' in row else ''
                actor2 = str(row['Actor 2'])[:100] if 'Actor 2' in row else ''
                actor3 = str(row['Actor 3'])[:100] if 'Actor 3' in row else ''
                
                # Extract genre columns
                genre_cols = [col for col in df.columns if col.startswith('Genre_')]
                genre_dict = {}
                for genre_col in genre_cols:
                    genre_dict[genre_col] = int(row[genre_col]) if genre_col in row and pd.notna(row[genre_col]) else 0
                
                genre_json = json.dumps(genre_dict)
                
                cursor.execute('''
                    INSERT INTO movies 
                    (name, year, duration, avg_rating, votes, director, actor1, actor2, actor3, genre_vector, cluster_id, poster_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (name, year, duration, rating, votes, director, actor1, actor2, actor3, genre_json, None, None))
                
                inserted_count += 1
                
            except Exception as e:
                continue
        
        print(f"Inserted {inserted_count} movies from CSV")
        
    except Exception as e:
        print(f"Error loading CSV: {e}")
        insert_sample_movies(cursor)


def insert_sample_movies(cursor):
    """Insert sample movies for testing"""
    sample_movies = [
        ("The Shawshank Redemption", 1994, 142, 9.3, 2500000, "Frank Darabont", "Tim Robbins", "Morgan Freeman", "Bob Gunton", '{"Genre_Drama": 1}', None, None),
        ("The Godfather", 1972, 175, 9.2, 2000000, "Francis Ford Coppola", "Marlon Brando", "Al Pacino", "James Caan", '{"Genre_Crime": 1, "Genre_Drama": 1}', None, None),
        ("The Dark Knight", 2008, 152, 9.0, 2500000, "Christopher Nolan", "Christian Bale", "Heath Ledger", "Aaron Eckhart", '{"Genre_Action": 1, "Genre_Crime": 1, "Genre_Drama": 1}', None, None),
        ("Pulp Fiction", 1994, 154, 8.9, 1900000, "Quentin Tarantino", "John Travolta", "Uma Thurman", "Samuel L. Jackson", '{"Genre_Crime": 1, "Genre_Drama": 1}', None, None),
        ("Forrest Gump", 1994, 142, 8.8, 1800000, "Robert Zemeckis", "Tom Hanks", "Robin Wright", "Gary Sinise", '{"Genre_Drama": 1, "Genre_Romance": 1}', None, None),
    ]
    
    for movie in sample_movies:
        cursor.execute('''
            INSERT INTO movies 
            (name, year, duration, avg_rating, votes, director, actor1, actor2, actor3, genre_vector, cluster_id, poster_url)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', movie)
    
    print("Inserted sample movies")


def get_connection():
    """Get database connection - creates database if it doesn't exist"""
    # Initialize database if needed
    if not init_database_if_needed():
        raise Exception("Failed to initialize database")
    
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def build_tmdb_poster_url(poster_path, size='w500'):
    """Build TMDB poster URL"""
    if not poster_path:
        return None
    if poster_path.startswith('http://') or poster_path.startswith('https://'):
        return poster_path
    if poster_path.startswith('/'):
        return f"https://image.tmdb.org/t/p/{size}{poster_path}"
    return f"https://image.tmdb.org/t/p/{size}/{poster_path}"


def get_all_movies() -> Dict[int, Dict]:
    """Fetch all movies from database"""
    try:
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
                "Genre": row['genre_vector'],
                "cluster_id": row['cluster_id'],
                "poster": build_tmdb_poster_url(row['poster_url']) or f"https://via.placeholder.com/300x420?text={movie_id}",
            }
        
        return movies
    except Exception as e:
        print(f"Error in get_all_movies: {e}")
        return {}


def get_movie(movie_id: int) -> Optional[Dict]:
    """Fetch single movie by ID"""
    try:
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
            "Genre": row['genre_vector'],
            "cluster_id": row['cluster_id'],
            "poster": build_tmdb_poster_url(row['poster_url']) or f"https://via.placeholder.com/300x420?text={row['movie_id']}",
        }
    except Exception as e:
        print(f"Error in get_movie: {e}")
        return None


def get_user_ratings(user_id: int) -> Dict[int, float]:
    """Fetch all ratings for a user"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT movie_id, rating FROM ratings WHERE user_id = ?", (user_id,))
        rows = cursor.fetchall()
        conn.close()
        
        return {row['movie_id']: row['rating'] for row in rows}
    except Exception as e:
        print(f"Error in get_user_ratings: {e}")
        return {}


def save_rating(user_id: int, movie_id: int, rating: float):
    """Save or update user rating"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO ratings (user_id, movie_id, rating, timestamp)
            VALUES (?, ?, ?, datetime('now'))
        """, (user_id, movie_id, rating))
        
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error in save_rating: {e}")


def get_movie_features(movie_id: int, genre_cols: List[str], year_min: float, year_max: float, 
                       duration_max: float) -> Optional[np.ndarray]:
    """Generate feature vector for a movie"""
    try:
        movie = get_movie(movie_id)
        if not movie:
            return None
        
        try:
            genre_dict = json.loads(movie['Genre'])
        except:
            genre_dict = {}
        
        genre_vector = np.array([genre_dict.get(g, 0) for g in genre_cols], dtype=float)
        norm_year = (movie['Year'] - year_min) / (year_max - year_min) if year_max > year_min else 0.0
        norm_duration = movie['Duration'] / duration_max if duration_max > 0 else 0.0
        norm_rating = movie['Rating'] / 5.0
        log_votes = np.log1p(movie['Votes'])
        
        features = np.concatenate([genre_vector, [norm_year, norm_duration, norm_rating, log_votes]])
        return features
    except Exception as e:
        print(f"Error in get_movie_features: {e}")
        return None


def get_cluster_assignments() -> Dict[int, int]:
    """Fetch cluster assignments for all movies"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT movie_id, cluster_id FROM movies WHERE cluster_id IS NOT NULL")
        rows = cursor.fetchall()
        conn.close()
        
        return {row['movie_id']: row['cluster_id'] for row in rows}
    except Exception as e:
        print(f"Error in get_cluster_assignments: {e}")
        return {}


def get_all_movie_features(genre_cols: List[str]) -> Tuple[np.ndarray, Dict[int, int], Dict[str, float]]:
    """Compute feature matrix for all movies"""
    try:
        movies = get_all_movies()
        if not movies:
            return np.empty((0, len(genre_cols) + 4)), {}, {}
        
        years = [m['Year'] for m in movies.values()]
        durations = [m['Duration'] for m in movies.values()]
        
        year_min, year_max = min(years), max(years)
        duration_max = max(durations) if durations else 1.0
        
        norms = {
            'year_min': year_min,
            'year_max': year_max,
            'duration_max': duration_max
        }
        
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
    except Exception as e:
        print(f"Error in get_all_movie_features: {e}")
        return np.empty((0, len(genre_cols) + 4)), {}, {}


def update_cluster_assignments(movie_id: int, cluster_id: int):
    """Update cluster assignment for a movie"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("UPDATE movies SET cluster_id = ? WHERE movie_id = ?", (cluster_id, movie_id))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error in update_cluster_assignments: {e}")


def get_movies_for_onboarding(limit: int = 50) -> List[Dict]:
    """Fetch movies for onboarding page"""
    try:
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
                "poster": build_tmdb_poster_url(row['poster_url']) or f"https://via.placeholder.com/300x420?text={row['movie_id']}"
            })
        
        return movies
    except Exception as e:
        print(f"Error in get_movies_for_onboarding: {e}")
        return []


def get_user_by_email(email: str) -> Optional[Dict]:
    """Fetch a user by email"""
    try:
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
        }
    except Exception as e:
        print(f"Error in get_user_by_email: {e}")
        return None


def create_user(name: str, email: str, password_hash: str) -> int:
    """Create a new user"""
    try:
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
    except Exception as e:
        print(f"Error in create_user: {e}")
        return -1