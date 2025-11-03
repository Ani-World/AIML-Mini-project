import sqlite3
import pandas as pd
import json
import os
import joblib
import numpy as np
from datetime import datetime

# === 1. Paths ===
csv_path = "datasets/movies_data.csv"
db_dir = "instance"
db_path = os.path.join(db_dir, "movies.db")
os.makedirs(db_dir, exist_ok=True)

# === 2. Read CSV ===
df = pd.read_csv(csv_path)

# Normalize column names
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

# === 3. Create genre_vector JSON from Genre column ===
def make_genre_vector(genre_str):
    if pd.isna(genre_str):
        return json.dumps({})
    genres = [g.strip() for g in genre_str.split(",")]
    return json.dumps({g: 1 for g in genres})

df["genre_vector"] = df["Genre"].apply(make_genre_vector)

# === 4. Add placeholder columns ===
df["cluster_id"] = None
df["feature_path"] = None
df["poster_url"] = None

# === 5. Select + rename columns to match target schema ===
movies_df = df.rename(
    columns={
        "Name": "name",
        "Year": "year",
        "Duration": "duration",
        "Rating": "avg_rating",
        "Votes": "votes",
        "Director": "director",
        "Actor_1": "actor1",
        "Actor_2": "actor2",
        "Actor_3": "actor3"
    }
)[[
    "name",
    "year",
    "duration",
    "avg_rating",
    "votes",
    "director",
    "actor1",
    "actor2",
    "actor3",
    "genre_vector",
    "cluster_id",
    "feature_path",
    "poster_url"
]]

# === 6. Connect to SQLite ===
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row  # Enable column access by name
cursor = conn.cursor()

# === 7. Create tables ===

# Movies table
cursor.execute("""
CREATE TABLE IF NOT EXISTS movies (
    movie_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    year INTEGER,
    duration INTEGER,
    avg_rating REAL,
    votes INTEGER,
    director TEXT,
    actor1 TEXT,
    actor2 TEXT,
    actor3 TEXT,
    genre_vector JSON,
    cluster_id INTEGER,
    feature_path TEXT,
    poster_url TEXT
);
""")

# Users table
cursor.execute("DROP TABLE IF EXISTS users")
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    email TEXT UNIQUE,
    password_hash TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
""")

# Ratings table
cursor.execute("""
CREATE TABLE IF NOT EXISTS ratings (
    user_id INTEGER,
    movie_id INTEGER,
    rating REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, movie_id),
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
);
""")

conn.commit()

# === 8. Insert movies ===
# Drop existing table to ensure proper schema
cursor.execute("DROP TABLE IF EXISTS movies")

# Recreate table with proper AUTOINCREMENT
cursor.execute("""
CREATE TABLE movies (
    movie_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
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
    feature_path TEXT,
    poster_url TEXT
);
""")

# Insert movies manually to get proper movie_id AUTOINCREMENT
for _, row in movies_df.iterrows():
    cursor.execute("""
        INSERT INTO movies (name, year, duration, avg_rating, votes, director, actor1, actor2, actor3, genre_vector, cluster_id, feature_path, poster_url)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row['name'], row['year'], row['duration'], row['avg_rating'], row['votes'],
        row['director'], row['actor1'], row['actor2'], row['actor3'],
        row['genre_vector'], row['cluster_id'], row['feature_path'], row['poster_url']
    ))

conn.commit()
print(f"[OK] Inserted {len(movies_df)} movies into database")

# === 9. Populate cluster assignments using trained KMeans model ===
print("Loading KMeans model to assign clusters...")
try:
    kmeans_path = "Backend/models/kmeans_model.pkl"
    metadata_path = "Backend/models/model_metadata.json"
    
    if os.path.exists(kmeans_path) and os.path.exists(metadata_path):
        kmeans_model = joblib.load(kmeans_path)
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        genre_cols = metadata['genre_cols']
        
        # Get all movies with their features
        cursor.execute("SELECT movie_id, year, duration, avg_rating, votes, genre_vector FROM movies")
        rows = cursor.fetchall()
        
        # Create feature matrix using the same features as training
        X = []
        movie_ids = []
        for row in rows:
            # Parse genre vector
            try:
                genre_dict = json.loads(row['genre_vector'])
            except:
                genre_dict = {}
            
            # Build feature vector matching training format
            genre_vector = [genre_dict.get(g, 0) for g in genre_cols]
            
            # Normalize numeric features (must match preprocessing)
            year_min, year_max = df['Year'].min(), df['Year'].max()
            duration_max = df['Duration'].max()
            
            norm_year = (row['year'] - year_min) / (year_max - year_min) if year_max > year_min else 0.0
            norm_duration = row['duration'] / duration_max if duration_max > 0 else 0.0
            norm_rating = row['avg_rating'] / 10.0  # Must match preprocessing: Rating / 10.0
            log_votes = np.log1p(row['votes'])
            
            # Combine all features
            features = genre_vector + [norm_year, norm_duration, norm_rating, log_votes]
            X.append(features)
            movie_ids.append(row['movie_id'])
        
        # Predict clusters
        if X:
            X = np.array(X)
            clusters = kmeans_model.predict(X)
            
            # Update database with cluster assignments
            for movie_id, cluster_id in zip(movie_ids, clusters):
                cursor.execute("UPDATE movies SET cluster_id = ? WHERE movie_id = ?", 
                             (int(cluster_id), movie_id))
        
        conn.commit()
        print(f"[OK] Assigned clusters to {len(movie_ids)} movies")
    else:
        print("[WARNING] KMeans model or metadata not found, skipping cluster assignment")
except Exception as e:
    print(f"[ERROR] Error assigning clusters: {e}")

conn.close()

print(f"[OK] Database created at {db_path} with:")
print(f"   - {len(movies_df)} movies")
print("   - users table ready")
print("   - ratings table ready")
