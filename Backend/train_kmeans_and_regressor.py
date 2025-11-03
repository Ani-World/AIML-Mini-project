import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import joblib
import os

# Load clean dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "datasets", "movies_data_processed.csv")
df = pd.read_csv(data_path)

print(f"Loaded {len(df)} movies for training")

# Get all genre columns
genre_cols = [col for col in df.columns if col.startswith("Genre_")]

# Feature matrix for models: Use normalized features for better performance
# For KMeans and Regression, we'll use the normalized features
feature_cols = genre_cols + ['Year_norm', 'Duration_norm', 'Rating_norm', 'Votes_log']

print(f"Using {len(feature_cols)} features: {len(genre_cols)} genres + 4 numeric")

# Prepare features
X = df[feature_cols].fillna(0).values

# Train KMeans
print("Training KMeans model...")
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X)

# Save KMeans model
kmeans_path = os.path.join(BASE_DIR, "models", "kmeans_model.pkl")
os.makedirs(os.path.dirname(kmeans_path), exist_ok=True)
joblib.dump(kmeans, kmeans_path)
print(f"[OK] Saved KMeans model to {kmeans_path}")

# Train Linear Regressor (predicts rating)
print("Training Linear Regression model...")
regressor = LinearRegression()
regressor.fit(X, df["Rating"])

# Save Regressor model
regressor_path = os.path.join(BASE_DIR, "models", "regressor_model.pkl")
joblib.dump(regressor, regressor_path)
print(f"[OK] Saved Regressor model to {regressor_path}")

# Save feature column names for later use
metadata = {
    'feature_cols': feature_cols,
    'genre_cols': genre_cols,
    'n_features': len(feature_cols),
    'n_samples': len(df)
}
metadata_path = os.path.join(BASE_DIR, "models", "model_metadata.json")
import json
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"[OK] Saved model metadata to {metadata_path}")

print("\n[OK] All models saved successfully!")
print(f"   - KMeans clusters: {kmeans.n_clusters}")
print(f"   - Regressor R^2 score: {regressor.score(X, df['Rating']):.4f}")
