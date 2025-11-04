# clustering.py (improved saving)
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('person2/plots', exist_ok=True)
os.makedirs('person2/models', exist_ok=True)

# Load preprocessed data (ensure this file has Genre_ columns and Name)
df = pd.read_csv('datasets/preprocessed_movies_data.csv')

# Select features for clustering (numeric only)
features = ['Year', 'Duration', 'Votes'] + [col for col in df.columns if col.startswith('Genre_')]
X = df[features].fillna(0).values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# Save cluster assignments CSV
df[['Name', 'Cluster']].to_csv('person2/plots/Cluster_Assignments.csv', index=False)
print("Cluster assignments saved in person2/plots/Cluster_Assignments.csv")

# Save model artifacts for recommendation step
joblib.dump(kmeans, 'person2/models/kmeans.joblib')
joblib.dump(scaler, 'person2/models/scaler.joblib')
# Save feature column order so inference uses same columns
import json
with open('person2/models/feature_cols.json', 'w') as f:
    json.dump(features, f)
# Save dataframe (Name + features) for lookups
df[['Name'] + features].to_csv('person2/models/name_features_mapping.csv', index=False)

print("Saved kmeans, scaler, and feature mapping to person2/models/")

# Optional PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette='Set2', legend='full', s=20)
plt.title("K-Means Clustering of Movies (PCA projection)")
plt.savefig('person2/plots/KMeans_Clusters.png')
plt.close()
