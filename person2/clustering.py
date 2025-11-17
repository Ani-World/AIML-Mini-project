
'''
File : clustering.py
Work : This code performs K-Means clustering on movies using numeric features , assigns each movie to a cluster, saves the cluster assignments as a CSV, and visualizes the clusters in 2D using PCA in a scatter plot.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

# Create plots folder if it doesn't exist
os.makedirs('person2/plots', exist_ok=True)

# Load preprocessed data
df = pd.read_csv('datasets/preprocessed_movies_data.csv')

# Select features for clustering (numeric only)
features = ['Year', 'Duration', 'Votes'] + [col for col in df.columns if col.startswith('Genre_')]
X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# Save cluster assignments **inside plots folder**
df[['Name', 'Cluster']].to_csv('person2/plots/Cluster_Assignments.csv', index=False)
print("Cluster assignments saved in person2/plots/Cluster_Assignments.csv")

# Visualize clusters (2D PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(6,4))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters, palette='Set2')
plt.title("K-Means Clustering of Movies")
plt.savefig('person2/plots/KMeans_Clusters.png')
plt.close()
