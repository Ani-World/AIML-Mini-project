import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
import os
warnings.filterwarnings('ignore')

# Paths
raw_path = 'datasets/movies_data.csv'
processed_path = 'datasets/preprocessed_movies_data.csv'

# Check if raw dataset exists
if not os.path.exists(raw_path):
    raise FileNotFoundError(f"{raw_path} not found!")

# Load dataset
df = pd.read_csv(raw_path)
print("Original dataset loaded. Shape:", df.shape)

# Fill missing numeric values with median
for col in ['Year', 'Duration', 'Votes']:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Fill missing categorical values with 'Unknown'
for col in ['Genre', 'Director', 'Actor 1', 'Actor 2', 'Actor 3']:
    df[col].fillna('Unknown', inplace=True)

# One-hot encoding for Genre
df = pd.get_dummies(df, columns=['Genre'], drop_first=True)

# Label encoding for Director and actors
le_director = LabelEncoder()
df['Director'] = le_director.fit_transform(df['Director'].astype(str))

for col in ['Actor 1', 'Actor 2', 'Actor 3']:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Save the preprocessed dataset
df.to_csv(processed_path, index=False)
print(f"Preprocessed dataset saved as '{processed_path}' with shape {df.shape}")
