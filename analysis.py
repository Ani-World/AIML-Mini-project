# ==============================
# Movie Rating Prediction & Evaluation
# ==============================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ------------------------------
# Create folder to save plots
# ------------------------------
os.makedirs('plots', exist_ok=True)

# ------------------------------
# Load Preprocessed Dataset
# ------------------------------
df = pd.read_csv('datasets/preprocessed_movies_data.csv')
print("Preprocessed dataset loaded. Shape:", df.shape)

# ------------------------------
# Prepare Features and Target
# ------------------------------
X = df.drop(['Name', 'Rating'], axis=1)
y = df['Rating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ------------------------------
# Regression Models (Linear & Ridge)
# ------------------------------
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    results[name] = {"RMSE": rmse, "MAE": mae}
    
    # Scatter plot: Actual vs Predicted
    plt.figure(figsize=(6,4))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
    plt.xlabel('Actual Ratings')
    plt.ylabel('Predicted Ratings')
    plt.title(f'{name}: Actual vs Predicted Ratings')
    plt.savefig(f'plots/{name}_Actual_vs_Predicted.png')
    plt.close()

# Save results table
results_df = pd.DataFrame(results).T
results_df.to_csv('plots/Regression_Models_Comparison.csv')
print("\nRegression Models Comparison:")
print(results_df)

# ------------------------------
# Extra Visualization: Rating Distribution
# ------------------------------
plt.figure(figsize=(6,4))
sns.histplot(df['Rating'], bins=10, kde=True)
plt.title("Rating Distribution")
plt.savefig('plots/Rating_Distribution.png')
plt.close()
