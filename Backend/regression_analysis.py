import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import json
import joblib

# --- Load data ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(BASE_DIR, "..", "datasets", "movies_data_processed.csv")
df = pd.read_csv(data_path)
print(f"Loaded {len(df)} movies for evaluation")

# --- Features ---
genre_cols = [c for c in df.columns if c.startswith("Genre_")]
feature_cols = genre_cols + ['Year_norm', 'Duration_norm', 'Rating_norm', 'Votes_log']
X = df[feature_cols].fillna(0).values
y = df["Rating"].values

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train regressor ---
reg = LinearRegression()
reg.fit(X_train, y_train)

# --- Predictions ---
y_pred_train = reg.predict(X_train)
y_pred_test = reg.predict(X_test)

# --- Metrics ---
print("\n=== Linear Regression Evaluation ===")
print(f"Train R²: {r2_score(y_train, y_pred_train):.4f}")
print(f"Test  R²: {r2_score(y_test, y_pred_test):.4f}")
print(f"Train RMSE: {mean_squared_error(y_train, y_pred_train, squared=False):.4f}")
print(f"Test  RMSE: {mean_squared_error(y_test, y_pred_test, squared=False):.4f}")

# # --- Feature importance ---
# coef_df = pd.DataFrame({
#     'Feature': feature_cols,
#     'Coefficient': reg.coef_
# }).sort_values('Coefficient', key=abs, ascending=False)

# print("\nTop features influencing ratings:")
# print(coef_df.head(15).to_string(index=False))

# # Optional: save model and metadata
# model_path = os.path.join(BASE_DIR, "models", "regressor_model.pkl")
# joblib.dump(reg, model_path)
# metadata_path = os.path.join(BASE_DIR, "models", "regressor_metadata.json")
# metadata = {'feature_cols': feature_cols}
# with open(metadata_path, 'w') as f:
#     json.dump(metadata, f, indent=2)

from sklearn.metrics import mean_squared_error
import numpy as np

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test  = mean_squared_error(y_test, y_pred_test)

rmse_train = np.sqrt(mse_train)
rmse_test  = np.sqrt(mse_test)

print(f"Train RMSE: {rmse_train:.4f}")
print(f"Test  RMSE: {rmse_test:.4f}")


print(f"\n[OK] Model and metadata saved to {model_path} / {metadata_path}")
