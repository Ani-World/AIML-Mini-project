from flask import render_template, Flask
import os
import pandas as pd

app = Flask(__name__)

# Get absolute path of the current app directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Correct dataset path
data_path = os.path.join(base_dir, "..", "datasets", "preprocessed_movies_data.csv")

# Load the cleaned dataset
df = pd.read_csv(data_path)

# Just to confirm columns
print(df.columns)

# Filtered view of popular films (rating ≥ 4)
popular_df = df[df['Rating'] >= 4].sort_values(by='Rating', ascending=False)

# Function to get random 20 films
def get_random_films(n=20):
    return df.sample(n).to_dict(orient="records")

# Function to get top-rated films
def get_popular_films(n=20):
    if len(popular_df) >= n:
        return popular_df.head(n).to_dict(orient="records")
    return popular_df.to_dict(orient="records")

@app.route("/")
def home():
    films = get_random_films()
    return render_template("home.html", films=films)

@app.route("/popular")
def popular():
    films = get_popular_films()
    return render_template("home.html", films=films)

if __name__ == "__main__":
    app.run(debug=True)
