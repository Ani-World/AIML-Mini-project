from flask import Flask, jsonify, request

app = Flask(__name__)

# Dummy movie data
movies = [
    {"movie_id": 1, "Name": "Inception", "Year": 2010, "Duration": 148, "Rating": 8.8, "Votes": 2000000,
     "Director": "Christopher Nolan", "Actor1": "Leonardo DiCaprio", "Actor2": "Joseph Gordon-Levitt", "Actor3": "Elliot Page"},
    {"movie_id": 2, "Name": "Interstellar", "Year": 2014, "Duration": 169, "Rating": 8.6, "Votes": 1800000,
     "Director": "Christopher Nolan", "Actor1": "Matthew McConaughey", "Actor2": "Anne Hathaway", "Actor3": "Jessica Chastain"},
]

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    user_id = request.args.get('user_id')
    n = int(request.args.get('n', 10))

    print(f"[MODEL CALL] Hybrid Recommendation triggered for user_id={user_id}")
    print("Calling: Regression + KNN + Apriori + KMeans Models...")

    recs = movies[:n]
    return jsonify({"user_id": user_id, "recommendations": recs})


@app.route('/api/predict', methods=['GET'])
def predict_rating():
    user_id = request.args.get('user_id')
    movie_id = request.args.get('movie_id')

    print(f"[MODEL CALL] Predict rating for user_id={user_id}, movie_id={movie_id}")
    print("Calling Regression model...")

    prediction = {
        "Name": "Inception",
        "movie_id": movie_id,
        "predicted_rating": 8.2,
        "confidence": 0.85,
        "source": "regression+knn"
    }
    return jsonify(prediction)


@app.route('/api/movie/<movie_id>', methods=['GET'])
def movie_info(movie_id):
    print(f"[MODEL CALL] Movie Metadata for movie_id={movie_id}")
    print("Calling Apriori Rules and Cluster Model...")

    movie = next((m for m in movies if str(m['movie_id']) == movie_id), None)
    if not movie:
        return jsonify({"error": "Movie not found"}), 404

    movie["cluster"] = 3
    movie["apriori_reasons"] = ["You liked Inception", "Sci-Fi cluster"]
    return jsonify(movie)


if __name__ == '__main__':
    app.run(debug=True)
