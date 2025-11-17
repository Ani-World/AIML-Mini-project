
'''
File : recommend_by_cluster.py
Work : This code provides movie recommendations based on clusters: it finds the cluster of a given movie (supports partial/case-insensitive input) and suggests other movies from the same cluster interactively.
'''

import pandas as pd

# Load cluster assignments
clusters_df = pd.read_csv('person2/plots/Cluster_Assignments.csv')

def recommend_movies_by_cluster(movie_name, top_n=5):
    # Find movies containing the input string (case-insensitive)
    matches = clusters_df[clusters_df['Name'].str.contains(movie_name, case=False, na=False)]
    if matches.empty:
        print(f"No movies found matching '{movie_name}'")
        return []
    # Take the first match for recommendation
    movie_name_exact = matches.iloc[0]['Name']
    cluster_id = matches.iloc[0]['Cluster']
    # Get other movies in the same cluster
    cluster_movies = clusters_df[clusters_df['Cluster'] == cluster_id]
    cluster_movies = cluster_movies[cluster_movies['Name'] != movie_name_exact]  # exclude input movie
    recommended = cluster_movies['Name'].head(top_n).tolist()
    return movie_name_exact, recommended


if __name__ == "__main__":
    movie = input("Enter a movie name (partial allowed): ").strip()
    num_recs = input("How many recommendations do you want? (default 5): ").strip()
    
    try:
        num_recs = int(num_recs)
    except:
        num_recs = 5  # default
    
    movie_exact, recs = recommend_movies_by_cluster(movie, top_n=num_recs)
    
    if recs:
        print(f"\nRecommendations based on '{movie_exact}':")
        for i, r in enumerate(recs, start=1):
            print(f"{i}. {r}")


'''
output : Enter a movie name (partial allowed): Love Story
How many recommendations do you want? (default 5): 5

Recommendations based on '1942: A Love Story':
1. #Gadhvi (He thought he was Gandhi)
2. #Yaaram
3. ...Aur Pyaar Ho Gaya
4. ...Yahaan
5. ?: A Question Mark
PS C:\Users\VAISHNAVI AHIRE\OneDrive\Desktop\AIML-Mini-project> 







'''