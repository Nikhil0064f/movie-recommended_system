import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv("tmdb_movies.csv")

# Clean data
movies["overview"] = movies["overview"].fillna("")
movies["movie_title"] = movies["movie_title"].fillna("Unknown")

# Combine movie descriptions with title
movies["content"] = (
    movies["movie_title"] + " " + movies["overview"]
)

# Convert text into numerical vectors
tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

tfidf_matrix = tfidf.fit_transform(movies["content"])

# Calculate cosine similarity
similarity = cosine_similarity(tfidf_matrix)

# Recommendation function
def recommend_movies(movie_name, n=5):

    matches = movies[
        movies["movie_title"].str.contains(
            movie_name,
            case=False,
            regex=False
        )
    ]

    if matches.empty:
        return pd.DataFrame()

    # Get selected movie index
    index = matches.index[0]

    # Calculate similarity scores
    scores = list(enumerate(similarity[index]))

    # Sort by similarity
    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove selected movie
    scores = [
        item for item in scores if item[0] != index
    ]

    # Get top recommendations
    top_movies = scores[:n]

    # Create result dataframe
    result = movies.iloc[
        [item[0] for item in top_movies]
    ].copy()

    result["similarity"] = [
        round(item[1] * 100, 2)
        for item in top_movies
    ]

    return result[
        [
            "movie_title",
            "release_date",
            "rating",
            "popularity",
            "similarity",
            "overview"
        ]
    ]


# Test the recommendation system
if __name__ == "__main__":

    movie = "Spider-Man: Brand New Day"

    print("Selected Movie:", movie)
    print("\nRecommended Movies:\n")

    recommendations = recommend_movies(movie)

    print(recommendations.to_string(index=False))
