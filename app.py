
import streamlit as st
import pandas as pd
from model import recommend_movies

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# Load dataset
movies = pd.read_csv("tmdb_movies.csv")

# Website heading
st.title("🎬 Movie Recommendation System")

st.write(
    "Discover movies similar to your favourite movies!"
)

st.divider()

# Movie selection
movie_list = sorted(movies["movie_title"].tolist())

selected_movie = st.selectbox(
    "Select Your Favourite Movie",
    movie_list
)

# Number of recommendations
number = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=10,
    value=5
)

# Recommendation button
if st.button("Recommend Movies", type="primary"):

    result = recommend_movies(
        selected_movie,
        number
    )

    st.subheader("🍿 Recommended Movies")

    if result.empty:
        st.warning("No recommendations found.")

    else:

        for _, row in result.iterrows():

            st.markdown(
                f"### 🎥 {row['movie_title']}"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Movie Rating",
                f"{row['rating']:.1f}/10"
            )

            col2.metric(
                "Similarity",
                f"{row['similarity']:.1f}%"
            )

            col3.metric(
                "Popularity",
                f"{row['popularity']:.1f}"
            )

            st.write(
                "**Release Date:**",
                row["release_date"]
            )

            st.write(
                "**Story:**",
                row["overview"]
            )

            st.divider()

st.caption(
    "Powered by Python, Pandas, Scikit-learn and Streamlit"
)
