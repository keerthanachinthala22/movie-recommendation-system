import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv("tmdb_5000_movies.csv")
df = df[["title", "overview"]]
df["overview"] = df["overview"].fillna("")
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["overview"])
similarity = cosine_similarity(tfidf_matrix)
def recommend(movie_name):
    movies = df["title"].str.lower()
    movie_name = movie_name.lower()
    if movie_name not in movies.values:
        return[]
    index = movies[movies == movie_name].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:6]
    recommendations = []
    for score in scores:
        movie_index = score[0]
        recommendations.append(df.iloc[movie_index]["title"])
    return recommendations
st.title("Movie Recommendation System")
selected_movie = st.selectbox(
    "Select a Movie",sorted(df["title"].tolist())
)
if st.button("Recommend"):
    recommended_movies = recommend(selected_movie)
    if len(recommended_movies) == 0:
        st.write("Movie not found.")
    else:
        st.subheader("Recommended Movies")
        for movie in recommended_movies:
            st.write(movie)