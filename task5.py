import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv("tmdb_5000_movies.csv")
df["overview"] = df["overview"].fillna("")
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["overview"])
similarity = cosine_similarity(tfidf_matrix)
def recommend(movie_name, top_n=5):
    movie_name = movie_name.lower()
    movies = df["title"].str.lower()
    if movie_name not in movies.values:
        print("Movie not found")
        return
    index = movies[movies == movie_name].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = scores[1:top_n+1]
    print("Recommended Movies:\n")
    for i in scores:
        print(df.iloc[i[0]]["title"])
recommend("Avatar")   
    