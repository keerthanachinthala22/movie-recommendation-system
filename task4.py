import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv("tmdb_5000_movies.csv")
df["overview"] = df["overview"].fillna("")
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["overview"])
similarity = cosine_similarity(tfidf_matrix)
print("Similarity Matrix Shape:")
print(similarity.shape)
print("\nFirst 5 x 5 Similarity Matrix:")
print(similarity[:5, :5])