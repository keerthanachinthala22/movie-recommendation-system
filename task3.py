import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_csv("tmdb_5000_movies.csv")
df["overview"] = df["overview"].fillna("")
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(df["overview"])
print("Vocabulary Size :")
print(len(vectorizer.get_feature_names_out()))
print("\nTF-IDF Matrix Shape :")
print(tfidf_matrix.shape)