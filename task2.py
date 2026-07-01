import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
df = pd.read_csv("tmdb_5000_movies.csv")
df["overview"] = df["overview"].fillna("")
stop_words = set(stopwords.words("english"))
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", "",text)
    words = text.split()
    words = [word for word in words 
             if word not in stop_words]
    return" ".join(words)
df["clean_text"]=df["overview"].apply(clean_text)
print(df[["title","clean_text"]].head())
