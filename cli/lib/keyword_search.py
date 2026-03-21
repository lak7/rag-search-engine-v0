from lib.load_data import load_data, load_stopwords
from nltk.stem import PorterStemmer
import string

def clean_text(txt):
    txt = txt.lower()
    txt = txt.translate(str.maketrans("", "", string.punctuation))
    return txt

def tokenize_text(txt):
    txt = clean_text(txt).split()
    stopwords = load_stopwords()
    stemmer = PorterStemmer()
    tokens = []
    for tok in txt:
        if tok not in stopwords:
            tokens.append(stemmer.stem(tok))
    return tokens

def has_matching_token(query_token, movie_token):
    for qTok in query_token:
        for mTok in movie_token:
            if qTok in mTok:
                return True
    return False 

def search_query(query):
    results = []
    queryToken = tokenize_text(query)
    print(f"This is the query token: {queryToken}")
    movies = load_data()
    for movie in movies:
        movieToken = tokenize_text(movie["title"])
        if has_matching_token(query_token=queryToken, movie_token=movieToken):
            results.append(movie["title"])
    return results
    

