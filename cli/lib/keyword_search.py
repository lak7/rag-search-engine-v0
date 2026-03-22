from ntpath import exists
import os
from lib.load_data import load_data, load_stopwords
from nltk.stem import PorterStemmer
import pickle
import string
from lib.common import CACHE_PATH
import collections

class InvertedIndex:
    def __init__(self) -> None:
        self.index = collections.defaultdict(set)
        self.docmap = {}
        self.index_path = CACHE_PATH / 'index.pkl'
        self.docmap_path = CACHE_PATH / 'docmap.pkl'

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for tok in set(tokens):
            self.index[tok].add(doc_id)
    
    def get_documents(self, term):
        term = term.lower()
        return sorted(list(self.index[term]))

    def build(self):
        movies = load_data()
        for mov in movies:
            movId = mov["id"]
            movText = f"{mov["title"]} {mov["description"]}"
            self.__add_document(movId, movText)
            self.docmap[movId] = mov

    def save(self):
        os.makedirs(CACHE_PATH, exist_ok={True})
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)

        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

def build_command():
    buildInvertedIndex = InvertedIndex()
    buildInvertedIndex.build()
    buildInvertedIndex.save()
    print("Cache for inverted indexing has been successsfully built")

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
    

