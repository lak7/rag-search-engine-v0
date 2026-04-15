import math
from ntpath import exists
import os
import time
import token
from lib.load_data import load_data, load_stopwords
from nltk.stem import PorterStemmer
import pickle
import string
from lib.common import BM25_B, BM25_K1, CACHE_PATH
from collections import Counter, defaultdict

class InvertedIndex:
    def __init__(self) -> None:
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths= {}
        self.index_path = CACHE_PATH / 'index.pkl'
        self.docmap_path = CACHE_PATH / 'docmap.pkl'
        self.term_frequencies_path = CACHE_PATH / 'term_frequencies.pkl'
        self.doc_lengths_path = CACHE_PATH / "doc_lengths.pkl"

    def __add_document(self, doc_id, text):
        tokens = tokenize_text(text)
        for tok in set(tokens):
            self.index[tok].add(doc_id)
        self.term_frequencies[doc_id].update(Counter(tokens))
        self.doc_lengths[doc_id] = len(tokens)

    def __get_avg_doc_length(self) -> float:
        lengths = list(self.doc_lengths.values())
        if len(lengths) == 0:
            return 0.0
        total_toks_across_all_docs = 0
        for l in lengths:
             total_toks_across_all_docs += l
        return (total_toks_across_all_docs / len(lengths))

    
    def get_documents(self, term):
        term = term.lower()
        return sorted(list(self.index[term]))

    def get_tf(self, doc_id, term):
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("There can only be 1 token")
        return self.term_frequencies[doc_id][token[0]]

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        tf = self.get_tf(doc_id, term)
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()
        if avg_doc_length > 0:
            length_norm = 1 - b + b * (doc_length / avg_doc_length)
        else:
            length_norm = 1
        return (tf * (k1 + 1)) / (tf + k1 * length_norm)
        

    def get_idf(self, term):
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Term should not be greater than 1")
        token = token[0]
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[token])
        return math.log((doc_count + 1) / (term_doc_count + 1))

    def get_bm25_idf(self, term: str) -> float:
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Term should not be greater than 1")
        token = token[0]
        doc_count = len(self.docmap)
        df = len(self.index[token])
        return math.log((doc_count - df + 0.5) / (df + 0.5) + 1)

    def bm25(self, doc_id, term):
        final_bm25_tf = self.get_bm25_tf(doc_id, term)
        final_bm25_idf = self.get_bm25_idf(term)
        return final_bm25_tf*final_bm25_idf

    def bm25_search(self, query, limit):
        tokens = tokenize_text(query)
        scores = defaultdict()
        for docId in self.docmap:
            score = 0
            for tok in tokens:
                score += self.bm25(docId, tok)
            scores[docId] = score
        
        sorted_scores = sorted(scores.items(), 
            key = lambda x: x[1],
            reverse=True
        )
        sorted_scores = sorted_scores[:limit]

        # for doc_id, score in sorted_scores:
        #     print(f"{doc_id} {self.docmap[doc_id]['title']} - Score: {score}")
        return sorted_scores



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
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        try:
            with open(self.index_path, "rb") as f:
                self.index = pickle.load(f)
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
            with open(self.term_frequencies_path, "rb") as f:
                self.term_frequencies = pickle.load(f)
            with open(self.doc_lengths_path, "rb") as f:
                self.doc_lengths = pickle.load(f)
        except:
            print("Whoops, something went wrong")


def tf_command(doc_id, term):
    invIdx = InvertedIndex()
    invIdx.load()
    print(f"This many number of times in given doc {invIdx.get_tf(doc_id, term)}")

def idf_command(term):
    invIdx = InvertedIndex()
    invIdx.load()
    idf = invIdx.get_idf(term)
    print(f"Inverse document frequency of '{term}': {idf:.2f}")

def tfidf_command(doc_id, term):
    invIdx = InvertedIndex()
    invIdx.load()
    tf = invIdx.get_tf(doc_id, term)
    idf = invIdx.get_idf(term)
    tf_idf = tf*idf
    print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")

def bm25_idf_command(term):
    invIdx = InvertedIndex()
    invIdx.load()
    bm25_idf = invIdx.get_bm25_idf(term)
    print(f"BM25 IDF score of '{term}': {bm25_idf:.2f}")
    return float(bm25_idf)

def bm25_tf_command(doc_id, term):
    invIdx = InvertedIndex()
    invIdx.load()
    bm25tf = invIdx.get_bm25_tf(doc_id, term)
    print(f"BM25 TF score of '{term}' in document '{doc_id}': {bm25tf:.2f}")
    return bm25tf

def bm25search(query, limit=5):
    invIdx = InvertedIndex()
    invIdx.load()
    bm25_result = invIdx.bm25_search(query, limit)

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
    start = time.time()
    seen, results = set(), []
    queryToken = tokenize_text(query)
    invIdx = InvertedIndex()
    invIdx.load()

    for qTok in queryToken:
        matching_index_ids = invIdx.get_documents(qTok)
        for mid in matching_index_ids:
            if mid in seen:
                continue
            results.append(invIdx.docmap[mid])
            if len(results) >= 5:
                end = time.time()
                print("Time taken:", end - start, "seconds")
                return results

    end = time.time()
    print("Time taken:", end - start, "seconds")
    return results
    

