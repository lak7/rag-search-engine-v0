import numpy as np
from sentence_transformers import SentenceTransformer

from lib.common import CACHE_PATH
from lib.load_data import load_data

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = CACHE_PATH / 'movie_embeddings.npy'
    
    def generate_embedding(self, text):
        if (text is None) or (text == ""):
            return ValueError("Text cannot be None or empty")
        return (self.model.encode([text]))[0]
    
    def build_embeddings(self, document):
        self.documents = document
        self.document_map = {}
        movie_strings = []
        for doc in self.documents:
            self.document_map[doc['id']] = doc
            movie_strings.append(f"{doc['title']} {doc['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        np.save(self.embeddings_path, self.embeddings)
        return self.embeddings

    def search(self, query, limit):
        if self.embeddings is None:
            return ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        query_embedding = self.generate_embedding(query)
        similarities = []
        for doc_emb, doc in zip(self.embeddings, self.documents):
            similarity = cosine_similarity(doc_emb, query_embedding)
            similarities.append((similarity, doc))

        similarities.sort(key=lambda x: x[0], reverse=True)
        return [{'score': sc, 'title':doc["title"], 'desc':doc["description"]} for sc, doc in similarities[:limit]]


    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc['id']] = doc

        if self.embeddings_path.exists():
            self.embeddings = np.load(self.embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

def search_query(query):
    semanticSearch = SemanticSearch()
    documents = load_data()
    semanticSearch.load_or_create_embeddings(documents)
    result = semanticSearch.search(query, 2)
    for idx, res in enumerate(result):
        print(f"{idx}. {res['title']}")

def verify_model():
    semanticSearch = SemanticSearch()
    print(f"Model loaded: {semanticSearch.model}")
    print(f"Max sequence length: {semanticSearch.model.max_seq_length}")

def embed_text(text):
    semanticSearch = SemanticSearch()
    embedding = semanticSearch.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    semanticSearch = SemanticSearch()
    documents = load_data()
    embeddings = semanticSearch.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    semanticSearch = SemanticSearch()
    documents = load_data()
    query_embedding = semanticSearch.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {query_embedding[:5]}")
    print(f"Shape: {query_embedding.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)