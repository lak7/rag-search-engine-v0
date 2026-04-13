from collections import defaultdict
import json
import re
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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.chunk_embeddings_path = CACHE_PATH / "chunk_embeddings.npy"
        self.chunk_metadata_path = CACHE_PATH / "chunk_metadata.json"

    def build_chunk_embeddings(self, documents):
        self.documents = documents
        self.document_map = {doc['id']:doc for doc in documents}

        all_chunks = []
        chunk_metadata = []

        for midx, doc in enumerate(documents):
            desc = doc["description"]
            if desc.strip == '':
                continue
            _chunk = semantic_chunking(desc, overlap=1, chunk_size=4)
            all_chunks += _chunk
            for cidx in range(len(_chunk)):
                chunk_metadata.append(
                    {"movie_idx": midx,
                    "chunk_idx": cidx,
                    "total_chunks": len(_chunk)}
                )
            
        self.chunk_embeddings = self.model.encode(all_chunks)
        self.chunk_metadata = chunk_metadata
        np.save(self.chunk_embeddings_path, self.chunk_embeddings)
        with open(self.chunk_metadata_path, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)},
                f,
                indent=2,
            )
        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc['id']: doc for doc in documents}

        if (
            self.chunk_embeddings_path.exists()
            and self.chunk_metadata_path.exists()
        ):
            self.chunk_embeddings = np.load(self.chunk_embeddings_path)
            with open(self.chunk_metadata_path, "r") as f:
                metadata = json.load(f)
            self.chunk_metadata = metadata["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        embedded_query = self.generate_embedding(query)
        chunk_scores = []
        movie_scores = defaultdict(lambda: 0)
        for idx in range(len(self.chunk_embeddings)):
            chunk_emb = self.chunk_embeddings[idx]
            metadata = self.chunk_metadata[idx]
            midx, cidx = metadata['movie_idx'], metadata['chunk_idx']
            similarity = cosine_similarity(embedded_query, chunk_emb)
            chunk_scores.append({
                "movie_idx": midx,
                "chunk_idx": cidx,
                "score": similarity
            })
            movie_scores[midx] = max(movie_scores[midx], similarity)
        sorted_final_scores = sorted(movie_scores.items(), key=lambda x:x[1], reverse=True)
        res = []
        for midx, score in sorted_final_scores[:limit]:
            doc = self.document_map[midx]
            res.append(
                {
                    "id": doc['id'],
                    "title": doc['title'],
                    "document": doc['description'][:100],
                    "score": round(score, 4),
                    "metadata": {}
                }
            )
        return res

def search_chunked(query, limit=5):
    chunked_obj = ChunkedSemanticSearch()
    movies = load_data()
    chunked_obj.load_or_create_chunk_embeddings(movies)
    results = chunked_obj.search_chunks(query, limit)
    for i, res in enumerate(results):
        print(f"\n{i+1}. {res['title']} (score: {res['score']:.4f})")
        print(f"   {res['document']}...")


def fixed_size_chunking(text, overlap, chunk_size=50):
    words = text.split()
    chunks = []
    step_size = chunk_size-overlap
    for i in range(0, len(words), step_size):
        chunk_words=words[i: i+chunk_size]
        if len(chunk_words) <= overlap:
            break
        chunks.append("".join(chunk_words))
            
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")
    return chunks

def semantic_chunking(text, overlap=0, chunk_size=4): 
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    step_size = chunk_size-overlap
    for i in range(0, len(sentences), step_size):
        chunk_words=sentences[i: i+chunk_size]
        if len(chunk_words) <= overlap:
            break
        chunks.append(" ".join(chunk_words))

    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")
    return chunks

    
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

def embed_chunks():
    documents = load_data()
    chunked_search = ChunkedSemanticSearch()
    embeddings = chunked_search.load_or_create_chunk_embeddings(documents)
    print(f"Generated {len(embeddings)} chunked embeddings")

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