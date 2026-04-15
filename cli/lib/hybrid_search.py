import os

from lib.load_data import load_data

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha=0.5, limit=5):
        bm25_res = self._bm25_search(query, limit*500 )
        sem_res = self.semantic_search.search_chunks(query, limit*500)
        combined_res = combined_result(bm25_res, sem_res, alpha)
        return combined_res

    # NOT IMPLEMENTING YET 
    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def weighted_search(query, alpha=0.5, limit=5):
    documents = load_data()
    hybrid_search_obj = HybridSearch(documents)
    res = hybrid_search_obj.weighted_search(query, alpha, limit)
    # print(f"Final RES: {res[:limit]}")
    for idx, re in enumerate(res[:limit]):
        print(f"{idx+1}. {re['title']}")
        print(f"Hybrid Score: {re['hybrid_score']}")
        print(f"BM25: {re['bm25_score']}, Semantic: {re['sem_score']}")


def normalize_search_results(results):
    scores = [r['score'] for r in results]
    norm_scores = normalize_list(scores)
    for idx, result in enumerate(results):
        result['normalized_score'] = norm_scores[idx]

    return results

def combined_result(bm25_res, sem_res, alpha):
    # print(f"SEM RES: {sem_res}")
    bm25_norm = normalize_search_results(bm25_res)
    sem_norm = normalize_search_results(sem_res)

    combined_norm = {}
    for norm in bm25_norm:
        doc_id = norm['doc_id']
        combined_norm[doc_id] = {
            'doc_id': doc_id,
            'bm25_score': norm['normalized_score'],
            'sem_score': 0.,
            'title': norm['title'],
            # 'description': norm['description']
        }
    
    for norm in sem_norm:
        doc_id = norm['id']
        if doc_id not in combined_norm:
            combined_norm[doc_id] = {
                'doc_id': doc_id,
                'bm25_score': 0.,
                'sem_score': norm['normalized_score'],
                'title': norm['title'],
                # 'description': norm['description']
            }
    
    for k,v in combined_norm.items():
        combined_norm[k]['hybrid_score'] = hybrid_score(v['bm25_score'], v['sem_score'], alpha)

    final_res = sorted(list(combined_norm.values()), key=lambda x:x['hybrid_score'], reverse=True)
    return final_res


def normalize_list(scores):
    if not scores: return []
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    if score_range == 0: return [1.]*len(scores)
    return [(score - min_score) / score_range for score in scores]

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return (alpha * bm25_score) + ((1 - alpha) * semantic_score)