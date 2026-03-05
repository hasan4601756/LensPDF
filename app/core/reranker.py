from sentence_transformers import CrossEncoder
from app.config import settings

class ReRanker:
    def __init__(self):
        self.model = CrossEncoder(settings.RERANK_MODEL)

    def rerank(self, query: str, documents: list):
        if not documents: return []
        pairs = [[query, doc['text']] for doc in documents]
        scores = self.model.predict(pairs)
        for i, score in enumerate(scores):
            documents[i]['rerank_score'] = float(score)
        return sorted(documents, key=lambda x: x['rerank_score'], reverse=True)

reranker = ReRanker()