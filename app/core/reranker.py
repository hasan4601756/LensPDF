from sentence_transformers import CrossEncoder
from app.config import settings

class ReRanker:
    def __init__(self):
        self.model = None

    def rerank(self, query: str, documents: list):
        if self.model is None:
            self.get_reranker()
        
        if not isinstance(query, str) or not query.strip():
            return documents

        # if not isinstance(documents, list):
        #     raise ValueError("documents must be a list")
        
        if not documents: return []

        pairs = [[query, doc.get('content', '')] for doc in documents]

        try:
            for doc, score in zip(documents, self.model.predict(pairs)):
                doc['rerank_score'] = float(score)
        except Exception as e:
            raise RuntimeError("Reranking inference failed") from e
        return sorted(
            documents,
            key=lambda x: (x.get('rerank_score', 0), x.get('score', 0)),
            reverse=True
        )
    
    def get_reranker(self):
        try:
            self.model = CrossEncoder(settings.RERANK_MODEL)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load reranker model: {settings.RERANK_MODEL}"
            ) from e

reranker = ReRanker()