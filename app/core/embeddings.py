from sentence_transformers import SentenceTransformer
from app.config import settings

class EmbeddingEngine:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def generate(self, text: str):
        return self.model.encode(text).tolist()

embedding_engine = EmbeddingEngine()