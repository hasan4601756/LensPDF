from sentence_transformers import SentenceTransformer
from app.config import settings

class EmbeddingEngine:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)

    def generate(self, text: str, is_query: bool = False):
        # E5 models REQUIRE specific prefixes to map queries to passages correctly
        if "e5" in settings.EMBEDDING_MODEL.lower():
            prefix = "query: " if is_query else "passage: "
            text = prefix + text
            
        return self.model.encode(text).tolist()

embedding_engine = EmbeddingEngine()