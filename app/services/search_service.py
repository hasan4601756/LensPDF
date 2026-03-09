from app.core.opensearch_client import get_opensearch_client
from app.core.embeddings import embedding_engine
from app.core.reranker import reranker_engine
from app.config import settings

class SearchService:
    def __init__(self):
        self.client = get_opensearch_client()

    async def search(self, query: str, top_k: int):
        query_vector = embedding_engine.generate(query)
        
        search_body = {
            "size": top_k * 2, # Fetch more for reranking
            "query": {
                "knn": {
                    "vector_field": {"vector": query_vector, "k": top_k * 2}
                }
            }
        }
        
        response = self.client.search(index=settings.INDEX_NAME, body=search_body)
        results =[]
        for hit in response['hits']['hits']:
            results.append({
                "text": hit['_source']['text'],
                "metadata": hit['_source']['metadata'],
                "vector_score": hit['_score']
            })
            
        # Rerank
        final_results = reranker_engine.rerank(query, results)
        return final_results[:top_k]

search_service = SearchService()