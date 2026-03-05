from app.core.embeddings import embedding_pipeline
from app.core.opensearch_client import client
from app.core.reranker import reranker

def search_service(query):
    query_embedding = embedding_pipeline.embed_single(query)

    result = client.hybrid_search(query_text=query, query_vector=query_embedding)

    reranked_search = reranker.rerank(query, result)

    return result