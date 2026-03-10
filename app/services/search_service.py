from app.core.embeddings import get_embedding_pipeline
from app.core.opensearch_client import get_client
from app.core.reranker import reranker

def search_service(query, limit):
    embedding_pipeline = get_embedding_pipeline()
    query_embedding = embedding_pipeline.embed_single(query)

    client = get_client()
    result = client.hybrid_search(query_text=query, query_vector=query_embedding, top_k=limit)
    if result['success'] == False:
        raise RuntimeError(f"Search Failed {result['message']}")

    reranked_search = reranker.rerank(query, result['search_result'])

    return reranked_search