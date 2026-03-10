from app.core.embeddings import EmbeddingPipeline
from app.core.opensearch_client import OpenSearchClient
from app.core.reranker import ReRanker

def search_service(
    query: str,
    limit: int,
    embedding_pipeline: EmbeddingPipeline,
    client: OpenSearchClient,
    reranker: ReRanker,
):
    query_embedding = embedding_pipeline.embed_single(query)
    result = client.hybrid_search(query_text=query, query_vector=query_embedding, top_k=limit)

    if not result['success']:
        raise RuntimeError(f"Search Failed: {result['message']}")

    return reranker.rerank(query, result['search_result'])