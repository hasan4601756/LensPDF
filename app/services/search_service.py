from app.core.embeddings import embedding_pipeline
from app.core.opensearch_client import client

def search_service(query):
    query_embedding = embedding_pipeline.embed_single(query)

    result = client.hybrid_search(query_text=query, query_vector=query_embedding)

    return result