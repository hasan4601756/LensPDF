from opensearchpy import OpenSearch
from app.config import settings

def get_opensearch_client():
    client = OpenSearch(
        hosts=[{'host': settings.OPENSEARCH_HOST, 'port': settings.OPENSEARCH_PORT}],
        http_auth=(settings.OPENSEARCH_USER, settings.OPENSEARCH_PASS),
        use_ssl=False, # Set to True in production
        verify_certs=False
    )
    return client

def create_index_if_not_exists(client: OpenSearch):
    if not client.indices.exists(index=settings.INDEX_NAME):
        settings_body = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "vector_field": {
                        "type": "knn_vector",
                        "dimension": 384, # L6-v2 dimension
                        "method": {"name": "hnsw", "space_type": "l2", "engine": "nmslib"}
                    },
                    "metadata": {"type": "object"}
                }
            }
        }
        client.indices.create(index=settings.INDEX_NAME, body=settings_body)