from app.services.document_service import document_service
import os
from app.core.embeddings import EmbeddingPipeline
from app.core.opensearch_client import OpenSearchClient


def document_api(file_path: str,
    embedding_pipeline: EmbeddingPipeline,
    client: OpenSearchClient):

    if not os.path.exists(file_path):
        print("Error in Document Api: file does not exist at generated path")
        return {"success": False, "message": "Error uploading file."}
    result = document_service(file_path, embedding_pipeline, client)

    return result