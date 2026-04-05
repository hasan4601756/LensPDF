from app.core.embeddings import EmbeddingPipeline
from app.core.opensearch_client import OpenSearchClient

def document_service(
    file_path: str,
    embedding_pipeline: EmbeddingPipeline,
    client: OpenSearchClient,
):
    from app.ingestion.pipeline import run_ingestion_pipeline
    import os

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    result = run_ingestion_pipeline(filepath=file_path)
    all_chunks = result.get("chunks", [])
    file_metadata = result.get("document", {})

    if not all_chunks:
        return {"success": False, "message": "No chunks returned"}

    embedded_chunks = embedding_pipeline.embed_documents(all_chunks)
    client.create_index()
    return client.bulk_index(embedded_chunks, file_metadata)