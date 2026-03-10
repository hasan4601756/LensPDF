from app.ingestion.pipeline import run_ingestion_pipeline
from app.core.embeddings import get_embedding_pipeline
from app.core.opensearch_client import get_client

def document_service(file_path : str):
    import os

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    result = run_ingestion_pipeline(filepath=file_path)

    all_chunks, file_metadata = result.get("chunks", []), result.get("document", {})

    if len(all_chunks) == 0:
        print("No chunks returned.")
        return {"success": False, "message": "No chunks returned"}
    
    embedding_pipeline = get_embedding_pipeline()
    embedded_chunks = embedding_pipeline.embed_documents(all_chunks)

    client = get_client()
    index_creation_result = client.create_index()

    if index_creation_result['success'] == False:
        return index_creation_result

    response = client.bulk_index(embedded_chunks, file_metadata)
    
    return response