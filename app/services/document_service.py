from app.ingestion.pipeline import run_ingestion_pipeline
from app.core.embeddings import embedding_pipeline
from app.core.opensearch_client import client

def document_service(file_path : str):
    all_chunks = run_ingestion_pipeline(filepath=file_path)

    if len(all_chunks) == 0:
        print("No chunks returned.")
        return

    embedded_chunks = embedding_pipeline.embed_documents(all_chunks)

    client.create_index()
    response = client.bulk_index(embedded_chunks)
    
    return response