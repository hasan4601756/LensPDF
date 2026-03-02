from app.ingestion.extractor import extractor
from app.ingestion.chunking import chunker
from app.core.embeddings import embedding_engine

async def process_document(file_path: str, filename: str):
    text = extractor.extract(file_path)
    chunks = chunker.chunk(text)
    
    processed_chunks = []
    for chunk in chunks:
        vector = embedding_engine.generate(chunk)
        processed_chunks.append({
            "text": chunk,
            "vector_field": vector,
            "metadata": {"filename": filename}
        })
    return processed_chunks