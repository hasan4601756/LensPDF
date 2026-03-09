from app.ingestion.extractor import extractor
from app.ingestion.preprocessing import preprocess_text
from app.ingestion.chunking import chunker
from app.core.embeddings import embedding_engine

async def run_ingestion_pipeline(file_path: str, filename: str):
    # 1. Extraction (Using your sophisticated logic)
    raw_text, source_type = extractor.extract_from_pdf(file_path)
    
    if not raw_text.strip():
        return []

    # 2. Preprocessing
    clean_text = preprocess_text(raw_text, source_type)
    
    # 3. Intelligent Chunking
    chunks = chunker.chunk(clean_text)
    
    # 4. Multilingual Embedding
    processed_data = []
    for chunk in chunks:
        vector = embedding_engine.generate(chunk)
        processed_data.append({
            "text": chunk,
            "vector_field": vector,
            "metadata": {
                "filename": filename,
                "source": source_type
            }
        })
    return processed_data