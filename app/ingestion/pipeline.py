from app.ingestion.extractor import extractor
from app.ingestion.preprocessing import preprocess_page
from app.ingestion.chunking import chunker
from app.core.embeddings import embedding_engine

async def run_ingestion_pipeline(file_path: str, filename: str):
    # 1. Extraction (Using your dictionary output)
    extracted_data = extractor.extract(file_path)
    
    if not extracted_data or not extracted_data.get("pages"):
        return []

    processed_data =[]
    base_metadata = extracted_data.get("metadata", {"filename": filename})

    # 2. Process Page by Page
    for page in extracted_data["pages"]:
        # Preprocess using your custom logic
        clean_text = preprocess_page(page)
        
        if not clean_text.strip():
            continue
        
        # 3. Chunk the clean text for this page
        chunks = chunker.chunk(clean_text)
        
        # 4. Multilingual Embedding
        for chunk in chunks:
            vector = embedding_engine.generate(chunk)
            
            # Combine PDF metadata with specific Page metadata
            meta = base_metadata.copy()
            meta["page_number"] = page["page_number"]
            meta["source"] = page["source_type"]

            processed_data.append({
                "text": chunk,
                "vector_field": vector,
                "metadata": meta
            })
            
    return processed_data