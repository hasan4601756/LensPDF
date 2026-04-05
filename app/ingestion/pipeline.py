from typing import Dict, Any
import hashlib

def calculate_binary_hash(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def run_ingestion_pipeline(filepath: str) -> Dict[str, Any]:
    from .extractor import extract_text
    result = extract_text(filepath)
    file_hash = calculate_binary_hash(file_path=filepath)

    # time.sleep(15)

    if not result or not result.get("pages"):
        return {"document": None, "chunks": []}
    
    file_metadata = result.get("metadata", {})

    all_chunks = []
    
    from .preprocessing import preprocess_page
    from .chunking import chunk_text
    for page in result["pages"]:
        cleaned_text = preprocess_page(page)
        page_number = page.get("page_number", 0)

        if not cleaned_text.strip():
            continue
        
        chunks = chunk_text(cleaned_text)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "content": chunk,
                "metadata": { 
                    "page_number": page_number,
                    "chunk_id" : page_number + i,
                }
            })

    return {
        "document": {
            "file_hash": file_hash,
            **file_metadata
        },
        "chunks": all_chunks
    }