import time
import hashlib

def calculate_binary_hash(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def run_ingestion_pipeline(filepath: str) -> list:
    from .extractor import extract_text
    result = extract_text(filepath)
    file_metadata = result['metadata']
    file_hash = calculate_binary_hash(file_path=filepath)

    # time.sleep(15)

    if not result or not result.get("pages"):
        return []

    all_chunks = []

    for page in result["pages"]:
        from .preprocessing import preprocess_page
        cleaned_text = preprocess_page(page)
        page_number = page["page_number"]
        
        from .chunking import chunk_text
        chunks = chunk_text(cleaned_text)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    **file_metadata, 
                    "page_number": page_number,
                    "chunk_id" : i,
                    "file_hash": file_hash
                }
            })

    return all_chunks