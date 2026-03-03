import time

def run_ingestion_pipeline(filepath: str):
    from .extractor import extract_text
    result = extract_text(filepath)
    file_metadata = result['metadata']

    # time.sleep(15)

    if not result or not result.get("pages"):
        return [], result.get("metadata") if result else {}

    all_chunks = []

    for page in result["pages"]:
        from .preprocessing import preprocess_page
        cleaned_text = preprocess_page(page)
        page_number = page["page_number"]
        
        from .chunking import chunk_text
        chunks = chunk_text(cleaned_text)

        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    **file_metadata, 
                    "page_number": page_number
                }
            })

    return all_chunks, result.get("metadata", {})