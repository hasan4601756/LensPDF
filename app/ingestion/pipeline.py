from extractor import extract_text
from preprocessing import preprocess_page
from chunking import chunk_text

def run_ingestion_pipeline(filepath: str):
    result = extract_text(filepath)

    if not result or not result.get("pages"):
        return [], result.get("metadata") if result else {}

    all_chunks = []

    for page in result["pages"]:
        cleaned_text = preprocess_page(page)
        page_number = page["page_number"]

        chunks = chunk_text(cleaned_text)

        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "page_number": page_number
            })

    return all_chunks, result.get("metadata", {})