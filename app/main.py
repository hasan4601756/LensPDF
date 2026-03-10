import os
import glob
import asyncio
import gc
from fastapi import FastAPI, HTTPException
from app.ingestion.pipeline import run_ingestion_pipeline
from app.core.opensearch_client import get_opensearch_client, create_index_if_not_exists
from app.services.search_service import search_service

app = FastAPI(title="LensPDF Search")

def is_file_indexed(client, filename: str, index_name: str) -> bool:
    """Checks the Database to see if this file is already embedded."""
    try:
        query = {
            "query": {
                "match_phrase": {
                    "metadata.filename": filename
                }
            }
        }
        response = client.search(index=index_name, body=query, size=1)
        # If we find at least 1 chunk belonging to this file, it's already indexed
        return response['hits']['total']['value'] > 0
    except Exception:
        return False

@app.on_event("startup")
async def startup_event():
    print("---[MODERATION] Waiting 15s for Database to settle... ---")
    await asyncio.sleep(15)
    
    client = get_opensearch_client()
    create_index_if_not_exists(client)
    
    test_folder = "/app/tests"
    if not os.path.exists(test_folder):
        print(f"CRITICAL ERROR: Folder {test_folder} not found in container.")
        return

    pdf_files = glob.glob(os.path.join(test_folder, "*.pdf"))
    print(f"--- [AUTO-INGEST] Scanning {len(pdf_files)} files ---")
    
    for pdf_path in pdf_files:
        fname = os.path.basename(pdf_path)
        
        # --- THE FIX: ASK THE DATABASE BEFORE PROCESSING ---
        if is_file_indexed(client, fname, "documents_index"):
            print(f"[SKIPPED] {fname} is already embedded in the database.")
            continue
            
        try:
            print(f"\n>>> Processing: {fname}")
            data = await run_ingestion_pipeline(pdf_path, fname)
            
            if data:
                for item in data:
                    client.index(index="documents_index", body=item)
                print(f"[OK] Indexed {fname}")
            
            # Extreme Cleanup
            del data
            gc.collect()
            print("---[COOLDOWN] Resting 5 seconds... ---")
            await asyncio.sleep(5)

        except Exception as e:
            print(f"[FAIL] {fname}: {e}")
            continue

    print("\n--- ALL STARTUP TASKS DONE ---")
    print("API is now ready for search queries at http://localhost:8000/docs")

@app.get("/api/search")
async def search(query: str, limit: int = 5):
    try:
        results = await search_service.search(query, limit)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}