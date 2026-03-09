import os
import glob
import asyncio
import gc
from fastapi import FastAPI, HTTPException
from app.ingestion.pipeline import run_ingestion_pipeline
from app.core.opensearch_client import get_opensearch_client, create_index_if_not_exists
from app.services.search_service import search_service

app = FastAPI(title="LensPDF Search API")

@app.on_event("startup")
async def startup_event():
    # 1. Wait for System to Settle
    print("--- [MODERATION] Waiting 15s for Database to settle... ---")
    await asyncio.sleep(15)
    
    client = get_opensearch_client()
    create_index_if_not_exists(client)
    
    test_folder = "/app/tests"
    if not os.path.exists(test_folder):
        print(f"CRITICAL ERROR: Folder {test_folder} not found in container.")
        return

    pdf_files = glob.glob(os.path.join(test_folder, "*.pdf"))
    print(f"--- [AUTO-INGEST] Processing {len(pdf_files)} files one by one ---")
    
    for pdf_path in pdf_files:
        fname = os.path.basename(pdf_path)
        try:
            print(f"\n>>> Processing: {fname}")
            data = await run_ingestion_pipeline(pdf_path, fname)
            
            if data:
                for item in data:
                    client.index(index="documents_index", body=item)
                print(f"[OK] Indexed {fname}")
            else:
                print(f"[-] Skipping {fname}: No text could be extracted.")
            
            # 2. Exhaustive Cleanup
            del data
            gc.collect()
            print("--- [COOLDOWN] Resting 5 seconds... ---")
            await asyncio.sleep(5)

        except Exception as e:
            print(f"[FAIL] {fname}: {str(e)}")
            continue

    print("\n--- ALL STARTUP TASKS DONE ---")
    print("API is now ready for search queries at http://localhost:8000/docs")

@app.get("/api/search")
async def search(query: str, limit: int = 5):
    try:
        return await search_service.search(query, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}