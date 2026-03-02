from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from app.ingestion.pipeline import process_document
from app.services.search_service import search_service
from app.core.opensearch_client import get_opensearch_client, create_index_if_not_exists
import shutil
import os

app = FastAPI(title="Document Search API")

@app.on_event("startup")
def startup_db_client():
    client = get_opensearch_client()
    create_index_if_not_exists(client)

@app.post("/api/documents")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process in background
    background_tasks.add_task(handle_ingestion, temp_path, file.filename)
    return {"message": "Processing started", "filename": file.filename}

async def handle_ingestion(path: str, filename: str):
    processed_chunks = await process_document(path, filename)
    client = get_opensearch_client()
    for chunk in processed_chunks:
        client.index(index="documents_index", body=chunk)
    os.remove(path)

@app.get("/api/search")
async def search(query: str, limit: int = 5):
    return await search_service.search(query, limit)