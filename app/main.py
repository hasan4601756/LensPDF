from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from app.ingestion.pipeline import run_ingestion_pipeline
from app.services.search_service import search_service
from app.services.document_service import document_service
import shutil
import os

app = FastAPI()

@app.get('/')
def main():
    return "Hello"

@app.post("/api/documents")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process in background
    background_tasks.add_task(document_service, temp_path)
    return {"message": "Processing started", "filename": file.filename}

@app.get("/api/search")
async def search(query: str, limit: int = 5):
    return search_service(query, limit)
