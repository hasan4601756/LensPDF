from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from app.services.search_service import search_service
from app.services.document_service import document_service
import shutil, uuid
import os

app = FastAPI()

@app.get('/')
def main():
    return "Hello"


job_store = {}

@app.post("/api/documents")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = str(uuid.uuid4())
    
    temp_path = f"temp_{job_id}_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job_store[job_id] = {"status": "queued", "filename": file.filename}

    background_tasks.add_task(document_service_wrapper, job_id, temp_path)

    return {
        "message": "Processing started",
        "job_id": job_id
    }


async def document_service_wrapper(job_id: str, path: str):
    try:
        job_store[job_id]["status"] = "processing"

        result = document_service(path)

        if (result['success'] == False):
            raise RuntimeError(result['message'])

        job_store[job_id]["status"] = "success"
    except Exception as e:
        job_store[job_id]["status"] = "failed"
        job_store[job_id]["error"] = str(e)
    finally:
        os.remove(path)


@app.get("/api/documents/{job_id}")
async def get_status(job_id: str):
    job = job_store.get(job_id)

    if not job:
        return {"error": "job not found"}

    return job


@app.get("/api/search")
async def search(query: str, limit: int = 5):
    return search_service(query, limit)