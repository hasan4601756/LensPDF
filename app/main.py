from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from app.core.embeddings import EmbeddingPipeline
from app.core.reranker import ReRanker
from app.core.opensearch_client import OpenSearchClient
from app.api.search import search_api
from app.api.documents import document_api
import shutil, uuid, os
from fastapi import Depends
from app.api.deps import get_embedding_pipeline, get_reranker, get_opensearch_client


job_store = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- STARTUP: load everything once, store on app.state ----
    print("Loading embedding model...")
    app.state.embedding_pipeline = EmbeddingPipeline()

    print("Loading reranker model...")
    app.state.reranker = ReRanker()
    app.state.reranker.get_reranker()  # force load now, not on first request

    print("Connecting to OpenSearch...")
    app.state.opensearch_client = OpenSearchClient()

    print("All models loaded. Ready to serve.")
    yield
    # ---- SHUTDOWN: clean up resources ----
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

@app.post("/api/documents")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    embedding_pipeline=Depends(get_embedding_pipeline),
    client=Depends(get_opensearch_client),
):
    job_id = str(uuid.uuid4())
    temp_path = f"/tmp/{file.filename}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    job_store[job_id] = {"status": "queued", "filename": file.filename}
    background_tasks.add_task(
        document_api_wrapper, job_id, temp_path, embedding_pipeline, client
    )
    return {"message": "Processing started", "job_id": job_id}


async def document_api_wrapper(job_id: str, path: str, embedding_pipeline: EmbeddingPipeline, client: OpenSearchClient):
    try:
        job_store[job_id]["status"] = "processing"

        result = document_api(path, embedding_pipeline, client)

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
async def search(
    query: str,
    limit: int = 5,
    embedding_pipeline=Depends(get_embedding_pipeline),
    client=Depends(get_opensearch_client),
    reranker=Depends(get_reranker),
):
    return search_api(query, limit, embedding_pipeline, client, reranker)