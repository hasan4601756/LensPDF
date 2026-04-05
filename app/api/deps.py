# app/api/deps.py
from fastapi import Request
from app.core.embeddings import EmbeddingPipeline
from app.core.reranker import ReRanker
from app.core.opensearch_client import OpenSearchClient

def get_embedding_pipeline(request: Request) -> EmbeddingPipeline:
    return request.app.state.embedding_pipeline

def get_reranker(request: Request) -> ReRanker:
    return request.app.state.reranker

def get_opensearch_client(request: Request) -> OpenSearchClient:
    return request.app.state.opensearch_client