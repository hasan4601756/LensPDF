from pydantic import BaseModel
from typing import List, Dict

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    text: str
    score: float
    metadata: Dict