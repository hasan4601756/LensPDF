from app.services.search_service import search_service

def search_api(query:str, limit:int):
    result = search_service(query, limit*10)

    doc_scores = {}

    for chunk in result:
        doc_id = chunk["document_id"]
        semantic_score = chunk["score"]
        rerank_score = chunk["rerank_score"]
        doc_name = chunk["metadata"]["filename"]

        doc = doc_scores.get(doc_id)

        if doc and doc["chunks"] == 3:
            continue

        if doc is None:
            doc = {"document_name": doc_name, "chunks": 0, "semantic_score": 0.0, "rerank_score": 0.0}
            doc_scores[doc_id] = doc

        doc["chunks"] += 1
        doc["semantic_score"] += semantic_score
        doc["rerank_score"] += rerank_score

    return {"Succeeded": True, "response": doc_scores}