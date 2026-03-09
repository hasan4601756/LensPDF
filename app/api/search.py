from app.services.search_service import search_service

def search_api(query:str):
    result = search_service(query)

    return {"Succeeded": True, "response": result}