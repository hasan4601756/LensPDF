from app.services.document_service import document_service
import os

def document_api(file_path:str):
    if os.path.exists(file_path):
        print("Error in Document Api: file does not exist at generated path")
        return {"Succeeded": False, "Error": "Error uploading file."}
    result = document_service(file_path)

    return {"Succeeded": True, "response": result}