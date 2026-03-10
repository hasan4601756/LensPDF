from app.services.document_service import document_service
import os

def document_api(file_path:str):
    if not os.path.exists(file_path):
        print("Error in Document Api: file does not exist at generated path")
        return {"success": False, "message": "Error uploading file."}
    result = document_service(file_path)

    return result