from pypdf import PdfReader
import easyocr
import numpy as np

class DocumentExtractor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def extract(self, file_path: str) -> str:
        text = ""
        reader = PdfReader(file_path)
        for page in reader.pages:
            content = page.extract_text()
            if content.strip():
                text += content + "\n"
            else:
                # Fallback to OCR if page is scanned/empty
                # Note: This is simplified for the demo
                pass
        return text

extractor = DocumentExtractor()