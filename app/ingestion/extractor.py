import pytesseract
from pypdf import PdfReader
from pdf2image import convert_from_path
import easyocr
import numpy as np
import cv2
import re
import gc

class DocumentExtractor:
    def __init__(self):
        self._urdu_reader = None
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

    def get_urdu_reader(self):
        if self._urdu_reader is None:
            self._urdu_reader = easyocr.Reader(['ur'], gpu=False)
        return self._urdu_reader

    def is_urdu_script(self, text):
        # Lightest way to detect Urdu without using extra libraries
        return bool(re.search(r'[\u0600-\u06FF]', text))

    def extract_from_pdf(self, file_path: str):
        try:
            pdf_reader = PdfReader(file_path)
            full_text = ""
            final_source = "digital"

            # 1. Quick check for digital text
            first_txt = pdf_reader.pages[0].extract_text() or ""
            if len(first_txt.strip()) > 50:
                for page in pdf_reader.pages:
                    full_text += (page.extract_text() or "") + "\n"
                return full_text, "digital"

            # 2. Sequential OCR at LOW resolution (100 DPI) to save 80% RAM
            print(f"INFO: OCR {file_path} at 100 DPI...")
            for i in range(len(pdf_reader.pages)):
                images = convert_from_path(file_path, dpi=100, first_page=i+1, last_page=i+1)
                if not images: continue
                
                img = np.array(images[0])
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
                # Try English Tesseract first
                ocr_text = pytesseract.image_to_string(gray, lang='eng')

                if self.is_urdu_script(ocr_text) or "urdu" in file_path.lower():
                    # Fallback to EasyOCR for Urdu
                    urdu_results = self.get_urdu_reader().readtext(gray, detail=1, paragraph=True)
                    full_text += "\n".join([res[1] for res in urdu_results]) + "\n"
                    final_source = "urdu"
                else:
                    full_text += ocr_text + "\n"
                    final_source = "ocr"

                # CLEAR RAM IMMEDIATELY
                del images, img, gray
                gc.collect()

            return full_text, final_source
        except Exception as e:
            print(f"Error: {e}")
            return "", "error"

extractor = DocumentExtractor()