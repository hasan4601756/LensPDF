from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_path, exceptions
from pypdf.errors import PdfReadError
import easyocr
from PIL import Image
import os
import numpy as np
import cv2
import re
from spellchecker import SpellChecker
from typing import Dict, List, Any
import gc

class DocumentExtractor:
    def __init__(self):
        self.spell = SpellChecker()
        self._urdu_reader = None
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

    def get_urdu_reader(self):
        if self._urdu_reader is None:
            print("Initializing EasyOCR Urdu reader...")
            self._urdu_reader = easyocr.Reader(['ur'], gpu=False)
        return self._urdu_reader

    def has_large_image(self, page) -> bool:
        if "/Resources" not in page: return False
        resources = page["/Resources"]
        if "/XObject" not in resources: return False
        xobjects = resources["/XObject"]
        for obj in xobjects:
            xobj = xobjects[obj]
            if xobj.get("/Subtype") == "/Image":
                return True
        return False

    def is_scanned(self, page, min_text_chars=50) -> tuple[str, bool]:
        text = page.extract_text() or ""
        has_img = self.has_large_image(page)
        return text, (len(text) < min_text_chars and has_img)

    def is_valid_english_ocr(self, text, min_word_count=10, max_error_ratio=0.5) -> bool:
        words = re.findall(r"[A-Za-z]{3,}", text.lower())
        if len(words) < min_word_count:
            return False
        misspelled = self.spell.unknown(words)
        return len(misspelled)/len(words) <= max_error_ratio

    def is_urdu_text(self, text) -> bool:
        return not self.is_valid_english_ocr(text)
        
    def preprocess_image_eng(self, pil_image):
        img = np.array(pil_image)
        # Handle cases where image might already be grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        gray = cv2.medianBlur(gray, 3)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh
        
    def extract_text_from_scanned_page(self, page) -> str:
        processed_image = self.preprocess_image_eng(page)
        text = pytesseract.image_to_string(
            processed_image,
            lang='eng',
            config='--psm 6 --oem 3'
        )
        return text

    def preprocess_image_ur(self, image):
        img = np.array(image)
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        gray = cv2.convertScaleAbs(gray, alpha=1.4, beta=0)
        return gray

    def extract_text_from_scanned_urdu_page(self, page, reader) -> str:
        processed = self.preprocess_image_ur(page)
        results = reader.readtext(processed, detail=1, paragraph=True)
        extracted_text = "" 
        for (bbox, text_content) in results:
            extracted_text += text_content + "\n"
        return extracted_text

    def extract(self, pdf_path: str) -> Dict[str, Any] | None:
        try:
            filename = os.path.basename(pdf_path)
            pdf_reader = PdfReader(pdf_path)
            structured_pages: List[Dict[str, Any]] =[]

            for i, page in enumerate(pdf_reader.pages):
                page_number = i + 1
                try:
                    text, scanned = self.is_scanned(page)

                    if not scanned:
                        structured_pages.append({
                            "page_number": page_number,
                            "text": text or "",
                            "source_type": "digital"
                        })
                        continue

                    # MEMORY FIX: Reduced DPI to 150 to prevent OOM errors
                    images = convert_from_path(
                        pdf_path, dpi=150, first_page=page_number, last_page=page_number
                    )
                    
                    if not images:
                        structured_pages.append({"page_number": page_number, "text": "", "source_type": "ocr"})
                        continue

                    image = images[0]
                    ocr_text = self.extract_text_from_scanned_page(image)

                    if self.is_urdu_text(ocr_text):
                        reader = self.get_urdu_reader()
                        urdu_text = self.extract_text_from_scanned_urdu_page(image, reader)
                        structured_pages.append({
                            "page_number": page_number,
                            "text": urdu_text or "",
                            "source_type": "urdu_ocr"
                        })
                    else:
                        structured_pages.append({
                            "page_number": page_number,
                            "text": ocr_text or "",
                            "source_type": "ocr"
                        })

                    # EXTREME CLEANUP
                    del image, images
                    gc.collect()

                except exceptions.PDFInfoNotInstalledError:
                    print("Poppler is not installed.")
                    return None
                except Exception as page_error:
                    print(f"Error processing page {page_number}: {page_error}")

            meta = pdf_reader.metadata
            metadata = {
                "title": getattr(meta, "title", None) if meta else None,
                "author": getattr(meta, "author", None) if meta else None,
                "filename": filename
            }

            return {"pages": structured_pages, "metadata": metadata}

        except Exception as e:
            print(f"Unexpected error reading PDF {pdf_path}: {e}")
            return None

extractor = DocumentExtractor()