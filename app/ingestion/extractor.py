from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_path, exceptions
from pypdf.errors import PdfReadError
import easyocr
from PIL import Image
from langdetect import detect
import os
import numpy as np
import cv2
import re
from spellchecker import SpellChecker

# reader = easyocr.Reader(['ur'], gpu=False)
spell = SpellChecker()

_urdu_reader = None

def get_urdu_reader():
    global _urdu_reader
    if _urdu_reader is None:
        print("Initializing EasyOCR Urdu reader...")
        _urdu_reader = easyocr.Reader(['ur'], gpu=False)
    return _urdu_reader


def has_large_image(page) -> bool:
    if "/Resources" not in page:
        return False

    resources = page["/Resources"]
    if "/XObject" not in resources:
        return False

    xobjects = resources["/XObject"]
    for obj in xobjects:
        xobj = xobjects[obj]
        if xobj.get("/Subtype") == "/Image":
            return True
    return False

def has_fonts(page) -> bool:
    if "/Resources" in page:
        resources = page["/Resources"]
        return "/Font" in resources
    return False

def text_length(page) -> int:
    text = page.extract_text()
    if not text:
        return 0
    return len(text.strip())

def is_scanned(page, min_text_chars=50) -> tuple[str, bool]:
    text = page.extract_text() or ""
    has_img = has_large_image(page)

    return text, (len(text) < min_text_chars and has_img)

def is_valid_english_ocr(text, min_word_count=10, max_error_ratio=0.5) -> bool:
    words = re.findall(r"[A-Za-z]{3,}", text.lower())
    if len(words) < min_word_count:
        return False
    misspelled = spell.unknown(words)
    return len(misspelled)/len(words) <= max_error_ratio

def is_urdu_text(text) -> bool:
    # try:
    #     if len(text) < 10:
    #         return False
    #     language = detect(text)
    #     return language != 'en'
    # except:
    #     return False

    return not is_valid_english_ocr(text)
    
def preprocess_image_eng(pil_image):
    img = np.array(pil_image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Denoise
    gray = cv2.medianBlur(gray, 3)

    # kernel = np.ones((2, 2), np.uint8)
    # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    # Binarize
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # thresh = cv2.adaptiveThreshold(
    # gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    # )
    return thresh
    
def extract_text_from_scanned_page(page) -> str:
    processed_image = preprocess_image_eng(page)

    text = pytesseract.image_to_string(
        processed_image,
        lang='eng',
        config='--psm 6 --oem 3'
    )
    return text


def preprocess_image_ur(image):
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Resize (very important for Urdu)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Light contrast boost
    gray = cv2.convertScaleAbs(gray, alpha=1.4, beta=0)

    return gray

def extract_text_from_scanned_urdu_page(page, reader) -> str:
    processed = preprocess_image_ur(page)

    results = reader.readtext(
        processed,
        detail=1,     
        paragraph=True
    )

    extracted_text = "" 
    for (bbox, text_content) in results:
        extracted_text += text_content + "\n"

    return extracted_text

import os
from typing import Dict, List, Any
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from pdf2image import convert_from_path, exceptions


def read_from_pdf(pdf_path: str) -> Dict[str, Any] | None:
    """
    Extracts text from a PDF and returns structured per-page output.

    Returns:
        {
            "pages": [
                {
                    "page_number": int,
                    "text": str,
                    "source_type": "digital" | "ocr" | "urdu_ocr"
                },
                ...
            ],
            "metadata": {...}
        }
    """

    try:
        filename = os.path.basename(pdf_path)
        pdf_reader = PdfReader(pdf_path)

        structured_pages: List[Dict[str, Any]] = []

        for i, page in enumerate(pdf_reader.pages):
            page_number = i + 1

            try:
                text, scanned = is_scanned(page)

                # DIGITAL PAGE
                if not scanned:
                    structured_pages.append({
                        "page_number": page_number,
                        "text": text or "",
                        "source_type": "digital"
                    })
                    continue

                # SCANNED PAGE → Convert only this page
                try:
                    images = convert_from_path(
                        pdf_path,
                        dpi=300,
                        first_page=page_number,
                        last_page=page_number
                    )

                    if not images:
                        structured_pages.append({
                            "page_number": page_number,
                            "text": "",
                            "source_type": "ocr"
                        })
                        continue

                    image = images[0]
                    ocr_text = extract_text_from_scanned_page(image)

                    # Detect Urdu
                    if is_urdu_text(ocr_text):
                        reader = get_urdu_reader()
                        urdu_text = extract_text_from_scanned_urdu_page(image, reader)
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

                except exceptions.PDFInfoNotInstalledError:
                    print("Poppler is not installed or not working.")
                    return None

                except exceptions.PDFPageCountError as e:
                    print(f"Invalid page count: {e}")
                    return None

            except Exception as page_error:
                print(f"Error processing page {page_number}: {page_error}")
                structured_pages.append({
                    "page_number": page_number,
                    "text": "",
                    "source_type": "unknown"
                })

        # Extract metadata
        meta = pdf_reader.metadata
        metadata = {
            "title": getattr(meta, "title", None),
            "author": getattr(meta, "author", None),
            "creation_date": getattr(meta, "creation_date", None),
            "filename": filename
        }

        return {
            "pages": structured_pages,
            "metadata": metadata
        }

    except PdfReadError:
        print("Error reading PDF.")
        return None

    except FileNotFoundError:
        print("File not found.")
        return None

    except Exception as e:
        print(f"Unexpected error reading PDF: {e}")
        return None


def extract_text(pdf_path) -> dict|None:
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    try:
        # return read_simple_pdf(pdf_path)
        # return extract_text_from_scanned_pdf(pdf_path)
        # return extract_text_from_scanned_urdu_pdf(pdf_path, reader)
        return read_from_pdf(pdf_path)
    except PdfReadError:
        print("Invalid PDF file")
        return None