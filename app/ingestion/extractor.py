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

def has_large_image(page):
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

def has_fonts(page):
    if "/Resources" in page:
        resources = page["/Resources"]
        return "/Font" in resources
    return False

def text_length(page):
    text = page.extract_text()
    if not text:
        return 0
    return len(text.strip())

def is_scanned(page, min_text_chars=50):
    text = page.extract_text()
    len_text = text_length(page)
    has_img = has_large_image(page)
    has_font = has_fonts(page)

    if len_text < min_text_chars and has_img and not has_font:
        return True
    return False

def is_valid_english_ocr(text, min_word_count=10, max_error_ratio=0.5):
    spell = SpellChecker()
    words = re.findall(r"[A-Za-z]{3,}", text.lower())
    if len(words) < min_word_count:
        return False
    misspelled = spell.unknown(words)
    return len(misspelled)/len(words) <= max_error_ratio

def is_urdu_text(text):
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

    # # Binarize
    # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return thresh
    
def extract_text_from_scanned_page(page):
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

def extract_text_from_scanned_urdu_page(page, reader):
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


def read_from_pdf(pdf_path, reader):
    try:
        filename = os.path.basename(pdf_path)
        pdf_reader = PdfReader(pdf_path)

        pages = []

        for i, page in enumerate(pdf_reader.pages):
            if is_scanned(page):
                try:
                    image = convert_from_path(pdf_path, first_page=i+1, last_page=i+1, dpi=300)[0]
                    if not image:
                        print(f"Failed to convert page {i+1} into image.")
                    text = extract_text_from_scanned_page(image)

                    if is_urdu_text(text):
                        text = extract_text_from_scanned_urdu_page(image, reader)

                    pages.append(text)

                except exceptions.PDFInfoNotInstalledError:
                    print(f"Poppler is not installed or not working. Please install Poppler.")
                    break
                except exceptions.PDFPageCountError as e:
                    print(f"Error: PDF has no pages or incorrect page count. {e}")
                    break
                except Exception as e:
                    print(f"Error converting page {i+1} to image: {e}")
                    pages.append("")  # Append an empty string to keep processing other pages.
            else:
                page_content = page.extract_text()
                pages.append(page_content)

        full_text = "\n".join(pages)
        meta = pdf_reader.metadata
        metadata = {
            "title": getattr(meta, "title", None),
            "author": getattr(meta, "author", None),
            "creation_date": getattr(meta, "creation_date", None),
            "filename": filename
        }

        return {
        "text": full_text,
        "metadata": metadata
    }

    except PdfReadError as e:
        print("Error reading file.")
    except FileNotFoundError:
        print("File not found.")
        return None
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None


def extract_text(pdf_path):
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    reader = easyocr.Reader(['ur'], gpu=False)
    try:
        # return read_simple_pdf(pdf_path)
        # return extract_text_from_scanned_pdf(pdf_path)
        # return extract_text_from_scanned_urdu_pdf(pdf_path, reader)
        return read_from_pdf(pdf_path, reader)
    except PdfReadError:
        print("Invalid PDF file")