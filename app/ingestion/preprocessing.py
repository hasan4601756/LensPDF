import re
import unicodedata
import ftfy

def clean_encoding(text: str) -> str:
    text = ftfy.fix_text(text)
    text = unicodedata.normalize('NFKC', text)
    # Remove private use area characters
    text = re.sub(r'[\uE000-\uF8FF\uf000-\uf0ff]', ' ', text)
    return text

def remove_headers_footers(text: str) -> str:
    # Pattern for common page numbers and footers
    patterns = [
        r'\d+\s*\|\s*P\s*a\s*g\s*e', 
        r'^\s*\d+\s*$',
        r'Page \d+ of \d+'
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    return text

def normalize_whitespace(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def preprocess_text(text: str, source_type: str) -> str:
    if not text:
        return ""
    
    text = clean_encoding(text)
    
    if source_type == "digital":
        text = remove_headers_footers(text)
        # Fix hyphenation at line breaks
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    elif source_type == "urdu":
        # Remove common OCR artifacts for Urdu script
        text = re.sub(r"[\^~\|&*_]", "", text)
        # Ensure Urdu characters are preserved, but extra spaces between words are cleaned
        text = re.sub(r'\s+', ' ', text)
    
    elif source_type == "ocr":
        # English OCR specific cleaning
        text = re.sub(r'(?<=\D)0(?=\D)', 'O', text) # common 0/O confusion
        text = re.sub(r'(?<=\D)1(?=\D)', 'l', text) # common 1/l confusion

    return normalize_whitespace(text)