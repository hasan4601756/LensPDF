import re
import unicodedata
import ftfy
from collections import OrderedDict, Counter
from typing import List


class AdaptiveNoiseCleaner:
    """
    Learns headers/footers that frequently appear in a corpus, then removes them.
    """
    def __init__(self, frequency_threshold: float = 0.3):
        self.threshold = frequency_threshold
        self.global_noise = set()
        self.doc_count = 0

    def fit(self, all_docs: List[str]):
        self.doc_count = len(all_docs)
        temp_stats = Counter()
        
        for doc in all_docs:
            lines = [line.strip() for line in doc.split('\n') if line.strip()]
            structural_lines = lines[:3] + lines[-3:] if len(lines) > 6 else lines
            for line in set(structural_lines):
                temp_stats[line] += 1

        self.global_noise = {
            line for line, count in temp_stats.items() 
            if (count / self.doc_count) >= self.threshold
        }

    def remove_adaptive_noise(self, text: str) -> str:
        lines = text.split('\n')
        cleaned_lines = [
            line for line in lines
            if line.strip() not in self.global_noise or len(line.strip()) > 100
        ]
        return "\n".join(cleaned_lines)


def clean_encoding(text: str) -> str:
    """
    Fixes unicode issues and mojibake, normalizes text, removes private use characters.
    """
    text = ftfy.fix_text(text)
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[\uE000-\uF8FF\uf000-\uf0ff]', ' ', text)
    return text


def strip_structural_noise(text: str) -> str:
    text = re.sub(r'(?m)^[^a-zA-Z0-9\s]{3,}.*$', '', text)
    return text


def remove_headers_footers(text: str) -> str:
    patterns = [
        r'PIA Safety e-Magazine\s*\|\s*Issue-\d+.*',
        r'\d+\s*\|\s*P\s*a\s*g\s*e',
        r'^\s*\d+\s*$',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE)
    return text


def fix_hyphenation(text: str) -> str:
    return re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)


def normalize_linebreaks(text: str) -> str:
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def normalize_whitespace(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def deduplicate_paragraphs(text: str) -> str:
    paragraphs = text.split('\n\n')
    seen = OrderedDict()
    for p in paragraphs:
        clean_p = p.strip()
        if clean_p and clean_p not in seen:
            seen[clean_p] = None
    return '\n\n'.join(seen.keys())


def clean_formatting_artifacts(text: str) -> str:
    text = re.sub(r'(?i)page\s+\d+(\s+of\s+\d+)?', '', text)
    text = re.sub(r'[_\-\*\=\.]{3,}', ' ', text)
    return text


def remove_short_lines(text: str, min_length: int = 2) -> str:
    lines = text.split("\n")
    cleaned = [l for l in lines if len(l.strip()) >= min_length]
    return "\n".join(cleaned)


def fix_numeric_spacing(text: str) -> str:
    text = re.sub(r'(\d)\s+\.\s+(\d)', r'\1.\2', text)
    text = re.sub(r'(\d)\s*-\s*(\d)', r'\1-\2', text)
    text = re.sub(r'(\d)\s+%', r'\1%', text)
    return text


def fix_letter_spacing(text: str) -> str:
    return re.sub(
        r'\b(?:[A-Za-z]\s){2,}[A-Za-z]\b',
        lambda m: m.group(0).replace(" ", ""),
        text
    )


def fix_common_ocr_confusions(text: str) -> str:
    text = re.sub(r'(?<=\D)0(?=\D)', 'O', text)
    text = re.sub(r'(?<=\D)1(?=\D)', 'l', text)
    return text


def remove_garbage_lines(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        if len(line.strip()) < 2:
            continue
        if re.fullmatch(r'[\W_]+', line.strip()):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


# ===================
# Preprocessing Pipelines
# ===================

def preprocess_digital_text(
    text: str,
    adaptive_cleaner: AdaptiveNoiseCleaner | None = None
) -> str:
    text = clean_encoding(text)
    text = fix_hyphenation(text)
    text = remove_headers_footers(text)
    if adaptive_cleaner:
        text = adaptive_cleaner.remove_adaptive_noise(text)
    text = clean_formatting_artifacts(text)
    text = strip_structural_noise(text)
    text = normalize_linebreaks(text)
    text = remove_short_lines(text)
    text = deduplicate_paragraphs(text)
    text = fix_numeric_spacing(text)
    text = normalize_whitespace(text)
    return text


def preprocess_ocr_text(text: str) -> str:
    text = clean_encoding(text)
    text = fix_hyphenation(text)
    text = fix_letter_spacing(text)
    text = fix_common_ocr_confusions(text)
    text = remove_garbage_lines(text)
    text = normalize_linebreaks(text)
    text = fix_numeric_spacing(text)
    text = normalize_whitespace(text)
    return text


def preprocess_urdu_ocr_text(text: str) -> str:
    text = clean_encoding(text)
    text = re.sub(r"[\^~\|&]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def preprocess_page(page: dict) -> str:
    text = page["text"]
    source = page["source_type"]

    text = preprocess_digital_text(text)

    if source == "ocr":
        text = preprocess_ocr_text(text)

    elif source == "urdu_ocr":
        text = preprocess_urdu_ocr_text(text)

    return text