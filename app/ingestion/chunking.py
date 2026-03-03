import re
import spacy
from langdetect import detect
from transformers import AutoTokenizer

nlp = None
tokenizer = None

# nlp = spacy.load("en_core_web_sm")
# tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")

MODEL_LIMIT = 512
PREFIX = "passage: "
OVERLAP_TOKENS = 50


def get_nlp():
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_sm")
    return nlp


def get_tokenizer():
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
    return tokenizer


def get_safe_limit():
    tokenizer = get_tokenizer()
    prefix_tokens = len(tokenizer.encode(PREFIX, add_special_tokens=False))
    return MODEL_LIMIT - prefix_tokens - 5

def split_urdu_sentences(text: str):
    sentences = re.split(r'[۔?!]\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def split_english_sentences(text: str):
    nlp = get_nlp()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def truncate_to_token_limit(tokens, limit):
    if len(tokens) <= limit:
        return tokens
    return tokens[:limit]


def chunk_text(text: str):
    tokenizer = get_tokenizer()
    SAFE_LIMIT = get_safe_limit()
    if not text.strip():
        return []

    try:
        lang = detect(text)
    except:
        lang = "en"

    if lang == "ur":
        sentences = split_urdu_sentences(text)
    else:
        sentences = split_english_sentences(text)

    chunks = []

    current_tokens = []
    current_length = 0

    for sentence in sentences:
        if not sentence:
            continue

        sentence_tokens = tokenizer.encode(
            sentence,
            add_special_tokens=False
        )

        sentence_length = len(sentence_tokens)

        # Case 1: sentence itself exceeds safe limit
        if sentence_length > SAFE_LIMIT:
            # Flush current chunk first
            if current_tokens:
                chunks.append(tokenizer.decode(current_tokens))
                current_tokens = []
                current_length = 0

            # Split large sentence into multiple chunks
            stride = SAFE_LIMIT - OVERLAP_TOKENS
            start = 0

            while start < sentence_length:
                end = start + SAFE_LIMIT
                piece = sentence_tokens[start:end]
                chunks.append(tokenizer.decode(piece))

                if end >= sentence_length:
                    break

                start += stride

            continue


        # Case 2: sentence fits in current chunk
        if current_length + sentence_length <= SAFE_LIMIT:
            current_tokens.extend(sentence_tokens)
            current_length += sentence_length

        else:
            # Flush current chunk
            if current_tokens:
                chunks.append(tokenizer.decode(current_tokens))

            # Build overlap
            overlap_tokens = current_tokens[-OVERLAP_TOKENS:]

            # Try overlap + sentence
            new_tokens = overlap_tokens + sentence_tokens

            if len(new_tokens) <= SAFE_LIMIT:
                current_tokens = new_tokens
                current_length = len(new_tokens)
            else:
                # Drop overlap if it causes overflow
                current_tokens = sentence_tokens
                current_length = sentence_length

    # Final flush
    if current_tokens:
        chunks.append(tokenizer.decode(current_tokens))

    return chunks