# ─────────────────────────────────────────────
# Stage 1: Builder — install Python dependencies
# ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tools needed to compile some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*


# Copy and install Python dependencies into a prefix directory
COPY ../requirements.txt .
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       -r requirements.txt


# ─────────────────────────────────────────────
# Stage 2: Runtime image
# ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# ── System dependencies ───────────────────────
# - poppler-utils    : required by pdf2image (convert_from_path)
# - tesseract-ocr    : required by pytesseract
# - tesseract-ocr-eng: English language pack
# - libgl1           : required by OpenCV (cv2)
# - libglib2.0-0     : required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Copy installed Python packages from builder ──
COPY --from=builder /install /usr/local

# ── Download spaCy model ──────────────────────
# chunking.py lazy-loads en_core_web_sm for sentence splitting
RUN python -m spacy download en_core_web_sm

# ── Pre-download HuggingFace models ──────────
# Bakes both models into the image so the container has no
# runtime dependency on the internet.
# Model names match the defaults in config.py and embeddings.py.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('intfloat/multilingual-e5-base'); \
CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')"

# ── Copy application source ───────────────────
COPY . .

# ── Non-root user for security ────────────────
RUN adduser --disabled-password --gecos '' appuser \
    && chown -R appuser:appuser /app
USER appuser

# ── Runtime config ────────────────────────────
EXPOSE 8000

# Matches the hardcoded path in extractor.py
ENV TESSERACT_CMD=/usr/bin/tesseract

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]