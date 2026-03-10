FROM python:3.10-slim

WORKDIR /app

# 1. Install System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    poppler-utils \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 2. Torch CPU (Cached)
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 --extra-index-url https://download.pytorch.org/whl/cpu

# 3. AI Engines (Cached)
RUN pip install --no-cache-dir \
    sentence-transformers==2.3.1 \
    transformers==4.35.2 \
    easyocr==1.7.1 \
    spacy==3.7.2 \
    numpy==1.26.3

# 4. App Libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Language Model
RUN pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0.tar.gz

# 6. BAKE MULTILINGUAL MODELS (One-time download)
RUN python -c "import easyocr; easyocr.Reader(['ur', 'en'], gpu=False)"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')"

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]