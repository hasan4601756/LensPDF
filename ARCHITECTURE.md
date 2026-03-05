---

#  ROOT LEVEL

---

## 1️ `README.md`

**Purpose:** Documentation

Contains:

* Project overview
* Architecture diagram
* Installation steps
* API usage
* Project structure


---

## 2️ `requirements.txt`

Lists all Python dependencies:

```
fastapi
uvicorn
pypdf
easyocr
transformers
opensearch-py
spacy
```

Used by:

```bash
pip install -r requirements.txt
```

---

## 3️`docker-compose.yaml`

Defines multi-container setup:

* FastAPI backend
* OpenSearch
* (Optional) Dashboards

Used to run full system with one command.

---

## 4️`docker/`

Contains container configuration.

### `Dockerfile`

Defines:

* Python base image
* Copy app files
* Install dependencies
* Start FastAPI

### `opensearch.yml`

Custom OpenSearch configuration.

---

## 5️`scripts/`

Administrative utilities.

Examples:

* `create_index.py`
* `reindex_documents.py`
* `bulk_ingest.py`

These are:

* Not API endpoints
* Not business logic
* Used manually by developers

---

## 6️`tests/`

Unit and integration tests.

Examples:

* Test ingestion pipeline
* Test search accuracy
* Test API endpoints

Keeps system reliable and production-ready.

---

# `app/` (Main Application Code)

This is your backend system.

---

# 🔹 `main.py`

**Entry point of FastAPI**

Responsible for:

* Creating FastAPI app
* Registering routers
* Startup events
* Middleware
* Loading models

When you run:

```bash
uvicorn app.main:app
```

Execution starts here.

---

# 🔹 `config.py`

Central configuration file.

Loads:

* OpenSearch URL
* Index name
* Embedding model name
* Environment variables

Keeps config separate from logic.

---

# `api/` (Routing Layer)

This is the HTTP layer.

Contains only:

* Request handling
* Validation
* Calling services
* Returning responses

No heavy logic here.

---

## `documents.py`

Defines:

```
POST /api/documents
```

* Accepts PDF
* Calls document_service
* Returns ingestion response

---

## `search.py`

Defines:

```
POST /api/search
```

* Accepts query
* Calls search_service
* Returns results

---

## `deps.py`

Dependency injection:

* OpenSearch client
* Embedding model
* Reranker

Used by FastAPI `Depends()`.

---

# `schemas/` (Data Contracts)

Contains Pydantic models.

Used for:

* Request validation
* Response formatting
* OpenAPI documentation

---

## `document.py`

Defines:

* DocumentIngestResponse
* DocumentMetadata

---

## `search.py`

Defines:

* SearchRequest
* SearchResult
* SearchResponse

---

# `services/` (Business Logic Layer)

This is the orchestration layer.

It connects:

* ingestion
* embeddings
* OpenSearch
* reranking

---

## `document_service.py`

Responsible for:

```
ingest_document()
```

Flow:

1. Generate document_id
2. Call ingestion pipeline
3. Store chunks
4. Return metadata

---

## `search_service.py`

Responsible for:

```
search()
```

Flow:

1. Generate query embedding
2. Perform vector search
3. Rerank results
4. Return final list

---

# `ingestion/` (Document Processing Pipeline)

This handles the document transformation pipeline.

---

## `extractor.py`

Extracts text from PDF.

* Uses pypdf for text-based PDFs
* Uses easyocr for scanned PDFs

Output:
Raw text

---

## `preprocessing.py`

Cleans text:

* Remove extra whitespace
* Normalize encoding
* Remove junk characters

Output:
Clean text

---

## `chunking.py`

Splits text into semantic chunks.

Example:

* 500 tokens per chunk
* Overlapping windows (optional)

Output:
List of chunks

---

## `pipeline.py`

Orchestrates:

```
extract → clean → chunk
```

Returns structured chunks ready for embedding.

---

# `core/` (AI & Infrastructure Layer)

Contains low-level system components.

---

## `embeddings.py`

Loads embedding model.

Provides:

```
generate_embedding(text)
```

Returns:
Vector representation

---

## `reranker.py`

Loads cross-encoder model.

Provides:

```
rerank(query, documents)
```

Reorders search results based on semantic relevance.

---

## `opensearch_client.py`

Handles:

* Connecting to OpenSearch
* Creating index
* Storing documents
* Performing vector search

Encapsulates OpenSearch logic so rest of app doesn’t care about implementation.

---

## `logging.py`

Custom logging configuration:

* Log formatting
* Log levels
* File or console output

---

# `utils/`

Small reusable utilities.

---

## `helpers.py`

Contains small generic functions:

* Generate document_id
* Build chunk_id
* Normalize whitespace
* Sanitize filename

No business logic here.

---

# How Everything Connects

---

## Ingestion Flow

```
main.py
   ↓
api/documents.py
   ↓
services/document_service.py
   ↓
ingestion/pipeline.py
   ↓
core/embeddings.py
   ↓
core/opensearch_client.py
```

---

## Search Flow

```
main.py
   ↓
api/search.py
   ↓
services/search_service.py
   ↓
core/embeddings.py
   ↓
core/opensearch_client.py
   ↓
core/reranker.py
```
