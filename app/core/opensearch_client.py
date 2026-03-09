from typing import List, Dict, Any
from opensearchpy import OpenSearch, helpers
from app.config import settings
import logging


logger = logging.getLogger(__name__)


class OpenSearchClient:
    def __init__(self):
        self.client = OpenSearch(
            hosts=[settings.OPENSEARCH_HOST],
            http_auth=(
                settings.OPENSEARCH_USERNAME,
                settings.OPENSEARCH_PASSWORD,
            ) if settings.OPENSEARCH_USERNAME else None,
            use_ssl=settings.OPENSEARCH_USE_SSL,
            verify_certs=False,
        )
        self.index_name = settings.OPENSEARCH_INDEX

    # ---------------------------------------------------------
    # INDEX MANAGEMENT
    # ---------------------------------------------------------

    def create_index(self, vector_dimension: int = 768):
        """
        Creates index with vector mapping and metadata properties for hybrid search.
        """
        if self.client.indices.exists(index=self.index_name):
            logger.info(f"Index '{self.index_name}' already exists.")

            return {"success": False, "massage": "The index already exists."}

        index_body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": "100"
                }
            },
            "mappings": {
                "properties": {
                    "document_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "content": {
                        "type": "text",
                        "analyzer": "standard" # BM25 uses this for text search
                    },
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": vector_dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2", # or "cosinesimilary"
                            "engine": "lucene",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 16
                            }
                        }
                    },
                    # Metadata mappings
                    "metadata": {
                        "properties": {
                            "file_hash": {"type": "keyword"},
                            "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                            "author": {"type": "keyword"},
                            "creation_date": {"type": "date"},
                            "filename": {"type": "keyword"},
                            "page_number": {"type": "integer"},
                            "chunk_id": {"type": "integer"}
                        }
                    }
                }
            },
        }

        try:
            result = self.client.indices.create(index=self.index_name, body=index_body)
        except Exception as e:
            return e
        
        self._create_search_pipeline()
        
        logger.info(f"Index '{self.index_name}' and Search Pipeline created successfully.")

        return result

    def _create_search_pipeline(self):
        """
        Creates a search pipeline to normalize and combine scores for Hybrid Search.
        """
        pipeline_name = "hybrid-search-pipeline"
        pipeline_body = {
            "description": "Post-processor for hybrid search scoring",
            "phase_results_processors": [
                {
                    "normalization-processor": {
                        "normalization": {"technique": "min_max"},
                        "combination": {
                            "technique": "arithmetic_mean",
                            "parameters": {"weights": [0.3, 0.7]} # 0.3 Keyword, 0.7 Semantic
                        }
                    }
                }
            ]
        }

        try:
            self.client.transport.perform_request(
                "PUT", f"/_search/pipeline/{pipeline_name}", body=pipeline_body
            )
        except Exception as e:
            print(f"Error with search pipeline, Error: {e}")

    # ---------------------------------------------------------
    # BULK INSERT
    # ---------------------------------------------------------

    def bulk_index(self, documents: List[Dict[str, Any]]):
        if not documents:
            return

        # Extract info from the first document's metadata to check for existence
        # Based on your previous structure: doc["metadata"]["filename"]
        meta = documents[0].get("metadata", {})
        filename = meta.get("filename")
        file_hash = meta.get("file_hash")

        # 1. Exact Binary Match -> Stop
        if self.file_already_indexed(file_hash):
            logger.info(f"Document '{filename}' with hash {file_hash} is already indexed. Skipping.")
            return

        # 2. Filename Match but different hash -> It's an update, delete old version
        if self.filename_already_indexed(filename):
            logger.info(f"Document '{filename}' has changed. Removing old version before re-indexing.")
            return
            # self.remove_old_version(filename)

        # 3. Proceed with indexing
        actions = []
        for doc in documents:
            # We use a unique ID combining filename and chunk to prevent collisions
            # Format: filename_page_chunk
            custom_id = f"{filename}_{doc['metadata'].get('page_number')}_{doc['metadata'].get('chunk_id')}"
            
            actions.append({
                "_index": self.index_name,
                "_id": custom_id, 
                "_source": doc,
            })

        helpers.bulk(self.client, actions)
        logger.info(f"Successfully indexed {len(documents)} chunks for '{filename}'.")

    # ---------------------------------------------------------
    # VECTOR SEARCH
    # ---------------------------------------------------------

    def hybrid_search(self, query_text: str, query_vector: List[float], top_k: int = 5, metadata_filters: Dict[str, Any] = None):
        must_filters = []
        if metadata_filters:
            for key, value in metadata_filters.items():
                must_filters.append({"term": {f"metadata.{key}": value}})

        search_body = {
            "size": top_k,
            "_source": {"excludes": ["embedding", "file_hash", "chunk_id"]},
            "query": {
                "hybrid": {
                    "queries": [
                        {
                            "match": {
                                "content": {
                                    "query": query_text
                                }
                            }
                        },
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_vector,
                                    "k": top_k
                                }
                            }
                        }
                    ]
                }
            }
        }

        # Add filters if they exist
        if must_filters:
            search_body["query"]["hybrid"]["filter"] = {"bool": {"must": must_filters}}

        # Use the pipeline we created in create_index
        params = {"search_pipeline": "hybrid-search-pipeline"}

        response = self.client.search(
            index=self.index_name,
            body=search_body,
            params=params
        )

        return [
        {
            # Use .get() to avoid KeyError if the field is missing
            "content": hit["_source"].get("content", ""), 
            "metadata": hit["_source"].get("metadata", {}),
            "score": hit["_score"]
        }
        for hit in response["hits"]["hits"]
    ]
    
    # ---------------------------------------------------------
    # DELETE DOCUMENT
    # ---------------------------------------------------------

    def delete_by_file_name(self, filename: str):
        """
        Delete all chunks belonging to a document.
        """

        query = {
            "query": {
                "term": {
                    "metadata.filename": filename
                }
            }
        }

        try:
            self.client.delete_by_query(
                index=self.index_name,
                body=query
            )

            logger.info(f"Deleted document '{filename}' from index.")
        except Exception as e:
            logger.error(f"Error cleaning up old version: {e}")


    def remove_old_version(self, filename: str):
        
        query = {
            "query": {
                "term": {
                    "metadata.filename": filename
                }
            }
        }
        try:
            self.client.delete_by_query(index=self.index_name, body=query)
            logger.info(f"Cleaned up previous version of {filename}")
        except Exception as e:
            logger.error(f"Error cleaning up old version: {e}")


    def file_already_indexed(self, file_hash: str) -> bool:
        """
        Queries OpenSearch to see if any chunk contains this file hash.
        """
        query = {
            "query": {
                "term": {
                    "metadata.file_hash": file_hash 
                }
            },
            "_source": False, 
            "size": 1 # We only need to know if at least one exists
        }
        response = self.client.search(index=self.index_name, body=query)
        return response['hits']['total']['value'] > 0
    
    def filename_already_indexed(self, filename: str) -> bool:
        query = {
            "query": {
                "term": {
                    "metadata.filename": filename 
                }
            },
            "_source": False, 
            "size": 1
        }
        response = self.client.search(index=self.index_name, body=query)
        return response['hits']['total']['value'] > 0

    # ---------------------------------------------------------
    # HEALTH CHECK
    # ---------------------------------------------------------

    def ping(self) -> bool:
        """
        Check if OpenSearch is reachable.
        """
        return self.client.ping()
    
client = OpenSearchClient()