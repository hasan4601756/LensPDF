from typing import List, Dict, Any
from opensearchpy import OpenSearch, helpers
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class OpenSearchClient:
    def __init__(self):
        self.client = OpenSearch(
            hosts=[{'host': settings.OPENSEARCH_HOST, 'port': settings.OPENSEARCH_PORT}],
            http_auth=(
                settings.OPENSEARCH_USER,
                settings.OPENSEARCH_PASS,
            ) if settings.OPENSEARCH_USER else None,
            use_ssl=False,
            verify_certs=False,
        )
        self.index_name = settings.INDEX_NAME

    # ---------------------------------------------------------
    # INDEX MANAGEMENT
    # ---------------------------------------------------------

    def create_index(self, vector_dimension: int = 768): 
        """
        Creates index with vector mapping and metadata properties for hybrid search.
        """
        if self.client.indices.exists(index=self.index_name):
            logger.info(f"Index '{self.index_name}' already exists.")
            return {"success": False, "message": "The index already exists."}

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
                    "text": { 
                        "type": "text",
                        "analyzer": "standard" 
                    },
                    "vector_field": { 
                        "type": "knn_vector",
                        "dimension": vector_dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2", 
                            "engine": "lucene",
                            "parameters": {
                                "ef_construction": 128,
                                "m": 16
                            }
                        }
                    },
                    "metadata": {
                        "properties": {
                            "file_hash": {"type": "keyword"},
                            "title": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                            "author": {"type": "keyword"},
                            "creation_date": {"type": "date"},
                            "filename": {"type": "keyword"},
                            "page_number": {"type": "integer"},
                            "chunk_id": {"type": "integer"},
                            "source": {"type": "keyword"}
                        }
                    }
                }
            },
        }

        try:
            result = self.client.indices.create(index=self.index_name, body=index_body)
        except Exception as e:
            logger.error(f"Failed to create index: {e}")
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
            "phase_results_processors":[
                {
                    "normalization-processor": {
                        "normalization": {"technique": "min_max"},
                        "combination": {
                            "technique": "arithmetic_mean",
                            "parameters": {"weights": [0.3, 0.7]} 
                        }
                    }
                }
            ]
        }

        try:
            self.client.transport.perform_request(
                "PUT", f"/_search/pipeline/{pipeline_name}", body=pipeline_body
            )
            logger.info("Hybrid search pipeline established.")
        except Exception as e:
            logger.error(f"Error with search pipeline: {e}")

    # ---------------------------------------------------------
    # BULK INSERT
    # ---------------------------------------------------------

    def bulk_index(self, documents: List[Dict[str, Any]]):
        if not documents:
            return

        meta = documents[0].get("metadata", {})
        filename = meta.get("filename")
        file_hash = meta.get("file_hash")

        if file_hash and self.file_already_indexed(file_hash):
            logger.info(f"Document '{filename}' with hash {file_hash} is already indexed. Skipping.")
            return

        if self.filename_already_indexed(filename):
            logger.info(f"Document '{filename}' has changed. Removing old version before re-indexing.")
            return

        actions =[]
        for idx, doc in enumerate(documents):
            chunk_id = doc['metadata'].get('chunk_id', idx)
            page_num = doc['metadata'].get('page_number', 'unknown')
            custom_id = f"{filename}_{page_num}_{chunk_id}"
            
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
        must_filters =[]
        if metadata_filters:
            for key, value in metadata_filters.items():
                must_filters.append({"term": {f"metadata.{key}": value}})

        search_body = {
            "size": top_k,
            "_source": {"excludes":["vector_field"]}, 
            "query": {
                "hybrid": {
                    "queries":[
                        {
                            "match": {
                                "text": { 
                                    "query": query_text
                                }
                            }
                        },
                        {
                            "knn": {
                                "vector_field": { 
                                    "vector": query_vector,
                                    "k": top_k
                                }
                            }
                        }
                    ]
                }
            }
        }

        if must_filters:
            search_body["query"]["hybrid"]["filter"] = {"bool": {"must": must_filters}}

        params = {"search_pipeline": "hybrid-search-pipeline"}

        response = self.client.search(
            index=self.index_name,
            body=search_body,
            params=params
        )

        return[
        {
            "text": hit["_source"].get("text", ""),
            "metadata": hit["_source"].get("metadata", {}),
            "score": hit["_score"]
        }
        for hit in response["hits"]["hits"]
    ]
    
    # ---------------------------------------------------------
    # DELETE & CHECKS
    # ---------------------------------------------------------

    def delete_by_file_name(self, filename: str):
        query = {"query": {"term": {"metadata.filename.keyword": filename}}}
        try:
            self.client.delete_by_query(index=self.index_name, body=query)
            logger.info(f"Deleted document '{filename}' from index.")
        except Exception as e:
            logger.error(f"Error deleting document: {e}")

    def remove_old_version(self, filename: str):
        self.delete_by_file_name(filename)

    def file_already_indexed(self, file_hash: str) -> bool:
        query = {"query": {"term": {"metadata.file_hash": file_hash}}, "_source": False, "size": 1}
        response = self.client.search(index=self.index_name, body=query)
        return response['hits']['total']['value'] > 0
    
    def filename_already_indexed(self, filename: str) -> bool:
        query = {"query": {"term": {"metadata.filename.keyword": filename}}, "_source": False, "size": 1}
        response = self.client.search(index=self.index_name, body=query)
        return response['hits']['total']['value'] > 0

    def ping(self) -> bool:
        return self.client.ping()

# =====================================================================
# BACKWARD COMPATIBILITY BRIDGE
# (If this is missing, main.py will throw an ImportError)
# =====================================================================

os_client_instance = OpenSearchClient()

def get_opensearch_client():
    return os_client_instance.client

def create_index_if_not_exists(client=None):
    os_client_instance.create_index()