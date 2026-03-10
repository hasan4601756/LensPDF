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
            verify_certs=settings.OPENSEARCH_VERIFY_CERTS,
        )
        self.index_name = settings.OPENSEARCH_INDEX

        if not self.ping():
            raise ConnectionError(f"Could not connect to OpenSearch at {settings.OPENSEARCH_HOST}")


    def create_index(self, vector_dimension: int = 768):
        if not isinstance(vector_dimension, int) or vector_dimension <= 0:
            return {"success": False, "message": "Invalid vector_dimension, must be a positive integer."}
        
        try:
            if not self.client.ping():
                return {"success": False, "message": "Elasticsearch client not reachable."}
        except Exception as e:
            return {"success": False, "message": f"Error pinging Elasticsearch: {e}"}
        
        if self.client.indices.exists(index=self.index_name):
            logger.info(f"Index '{self.index_name}' already exists.")

            return {"success": True, "message": "The index already exists."}

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
                    "content": {
                        "type": "text",
                        "analyzer": "standard"
                    },
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": vector_dimension,
                        "method": {
                            "name": "hnsw",
                            "space_type": "l2", # or "cosinesimilarity"
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
            logger.info(f"Creating index '{self.index_name}' with vector dimension {vector_dimension}.")
            pipeline_result =  self.__create_search_pipeline()

            if (pipeline_result.get('result') == False):
                return {"success": False, "message": "Pipeline Creation Failed!"}
            logger.info(f"Index '{self.index_name}' and Search Pipeline created successfully.")
        except Exception as e:
            logger.error(f"Failed to create index '{self.index_name}': {e}")
            return {"success": False, "message": str(e)}

        return {"success": True, "message": f"Index '{self.index_name}' created successfully.", "data": result}


    def bulk_index(self, documents: List[Dict[str, Any]], file_metadata):
        filename = file_metadata.get("filename")
        file_hash = file_metadata.get("file_hash")

        if not documents:
            return {"success": False, "chunks_indexed": 0, "message": "No document Passed (None)"}
        
        try:
            if self.file_already_indexed(file_hash):
                logger.info(f"Document '{filename}' with hash {file_hash} is already indexed. Skipping.")
                return {"success": False, "chunks_indexed": 0, "message": "File Already Indexed!"}

            if self.filename_already_indexed(filename):
                logger.info(f"Document with filename:'{filename}' is already indexed")
                return {"success": False, "chunks_indexed": 0, "message": "File with this name is  already Indexed!"}
                # self.delete_by_file_name(filename)

            actions = []

            for doc in documents:
                # Safely get metadata fields to avoid KeyErrors
                meta = doc.get("metadata", {})
                page_num = meta.get("page_number", "0")
                chunk_id = meta.get("chunk_id", "0")
                
                custom_id = f"{filename}_{page_num}_{chunk_id}"

                actions.append({
                    "_index": self.index_name,
                    "_id": custom_id, 
                    "_source": {
                        "content": doc.get("content"),
                        "embedding": doc.get("embedding"),
                        "document_id": file_hash,
                        "metadata": {**file_metadata, **meta}
                    },
                })
                
            success_count, errors = helpers.bulk(self.client, actions)

            if errors:
                logger.error(f"Bulk indexing partially failed for {filename} therefore removing all of them. Errors: {len(errors)}")
                delete_result = self.delete_by_file_name(file_metadata.get('filename', ''))

                if delete_result['success'] == False:
                    for action in actions:
                        doc_id = action['_id']
                        self.client.delete(index='myindex', id=doc_id, ignore=[404])


                return {
                    "success": False, 
                    "chunks_indexed": success_count, 
                    "filename": filename, 
                    "message": str(errors[0])
                }
            
            logger.info(f"Successfully indexed {success_count} chunks for '{filename}'.")
            return {"success": True, "chunks_indexed": success_count, "filename": filename, "message": None}

        except Exception as e:
            logger.exception(f"Unexpected error indexing document {filename}: {str(e)}")
            return {"success": "False", "chunks_indexed": 0, "filename": filename, "message": str(e)}


    def hybrid_search(self, query_text: str, query_vector: List[float], top_k: int = 5, metadata_filters: Dict[str, Any] = None):
        must_filters = []
        if metadata_filters:
            for key, value in metadata_filters.items():
                must_filters.append({"term": {f"metadata.{key}": value}})

        search_body = {
            "size": top_k,
            "_source": {"excludes": ["embedding", "metadata.file_hash", "metadata.chunk_id"]},
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

        if must_filters:
            search_body["query"]["hybrid"]["filter"] = {"bool": {"must": must_filters}}

        params = {"search_pipeline": "hybrid-search-pipeline"}

        try:
            response = self.client.search(
                index=self.index_name,
                body=search_body,
                params=params
            )

            return {
                "success": True,
                "search_result": [
                    {
                        "document_id": hit["_source"].get("document_id", ""),
                        "content": hit["_source"].get("content", ""), 
                        "metadata": hit["_source"].get("metadata", {}),
                        "score": hit["_score"]
                    }
                    for hit in response["hits"]["hits"]
                ]}
        except Exception as e:
            logger.error('Search Failed for query: {query_text} with Error {str(e)}')
            return {'success': False, "message": "Query search Failed!"}
    
    def ping(self) -> bool:
        try:
            return self.client.ping()
        except Exception as e:
            raise e
        




    """   Internal Functions   """
        
    def __create_search_pipeline(self, pipeline_name = "hybrid-search-pipeline",
        pipeline_body = {
            "description": "Post-processor for hybrid search scoring",
            "phase_results_processors": [
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
        ):

        try:
            existing = self.client.transport.perform_request("GET", f"/_search/pipeline/{pipeline_name}")
            if existing.get(pipeline_name):
                logger.info(f"Pipeline '{pipeline_name}' already exists. Skipping creation.")
                return {"success": True, "message": f"Pipeline '{pipeline_name}' already exists."}
        except Exception:
            pass

        try:
            self.client.transport.perform_request(
                "PUT", f"/_search/pipeline/{pipeline_name}", body=pipeline_body
            )
            logger.info(f"Pipeline '{pipeline_name}' created/updated successfully.")
            return {"success": True, "message": f"Pipeline '{pipeline_name}' created successfully."}
        except Exception as e:
            logger.error(f"Failed to create/update pipeline '{pipeline_name}': {e}")
            return {"success": False, "message": str(e)}
        
    def delete_by_file_name(self, filename: str):

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
            return {"success": True, "message": "Deletion Successfull"}
        except Exception as e:
            logger.error(f"Error cleaning up old version: {e}")
            return {"success": False, "message": str(e)}


    def file_already_indexed(self, file_hash: str) -> bool:
        query = {
            "query": {
                "term": {
                    "metadata.file_hash": file_hash 
                }
            },
            "_source": False, 
            "size": 1
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

    

"""   Lazy Loading for the client   """
client = None

def get_client():
    global client

    if client is None:
        client = OpenSearchClient()
    return client