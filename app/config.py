from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "LensPDF Search"
    OPENSEARCH_HOST: str = "opensearch-node"
    OPENSEARCH_PORT: int = 9200
    OPENSEARCH_USER: str = "admin"
    OPENSEARCH_PASS: str = "admin"
    INDEX_NAME: str = "documents_index"
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-base"
    RERANK_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    class Config:
        env_file = ".env"

settings = Settings()