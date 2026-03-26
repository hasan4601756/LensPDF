from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    OPENSEARCH_HOST: str = "http://localhost:9200"
    OPENSEARCH_USERNAME: str | None = "admin"
    OPENSEARCH_PASSWORD: str | None = "StrongPassword123!"
    OPENSEARCH_USE_SSL: bool = False
    OPENSEARCH_INDEX: str = "default_index"
    OPENSEARCH_VERIFY_CERTS : bool = False

    RERANK_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()