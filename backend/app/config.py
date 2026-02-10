from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Application
    app_name: str = "Document Intelligence Platform"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Database
    database_url: str = "postgresql+asyncpg://docintelli:docintelli_dev@localhost:5432/docintelli"
    sync_database_url: str = "postgresql://docintelli:docintelli_dev@localhost:5432/docintelli"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # MinIO / S3
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin123"
    minio_bucket_raw: str = "raw-documents"
    minio_bucket_processed: str = "processed-documents"
    minio_use_ssl: bool = False

    # Gemini API
    gemini_api_key: str = ""

    # ML Models
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    # Document processing
    max_file_size_mb: int = 50
    chunk_size_tokens: int = 400
    chunk_overlap_pct: float = 0.15

    # RAG
    search_top_k: int = 20
    rerank_top_k: int = 10
    rag_max_context_chunks: int = 8
    similarity_threshold: float = 0.3

    # JWT Authentication
    jwt_secret_key: str = "change-me-in-production-use-openssl-rand-hex-32"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # Frontend
    frontend_url: str = "http://localhost:3000"

    # Observability
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"


settings = Settings()
