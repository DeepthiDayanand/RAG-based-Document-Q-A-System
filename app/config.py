"""
Central configuration. Everything is overridable via environment variables
(loaded from .env if present) so nothing is hard-coded in the pipeline code.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Embedding model (local, via sentence-transformers)
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # Vector DB (Chroma, persisted to disk)
    chroma_db_dir: str = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    chroma_collection: str = os.getenv("CHROMA_COLLECTION", "documents")

    # LLM (Anthropic)
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_model: str = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
    max_tokens: int = int(os.getenv("MAX_TOKENS", "1024"))

    # Retrieval / chunking
    docs_dir: str = os.getenv("DOCS_DIR", "data/sample_docs")
    top_k: int = int(os.getenv("TOP_K", "3"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))


settings = Settings()
