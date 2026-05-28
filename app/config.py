from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Where the source books (PDF / CSV) live.
    data_dir: Path = BASE_DIR / "data"
    # Where the built search index is persisted.
    index_dir: Path = BASE_DIR / "index"

    # Chunking
    chunk_size: int = 900
    chunk_overlap: int = 150

    # Retrieval
    top_k: int = 5

    # Retriever backend: "tfidf" (no model needed) or "ollama" (semantic embeddings).
    retriever: str = "tfidf"

    # Ollama (local/open model). Used for generation, and optionally embeddings.
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_timeout: float = 120.0


settings = Settings()
