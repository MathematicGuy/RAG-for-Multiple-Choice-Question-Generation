"""
Configuration settings for the RAG MCQ system.
Manages environment variables and application settings.
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from pathlib import Path


class LLMSettings(BaseModel):
    """LLM-related settings"""
    provider: str = "together_ai"
    together_api_key: str = "..."
    model: str = "Qwen/Qwen2.5-7B-Instruct-Turbo"
    max_retries: int = 3
    timeout: int = 30
    temperature: float = 0.7
    max_tokens: int = 800


class EmbeddingSettings(BaseModel):
    """Embedding-related settings"""
    model: str = "bkai-foundation-models/vietnamese-bi-encoder"
    chunk_size: int = 500
    chunk_overlap: int = 50
    diversity_threshold: float = 0.7


class VectorStoreSettings(BaseModel):
    """Vector store settings"""
    type: str = "faiss"
    similarity_threshold: float = 0.3
    max_results: int = 5


class APISettings(BaseModel):
    """API-related settings"""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    title: str = "RAG MCQ Generation API"
    version: str = "1.0.0"
    docs_url: str = "/docs"
    openapi_url: str = "/openapi.json"


class QualitySettings(BaseModel):
    """Quality validation settings"""
    minimum_score: float = 60.0
    good_score: float = 75.0
    excellent_score: float = 90.0
    min_question_length: int = 10
    max_question_length: int = 200
    min_explanation_length: int = 20


class LoggingSettings(BaseModel):
    """Logging settings"""
    level: str = "INFO"
    format: str = "json"  # json or text
    file_path: Optional[str] = None
    max_file_size: str = "10MB"
    backup_count: int = 5


class Settings(BaseModel):
    """Main application settings"""
    environment: str = "development"
    secret_key: str = "your-secret-key-change-in-production"

    # Sub-settings
    llm: LLMSettings = LLMSettings()
    embedding: EmbeddingSettings = EmbeddingSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    api: APISettings = APISettings()
    quality: QualitySettings = QualitySettings()
    logging: LoggingSettings = LoggingSettings()

    # Paths
    base_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    tokens_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "tokens")
    pdfs_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "pdfs")
    tmp_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "tmp")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load from environment variables
        self._load_from_env()
        # Create directories if they don't exist
        for dir_path in [self.tokens_dir, self.pdfs_dir, self.tmp_dir]:
            dir_path.mkdir(exist_ok=True)

    def _load_from_env(self):
        """Load settings from environment variables"""
        # LLM settings
        self.llm.provider = os.getenv("LLM_PROVIDER", self.llm.provider)
        self.llm.together_api_key = os.getenv("TOGETHER_API_KEY", self.llm.together_api_key)
        self.llm.model = os.getenv("LLM_MODEL", self.llm.model)
        self.llm.max_retries = int(os.getenv("LLM_MAX_RETRIES", self.llm.max_retries))
        self.llm.timeout = int(os.getenv("LLM_TIMEOUT", self.llm.timeout))
        self.llm.temperature = float(os.getenv("LLM_TEMPERATURE", self.llm.temperature))
        self.llm.max_tokens = int(os.getenv("LLM_MAX_TOKENS", self.llm.max_tokens))

        # API settings
        self.api.host = os.getenv("API_HOST", self.api.host)
        self.api.port = int(os.getenv("API_PORT", self.api.port))
        self.api.debug = os.getenv("API_DEBUG", "false").lower() == "true"

        # Environment
        self.environment = os.getenv("ENVIRONMENT", self.environment)
        self.secret_key = os.getenv("SECRET_KEY", self.secret_key)

    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.environment.lower() == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.environment.lower() == "production"

    def get_together_api_key(self) -> str:
        """Get Together AI API key from environment or file"""
        # Try environment variable first
        api_key = self.llm.together_api_key

        # If not in env, try to read from file
        if not api_key or api_key == "...":
            token_file = self.tokens_dir / "together_api_key.txt"
            if token_file.exists():
                api_key = token_file.read_text().strip()

        if not api_key or api_key == "...":
            raise ValueError(
                "Together AI API key not found. Set TOGETHER_API_KEY environment variable "
                "or create tokens/together_api_key.txt file"
            )

        return api_key

    def get_huggingface_token(self) -> Optional[str]:
        """Get HuggingFace token from file"""
        token_file = self.tokens_dir / "hugging_face_token.txt"
        if token_file.exists():
            return token_file.read_text().strip()
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            "environment": self.environment,
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "max_retries": self.llm.max_retries,
                "timeout": self.llm.timeout,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
            },
            "embedding": {
                "model": self.embedding.model,
                "chunk_size": self.embedding.chunk_size,
                "chunk_overlap": self.embedding.chunk_overlap,
                "diversity_threshold": self.embedding.diversity_threshold,
            },
            "vector_store": {
                "type": self.vector_store.type,
                "similarity_threshold": self.vector_store.similarity_threshold,
                "max_results": self.vector_store.max_results,
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "debug": self.api.debug,
                "title": self.api.title,
                "version": self.api.version,
            },
            "quality": {
                "minimum_score": self.quality.minimum_score,
                "good_score": self.quality.good_score,
                "excellent_score": self.quality.excellent_score,
            }
        }

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings"""
    return settings
