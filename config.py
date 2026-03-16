"""
Configuration management using pydantic-settings.
Loads from .env file and environment variables.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # Telegram
    telegram_bot_token: str = Field(..., env="TELEGRAM_BOT_TOKEN")

    # Groq LLM
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field("llama-3.3-70b-versatile", env="GROQ_MODEL")

    # Astra DB
    astra_db_application_token: str = Field(..., env="ASTRA_DB_APPLICATION_TOKEN")
    astra_db_api_endpoint: str = Field(..., env="ASTRA_DB_API_ENDPOINT")
    astra_db_collection: str = Field("research_knowledge", env="ASTRA_DB_COLLECTION")

    # Embedding
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    jina_api_key: str = Field("jina_79ca9069d7fc447d95393f9c62923508WpA_r8vIunAs_jKjI8-G9D62Tz6b", env="JINA_API_KEY")

    # API
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")

    # LLM
    max_context_tokens: int = Field(8000, env="MAX_CONTEXT_TOKENS")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
