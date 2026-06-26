"""Configuracion centralizada para el chatbot RAG de Arma Tu Turismo."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.nvidia import NVIDIA

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
DOCSTORE_PATH = CHROMA_DIR / "docstore.json"

CHROMA_COLLECTION = "arma_turismo_docs"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

DEFAULT_LLM_PROVIDER = "nvidia"
DEFAULT_LLM_MODEL = "meta/llama-3.1-8b-instruct"
DEFAULT_EMBED_MODEL = "baai/bge-m3"
DEFAULT_EMBED_BATCH_SIZE = 32

DEFAULT_CHAT_SIMILARITY_TOP_K = 5
_MAX_CHAT_SIMILARITY_TOP_K = 20

DEFAULT_CHAT_MEMORY_TOKEN_LIMIT = 3500
DEFAULT_SESSION_TTL_HOURS = 24
DEFAULT_LOG_RETENTION_DAYS = 30

DEFAULT_CORS_ORIGINS = "http://localhost:8080,http://localhost:5173,http://localhost:5174"

_CONFIGURED: bool = False


def _load_env() -> None:
    load_dotenv()


def get_cors_origins() -> list[str]:
    _load_env()
    raw = os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def get_internal_api_key() -> str | None:
    _load_env()
    value = os.getenv("INTERNAL_API_KEY", "").strip()
    return value or None


def get_chat_similarity_top_k() -> int:
    _load_env()
    raw = int(os.getenv("CHAT_SIMILARITY_TOP_K", str(DEFAULT_CHAT_SIMILARITY_TOP_K)))
    return max(1, min(_MAX_CHAT_SIMILARITY_TOP_K, raw))


def get_chat_memory_token_limit() -> int:
    _load_env()
    raw = int(os.getenv("CHAT_MEMORY_TOKEN_LIMIT", str(DEFAULT_CHAT_MEMORY_TOKEN_LIMIT)))
    return max(1000, min(6000, raw))


def get_session_ttl_seconds() -> int:
    _load_env()
    hours = float(os.getenv("SESSION_TTL_HOURS", str(DEFAULT_SESSION_TTL_HOURS)))
    return max(1, int(hours * 3600))


def get_log_retention_days() -> int:
    _load_env()
    raw = int(os.getenv("LOG_RETENTION_DAYS", str(DEFAULT_LOG_RETENTION_DAYS)))
    return max(1, raw)


def ensure_runtime_directories() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "logs").mkdir(parents=True, exist_ok=True)


def get_nvidia_api_key() -> str:
    _load_env()
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise ValueError("Variable de entorno NVIDIA_API_KEY no configurada.")
    return api_key


def get_llm_provider() -> str:
    _load_env()
    return os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()


def get_llm_model() -> str:
    _load_env()
    return os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL).strip()


def configure_settings() -> None:
    """Configura Settings de LlamaIndex una unica vez."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    ensure_runtime_directories()
    nvidia_api_key = get_nvidia_api_key()
    embed_model_name = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL).strip()
    embed_batch_size = int(os.getenv("EMBED_BATCH_SIZE", str(DEFAULT_EMBED_BATCH_SIZE)))
    llm_provider = get_llm_provider()
    llm_model = get_llm_model()

    if llm_provider != "nvidia":
        raise ValueError(
            f"LLM_PROVIDER '{llm_provider}' no soportado. Usa LLM_PROVIDER=nvidia."
        )

    Settings.llm = NVIDIA(model=llm_model, api_key=nvidia_api_key)
    Settings.embed_model = NVIDIAEmbedding(
        model=embed_model_name,
        api_key=nvidia_api_key,
        embed_batch_size=embed_batch_size,
        truncate="END",
    )
    _CONFIGURED = True
