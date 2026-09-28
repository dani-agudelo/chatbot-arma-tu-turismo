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
DEFAULT_LLM_MODEL = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
DEFAULT_EMBED_BATCH_SIZE = 32

DEFAULT_CHAT_SIMILARITY_TOP_K = 5
_MAX_CHAT_SIMILARITY_TOP_K = 20

DEFAULT_CHAT_MEMORY_TOKEN_LIMIT = 3500
DEFAULT_SESSION_TTL_HOURS = 24
DEFAULT_LOG_RETENTION_DAYS = 30

DEFAULT_CORS_ORIGINS = "http://localhost:8080,http://localhost:5173,http://localhost:5174"

_CONFIGURED: bool = False
_runtime_llm_model: str | None = None
_runtime_embed_model: str | None = None
_runtime_api_key: str | None = None


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
    api_key = (_runtime_api_key or os.getenv("NVIDIA_API_KEY", "")).strip()
    if not api_key:
        raise ValueError(
            "NVIDIA_API_KEY no configurada. Defínela en el panel o en el entorno."
        )
    return api_key


def get_llm_provider() -> str:
    _load_env()
    return os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()


def get_llm_model() -> str:
    _load_env()
    if _runtime_llm_model:
        return _runtime_llm_model
    return os.getenv("LLM_MODEL", DEFAULT_LLM_MODEL).strip() or DEFAULT_LLM_MODEL


def get_embed_model() -> str:
    _load_env()
    if _runtime_embed_model:
        return _runtime_embed_model
    return os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL).strip() or DEFAULT_EMBED_MODEL


def _nvidia_llm_kwargs(model: str, api_key: str) -> dict[str, object]:
    """Apaga el razonamiento interno de modelos NVIDIA que lo traen activo."""
    lowered = model.lower()
    extra: dict[str, object] = {}
    chat_template_kwargs: dict[str, object] = {}

    if lowered.startswith("deepseek-ai/"):
        chat_template_kwargs["thinking"] = False
        extra["reasoning_effort"] = "none"

    if "nemotron" in lowered:
        chat_template_kwargs["enable_thinking"] = False

    if chat_template_kwargs:
        extra["extra_body"] = {"chat_template_kwargs": chat_template_kwargs}

    kwargs: dict[str, object] = {
        "model": model,
        "api_key": api_key,
        "is_chat_model": True,
        "context_window": 32768,
        "max_tokens": 1024,
    }
    if extra:
        kwargs["additional_kwargs"] = extra
    return kwargs


def configure_settings(*, force: bool = False) -> None:
    """Configura Settings de LlamaIndex. Con force, vuelve a aplicar modelo y clave."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    ensure_runtime_directories()
    llm_provider = get_llm_provider()
    if llm_provider != "nvidia":
        raise ValueError(
            f"LLM_PROVIDER '{llm_provider}' no soportado. Usa LLM_PROVIDER=nvidia."
        )

    try:
        nvidia_api_key = get_nvidia_api_key()
    except ValueError:
        _CONFIGURED = False
        return

    embed_model_name = get_embed_model()
    embed_batch_size = int(os.getenv("EMBED_BATCH_SIZE", str(DEFAULT_EMBED_BATCH_SIZE)))
    llm_model = get_llm_model()

    Settings.llm = NVIDIA(**_nvidia_llm_kwargs(llm_model, nvidia_api_key))
    Settings.embed_model = NVIDIAEmbedding(
        model=embed_model_name,
        api_key=nvidia_api_key,
        embed_batch_size=embed_batch_size,
        truncate="END",
    )
    _CONFIGURED = True


def apply_runtime_settings(
    *,
    llm_model: str | None = None,
    embed_model: str | None = None,
    nvidia_api_key: str | None = None,
    clear_nvidia_api_key: bool = False,
) -> dict[str, object]:
    """Aplica modelo y clave enviados por el panel. La clave vacía no borra la del entorno."""
    global _runtime_llm_model, _runtime_embed_model, _runtime_api_key

    previous_embed = get_embed_model()
    previous_llm = get_llm_model()
    changed = False

    env_key = os.getenv("NVIDIA_API_KEY", "").strip()
    active_key = (_runtime_api_key or env_key).strip()

    if clear_nvidia_api_key:
        if _runtime_api_key is not None:
            _runtime_api_key = None
            changed = True
    elif nvidia_api_key and nvidia_api_key.strip():
        incoming = nvidia_api_key.strip()
        if incoming != active_key:
            _runtime_api_key = incoming
            changed = True
        elif _runtime_api_key is None:
            _runtime_api_key = incoming

    if llm_model and llm_model.strip() and llm_model.strip() != previous_llm:
        _runtime_llm_model = llm_model.strip()
        changed = True

    embed_changed = False
    if embed_model and embed_model.strip() and embed_model.strip() != previous_embed:
        _runtime_embed_model = embed_model.strip()
        embed_changed = True
        changed = True

    if changed:
        configure_settings(force=True)

    return {
        "llm_model": get_llm_model(),
        "embed_model": get_embed_model(),
        "reindex_required": embed_changed,
        "applied": changed,
    }
