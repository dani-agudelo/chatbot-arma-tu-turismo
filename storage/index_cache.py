"""Cache en memoria del indice vectorial compartido."""

from __future__ import annotations

from threading import Lock

from llama_index.core import VectorStoreIndex

from config import CHROMA_COLLECTION
from storage.index_store import get_or_create_index

_INDEX: VectorStoreIndex | None = None
_LOCK = Lock()


def get_shared_index(collection_name: str = CHROMA_COLLECTION) -> VectorStoreIndex:
    """Devuelve un VectorStoreIndex singleton por proceso."""
    global _INDEX

    with _LOCK:
        if _INDEX is None:
            _INDEX = get_or_create_index(collection_name=collection_name)
        return _INDEX


def reset_shared_index() -> None:
    """Invalida el indice cacheado tras una re-ingesta."""
    global _INDEX
    with _LOCK:
        _INDEX = None
