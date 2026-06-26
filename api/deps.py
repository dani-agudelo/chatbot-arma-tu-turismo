"""Dependencias inyectables de FastAPI."""

from __future__ import annotations

from functools import lru_cache

from services.chat_service import ChatService
from services.ingest_service import IngestService


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    return ChatService()


@lru_cache(maxsize=1)
def get_ingest_service() -> IngestService:
    return IngestService()
