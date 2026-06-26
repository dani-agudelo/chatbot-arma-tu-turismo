"""Memoria y chat engines por sesion con TTL."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from threading import Lock

from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer

from config import get_chat_memory_token_limit, get_session_ttl_seconds
from generation.engine_factory import build_chat_engine

logger = logging.getLogger(__name__)


@dataclass
class SessionEntry:
    memory: ChatMemoryBuffer
    engine: ContextChatEngine
    last_access: float


_SESSIONS: dict[str, SessionEntry] = {}
_LOCK = Lock()


def _purge_expired_sessions() -> None:
    ttl = get_session_ttl_seconds()
    now = time.time()
    expired = [
        session_id
        for session_id, entry in _SESSIONS.items()
        if now - entry.last_access > ttl
    ]
    for session_id in expired:
        _SESSIONS.pop(session_id, None)
        logger.info("event=session_expired session_id=%s", session_id)


def get_chat_engine_for_session(session_id: str) -> ContextChatEngine:
    """Obtiene o crea un chat engine cacheado para la sesion."""
    now = time.time()

    with _LOCK:
        _purge_expired_sessions()
        entry = _SESSIONS.get(session_id)

        if entry is None:
            memory = ChatMemoryBuffer.from_defaults(
                token_limit=get_chat_memory_token_limit()
            )
            engine = build_chat_engine(memory=memory)
            entry = SessionEntry(memory=memory, engine=engine, last_access=now)
            _SESSIONS[session_id] = entry
        else:
            entry.last_access = now

        return entry.engine


def clear_all_sessions() -> None:
    """Limpia engines y memorias (por ejemplo tras re-ingesta)."""
    with _LOCK:
        _SESSIONS.clear()
