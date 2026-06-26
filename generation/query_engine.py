"""API publica para obtener chat engines por sesion."""

from __future__ import annotations

from llama_index.core.chat_engine import ContextChatEngine

from generation.session_store import get_chat_engine_for_session


def get_chat_engine(
    session_id: str,
    similarity_top_k: int | None = None,
) -> ContextChatEngine:
    """Obtiene el chat engine cacheado de una sesion.

    Args:
        session_id: Identificador de conversacion.
        similarity_top_k: Mantenido por compatibilidad; el valor activo viene de config.

    Returns:
        ContextChatEngine: Motor listo para chat.
    """
    _ = similarity_top_k
    return get_chat_engine_for_session(session_id=session_id)
