"""Servicio de chat RAG."""

from __future__ import annotations

import logging
import time

from api.schemas import ChatRequest, ChatResponse, SourceItem
from generation.query_engine import get_chat_engine
from retrieval.postprocessor import extract_source_metadata

logger = logging.getLogger(__name__)


class ChatService:
    """Orquesta una consulta conversacional."""

    def reply(self, request: ChatRequest) -> ChatResponse:
        started = time.perf_counter()
        chat_engine = get_chat_engine(session_id=request.session_id)
        response = chat_engine.chat(request.message)
        sources_raw = extract_source_metadata(getattr(response, "source_nodes", None))
        sources = [SourceItem(**item) for item in sources_raw]

        logger.info(
            "event=chat_complete session_id=%s sources=%s latency_seconds=%.3f",
            request.session_id,
            len(sources),
            time.perf_counter() - started,
        )
        return ChatResponse(answer=str(response.response), sources=sources)
