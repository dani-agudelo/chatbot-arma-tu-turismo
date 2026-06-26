"""Servicio de ingesta de documentos."""

from __future__ import annotations

import logging
import time

from api.schemas import IngestResponse
from carga_documentos.pipeline import load_and_prepare_nodes
from config import CHROMA_COLLECTION
from generation.engine_factory import reset_shared_retriever
from generation.session_store import clear_all_sessions
from storage.index_cache import reset_shared_index
from storage.index_store import get_vector_store

logger = logging.getLogger(__name__)


class IngestService:
    """Orquesta la carga e indexacion de PDFs."""

    def run(self) -> IngestResponse:
        ingest_started = time.perf_counter()

        load_started = time.perf_counter()
        vector_store = get_vector_store(collection_name=CHROMA_COLLECTION)
        vector_store_ready_seconds = time.perf_counter() - load_started

        pipeline_started = time.perf_counter()
        documents, nodes = load_and_prepare_nodes(vector_store=vector_store)
        pipeline_seconds = time.perf_counter() - pipeline_started
        total_documents = len(documents)
        total_seconds = time.perf_counter() - ingest_started

        reset_shared_index()
        reset_shared_retriever()
        clear_all_sessions()

        logger.info(
            "event=ingest_complete collection=%s documents=%s nodes=%s "
            "vector_store_ready_seconds=%.3f pipeline_seconds=%.3f total_seconds=%.3f",
            CHROMA_COLLECTION,
            total_documents,
            len(nodes),
            vector_store_ready_seconds,
            pipeline_seconds,
            total_seconds,
        )

        return IngestResponse(
            indexed_documents=total_documents,
            indexed_nodes=len(nodes),
            total_documents=total_documents,
            collection_name=CHROMA_COLLECTION,
        )
