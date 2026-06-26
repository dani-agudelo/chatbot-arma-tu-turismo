"""Pipeline de carga de documentos: fragmentacion, embeddings y docstore."""

from __future__ import annotations

import logging
import time
from threading import Lock

from llama_index.core import Settings
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.ingestion.pipeline import DocstoreStrategy
from llama_index.core.schema import BaseNode, Document
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.vector_stores.types import BasePydanticVectorStore

from carga_documentos.loader import file_name_for_citation, load_documents
from carga_documentos.node_parsers import get_document_node_parser
from config import DOCSTORE_PATH

logger = logging.getLogger(__name__)

_CACHE_PIPELINE: IngestionCache | None = None
_CACHE_PIPELINE_LOCK = Lock()
_DOCSTORE: SimpleDocumentStore | None = None
_DOCSTORE_LOCK = Lock()


def obtener_cache_pipeline_documentos() -> IngestionCache:
    """Devuelve un cache de pipeline singleton para este proceso.

    Returns:
        IngestionCache: Cache compartido utilizado por la pipeline de LlamaIndex.
    """
    global _CACHE_PIPELINE
    with _CACHE_PIPELINE_LOCK:
        if _CACHE_PIPELINE is None:
            _CACHE_PIPELINE = IngestionCache()
    return _CACHE_PIPELINE


def get_docstore() -> SimpleDocumentStore:
    """Return a process singleton document store with disk persistence."""
    global _DOCSTORE
    with _DOCSTORE_LOCK:
        if _DOCSTORE is None:
            if DOCSTORE_PATH.exists():
                _DOCSTORE = SimpleDocumentStore.from_persist_path(str(DOCSTORE_PATH))
            else:
                _DOCSTORE = SimpleDocumentStore()
    return _DOCSTORE


def ejecutar_pipeline_carga_documentos(
    documents: list[Document],
    vector_store: BasePydanticVectorStore | None = None,
) -> list[BaseNode]:
    """Ejecuta la pipeline de LlamaIndex con caching para parsing y embeddings.

    Args:
        documents: Documentos de origen a transformar.

    Returns:
        list[BaseNode]: Nodos transformados listos para indexar.
    """
    pipeline = IngestionPipeline(
        transformations=[
            get_document_node_parser(),
            Settings.embed_model,
        ],
        cache=obtener_cache_pipeline_documentos(),
        docstore=get_docstore(),
        docstore_strategy=DocstoreStrategy.UPSERTS_AND_DELETE,
        vector_store=vector_store,
    )

    started = time.perf_counter()
    logger.info(
        "event=pipeline_start documents=%s batch_size=%s",
        len(documents),
        getattr(Settings.embed_model, "embed_batch_size", "?"),
    )
    nodes = pipeline.run(documents=documents, show_progress=True)
    elapsed = time.perf_counter() - started

    persist_started = time.perf_counter()
    get_docstore().persist(str(DOCSTORE_PATH))
    persist_elapsed = time.perf_counter() - persist_started

    logger.info(
        "event=pipeline_complete nodes=%s elapsed_seconds=%.2f persist_seconds=%.3f",
        len(nodes),
        elapsed,
        persist_elapsed,
    )
    return nodes


def enrich_node_metadata(
    nodes: list[BaseNode],
    documents: list[Document],
) -> list[BaseNode]:
    """Asegura que cada nodo incluya metadatos estandarizados de origen.

    Args:
        nodes: Nodos generados a partir de documentos de origen.
        documents: Documentos originales utilizados para crear los nodos.

    Returns:
        list[BaseNode]: Nodos con `file_name` y `page_label`.
    """
    documents_by_id = {
        document.doc_id: document for document in documents if getattr(document, "doc_id", None)
    }

    for node in nodes:
        source_document = documents_by_id.get(node.ref_doc_id)
        node.metadata = dict(node.metadata or {})
        node.metadata.pop("file_path", None)

        file_name = (
            node.metadata.get("file_name")
            or (source_document.metadata.get("file_name") if source_document else None)
            or "desconocido.pdf"
        )
        page_label = (
            node.metadata.get("page_label")
            or (source_document.metadata.get("page_label") if source_document else None)
            or "N/A"
        )

        node.metadata["file_name"] = file_name_for_citation(file_name)
        node.metadata["page_label"] = str(page_label)

    return nodes


def load_and_prepare_nodes(
    vector_store: BasePydanticVectorStore | None = None,
) -> tuple[list[Document], list[BaseNode]]:
    """Carga los documentos de origen y los convierte en nodos."""
    documents = load_documents()
    nodes = ejecutar_pipeline_carga_documentos(documents, vector_store=vector_store)
    enriched_nodes = enrich_node_metadata(nodes, documents)
    return documents, enriched_nodes
