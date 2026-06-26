"""Helpers para la construccion de retrievers."""

from __future__ import annotations

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core import VectorStoreIndex


def build_retriever(
    index: VectorStoreIndex,
    similarity_top_k: int = 5,
) -> BaseRetriever:
    """Crea un retriever configurado para busqueda semantica de similitud.

    Args:
        index: Indice vectorial de origen.
        similarity_top_k: Numero de nodos candidatos a recuperar.

    Returns:
        BaseRetriever: Instancia de retriever configurada.
    """
    return index.as_retriever(similarity_top_k=similarity_top_k)
