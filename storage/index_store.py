"""Creacion, helpers de persistencia de indices construidos en Chroma."""

from __future__ import annotations

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import CHROMA_COLLECTION
from storage.chroma_store import get_or_create_collection


def get_vector_store(collection_name: str = CHROMA_COLLECTION) -> ChromaVectorStore:
    """Crea un wrapper de vector store Chroma desde una coleccion.

    Args:
        collection_name: Nombre de la coleccion Chroma.

    Returns:
        ChromaVectorStore: Adaptador de vector store LlamaIndex.
    """
    collection = get_or_create_collection(collection_name=collection_name)
    return ChromaVectorStore(chroma_collection=collection)


def get_or_create_index(collection_name: str = CHROMA_COLLECTION) -> VectorStoreIndex:
    """Crea un objeto indice vinculado al vector store persistido.

    Args:
        collection_name: Nombre de la coleccion Chroma.

    Returns:
        VectorStoreIndex: Indice conectado al almacenamiento persistido de Chroma.
    """
    vector_store = get_vector_store(collection_name=collection_name)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)
