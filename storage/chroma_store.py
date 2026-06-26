"""Cliente Chroma y helpers para acceder a las colecciones."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import chromadb
from chromadb.api.models.Collection import Collection

from config import CHROMA_COLLECTION, CHROMA_DIR


class ChromaClientSingleton:
    """Singleton wrapper `chromadb.PersistentClient`."""

    _client: chromadb.PersistentClient | None = None
    _lock = Lock()

    @classmethod
    def get_client(cls, path: Path = CHROMA_DIR) -> chromadb.PersistentClient:
        """Devuelve un cliente persistente Chroma singleton.

        Args:
            path: Ruta del sistema de archivos para la persistencia de Chroma.

        Returns:
            chromadb.PersistentClient: Instancia compartida del cliente persistente.
        """
        with cls._lock:
            if cls._client is None:
                cls._client = chromadb.PersistentClient(path=str(path))
        return cls._client


def get_or_create_collection(collection_name: str = CHROMA_COLLECTION) -> Collection:
    """Obtiene o crea la coleccion Chroma configurada.

    Args:
        collection_name: Nombre de la coleccion Chroma objetivo.

    Returns:
        Collection: Coleccion existente o recien creada.
    """
    client = ChromaClientSingleton.get_client()
    return client.get_or_create_collection(name=collection_name)
