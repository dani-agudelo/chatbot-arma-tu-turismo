"""Utilidades para cargar documentos de turismo (PDF, TXT, MD)."""

from __future__ import annotations

from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

from config import DATA_DIR

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


def file_name_for_citation(value: str | Path | None, fallback: str = "desconocido") -> str:
    if value is None:
        return fallback
    s = str(value).strip()
    if not s:
        return fallback
    name = Path(s.replace("\\", "/")).name
    return name if name else fallback


def list_document_files(data_dir: Path = DATA_DIR) -> list[Path]:
    files: list[Path] = []
    for path in sorted(data_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
    return files


def load_documents_from_paths(file_paths: list[Path]) -> list[Document]:
    if not file_paths:
        return []

    reader = SimpleDirectoryReader(
        input_files=[str(path) for path in file_paths],
        filename_as_id=True,
    )
    documents = reader.load_data()
    for doc in documents:
        meta = dict(doc.metadata or {})
        raw_name = meta.get("file_name") or meta.get("file_path")
        if raw_name:
            meta["file_name"] = file_name_for_citation(raw_name)
        meta.pop("file_path", None)
        doc.metadata = meta
    return documents


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    file_paths = list_document_files(data_dir=data_dir)
    documents = load_documents_from_paths(file_paths)

    if not documents:
        raise ValueError(
            f"No se encontraron documentos ({', '.join(sorted(SUPPORTED_SUFFIXES))}) en {data_dir}."
        )
    return documents
