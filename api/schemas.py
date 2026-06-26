"""Modelos para la API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    indexed_documents: int
    indexed_nodes: int
    total_documents: int
    collection_name: str


class UploadResponse(BaseModel):
    file_name: str
    size_bytes: int


class DocumentItem(BaseModel):
    file_name: str
    size_bytes: int


class DocumentsListResponse(BaseModel):
    documents: list[DocumentItem]


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=1, description="Identificador de conversacion.")
    message: str = Field(..., min_length=1, description="Pregunta del usuario.")


class SourceItem(BaseModel):
    file_name: str
    page_label: str
    score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
