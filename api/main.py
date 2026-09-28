"""Punto de entrada FastAPI para carga de documentos y chat de Arma Tu Turismo."""

from __future__ import annotations

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from typing import Annotated

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from api.auth import verify_internal_api_key
from api.deps import get_chat_service, get_ingest_service
from api.exceptions import register_exception_handlers
from api.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentItem,
    DocumentsListResponse,
    IngestResponse,
    UploadResponse,
)
from carga_documentos.loader import list_document_files
from config import (
    CHROMA_COLLECTION,
    DATA_DIR,
    apply_runtime_settings,
    configure_settings,
    get_cors_origins,
    get_embed_model,
    get_llm_model,
)
from generation.engine_factory import reset_shared_retriever
from generation.session_store import clear_all_sessions
from storage.index_cache import reset_shared_index
from logging_config import PUBLIC_ERROR_MESSAGE, setup_logging
from services.chat_service import ChatService
from services.ingest_service import IngestService

logger = logging.getLogger(__name__)

OPENAPI_TAGS = [
    {"name": "health", "description": "Estado de la API"},
    {"name": "carga_documentos", "description": "Carga e indexacion de documentos en ChromaDB."},
    {"name": "chat", "description": "Consultas conversacionales con RAG y citas de fuentes."},
]

ALLOWED_UPLOAD_SUFFIXES = {".pdf", ".txt", ".md"}


class RuntimeSettingsBody(BaseModel):
    llm_model: str = Field(min_length=1, max_length=128)
    embed_model: str = Field(min_length=1, max_length=128)
    nvidia_api_key: str | None = Field(default=None, max_length=512)
    clear_nvidia_api_key: bool = False


def _apply_and_reset(
    *,
    llm_model: str | None = None,
    embed_model: str | None = None,
    nvidia_api_key: str | None = None,
    clear_nvidia_api_key: bool = False,
) -> dict[str, object]:
    result = apply_runtime_settings(
        llm_model=llm_model,
        embed_model=embed_model,
        nvidia_api_key=nvidia_api_key,
        clear_nvidia_api_key=clear_nvidia_api_key,
    )
    if result.get("applied"):
        reset_shared_index()
        reset_shared_retriever()
        clear_all_sessions()
    return result


def apply_request_runtime(
    _: Annotated[None, Depends(verify_internal_api_key)],
    x_llm_model: Annotated[str | None, Header()] = None,
    x_embed_model: Annotated[str | None, Header()] = None,
    x_nvidia_api_key: Annotated[str | None, Header()] = None,
) -> None:
    if not any((x_llm_model, x_embed_model, x_nvidia_api_key)):
        return
    _apply_and_reset(
        llm_model=x_llm_model,
        embed_model=x_embed_model,
        nvidia_api_key=x_nvidia_api_key,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    configure_settings()
    setup_logging()
    logger.info("event=startup chroma_collection=%s", CHROMA_COLLECTION)
    yield


app = FastAPI(
    title="Arma Tu Turismo RAG API",
    description="Indi-Bot: servicio RAG con LlamaIndex, ChromaDB y NVIDIA NIM.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/swagger",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=OPENAPI_TAGS,
    swagger_ui_parameters={
        "displayRequestDuration": True,
        "tryItOutEnabled": True,
    },
)

register_exception_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "chatbot-arma",
        "llm_model": get_llm_model(),
        "embed_model": get_embed_model(),
    }


@app.post(
    "/settings",
    tags=["health"],
    dependencies=[Depends(verify_internal_api_key)],
)
async def update_runtime_settings(body: RuntimeSettingsBody) -> dict[str, object]:
    return _apply_and_reset(
        llm_model=body.llm_model,
        embed_model=body.embed_model,
        nvidia_api_key=body.nvidia_api_key,
        clear_nvidia_api_key=body.clear_nvidia_api_key,
    )


def _ingest_sync(service: IngestService) -> IngestResponse:
    try:
        return service.run()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _chat_sync(service: ChatService, request: ChatRequest) -> ChatResponse:
    return service.reply(request)


@app.get("/documents", response_model=DocumentsListResponse, tags=["carga_documentos"])
async def list_documents() -> DocumentsListResponse:
    documents = [
        DocumentItem(file_name=path.name, size_bytes=path.stat().st_size)
        for path in list_document_files()
    ]
    return DocumentsListResponse(documents=documents)


@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["carga_documentos"],
    dependencies=[Depends(verify_internal_api_key)],
)
async def upload_document(
    file: UploadFile = File(...),
    _: None = Depends(apply_request_runtime),
) -> UploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nombre de archivo requerido.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no permitido. Usa: {', '.join(sorted(ALLOWED_UPLOAD_SUFFIXES))}",
        )

    safe_name = Path(file.filename).name
    target = DATA_DIR / safe_name

    try:
        with target.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        await file.close()

    size = target.stat().st_size
    logger.info("event=document_uploaded file=%s size=%s", safe_name, size)
    return UploadResponse(file_name=safe_name, size_bytes=size)


@app.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["carga_documentos"],
    dependencies=[Depends(verify_internal_api_key)],
)
async def ingest_documents(
    _: None = Depends(apply_request_runtime),
    service: IngestService = Depends(get_ingest_service),
) -> IngestResponse:
    try:
        return await run_in_threadpool(_ingest_sync, service)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("event=ingest_failed error=%s", exc)
        raise HTTPException(status_code=500, detail=PUBLIC_ERROR_MESSAGE) from exc


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    dependencies=[Depends(verify_internal_api_key)],
)
async def chat(
    request: ChatRequest,
    _: None = Depends(apply_request_runtime),
    service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    try:
        return await run_in_threadpool(_chat_sync, service, request)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "event=chat_failed session_id=%s error=%s",
            request.session_id,
            exc,
        )
        raise HTTPException(status_code=500, detail=PUBLIC_ERROR_MESSAGE) from exc
