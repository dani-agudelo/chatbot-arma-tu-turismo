"""Manejo global de excepciones y mensajes publicos."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from logging_config import PUBLIC_ERROR_MESSAGE

logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Registra handlers que ocultan detalles internos al cliente."""

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request,
        exc: HTTPException,
    ) -> JSONResponse:
        if exc.status_code >= 500:
            logger.warning(
                "event=http_error path=%s status=%s detail=%s",
                request.url.path,
                exc.status_code,
                exc.detail,
            )
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": PUBLIC_ERROR_MESSAGE},
            )

        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.exception(
            "event=unhandled_error path=%s error=%s",
            request.url.path,
            exc,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": PUBLIC_ERROR_MESSAGE},
        )
