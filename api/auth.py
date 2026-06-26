"""Verificacion de API key interna para endpoints administrativos."""

from __future__ import annotations

from fastapi import Header, HTTPException

from config import get_internal_api_key


def verify_internal_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = get_internal_api_key()
    if not expected:
        return
    if not x_api_key or x_api_key != expected:
        raise HTTPException(status_code=401, detail="API key invalida.")
