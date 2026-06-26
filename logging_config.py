"""Configuracion de logs: archivos diarios WARNING/ERROR y retencion."""

from __future__ import annotations

import logging
import time
from logging import Handler, LogRecord
from pathlib import Path

from config import BASE_DIR, get_log_retention_days

LOG_DIR = BASE_DIR / "logs"
LOG_FILE_PREFIX = "log-"
LOG_FILE_SUFFIX = ".log"
PUBLIC_ERROR_MESSAGE = "Lo siento, estamos teniendo problemas en este momento."


class DailyFileHandler(Handler):
    """Escribe en logs/log-YYYY-MM-DD.log y cambia de archivo a medianoche."""

    def __init__(self, level: int = logging.WARNING) -> None:
        super().__init__(level=level)
        self._active_date: str | None = None
        self._file_handler: logging.FileHandler | None = None

    def emit(self, record: LogRecord) -> None:
        today = time.strftime("%Y-%m-%d")
        if today != self._active_date:
            self._switch_file(today)

        if self._file_handler is not None:
            self._file_handler.emit(record)

    def _switch_file(self, date_key: str) -> None:
        if self._file_handler is not None:
            self._file_handler.close()

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        path = LOG_DIR / f"{LOG_FILE_PREFIX}{date_key}{LOG_FILE_SUFFIX}"
        self._file_handler = logging.FileHandler(path, encoding="utf-8")
        self._file_handler.setLevel(self.level)
        if self.formatter is not None:
            self._file_handler.setFormatter(self.formatter)
        self._active_date = date_key

    def close(self) -> None:
        if self._file_handler is not None:
            self._file_handler.close()
        super().close()


def setup_logging() -> None:
    """Configura logging a consola (INFO) y archivo diario (WARNING)."""
    log_format = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(log_format)
    root.addHandler(console)

    file_handler = DailyFileHandler(level=logging.WARNING)
    file_handler.setFormatter(log_format)
    root.addHandler(file_handler)

    _purge_expired_log_files()
    logging.getLogger(__name__).info("event=logging_ready log_dir=%s", LOG_DIR)


def _purge_expired_log_files() -> None:
    retention_days = get_log_retention_days()
    cutoff = time.time() - (retention_days * 86400)

    for path in LOG_DIR.glob(f"{LOG_FILE_PREFIX}*{LOG_FILE_SUFFIX}"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError as exc:
            logging.getLogger(__name__).warning(
                "event=log_purge_failed path=%s error=%s", path, exc
            )
