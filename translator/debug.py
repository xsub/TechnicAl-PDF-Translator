from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any


LOGGER_NAME = "tech_translator"
_HANDLER_MARKER = "_tech_translator_debug_handler"
_FILE_HANDLER_MARKER = "_tech_translator_debug_file_handler"


def env_debug_enabled() -> bool:
    return os.getenv("TRANSLATOR_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def configure_debug_logging(
    enabled: bool,
    *,
    job_id: str | None = None,
    output_dir: str | Path | None = None,
) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.propagate = False
    logger.setLevel(logging.DEBUG if enabled else logging.WARNING)

    stream_handler = _get_or_create_stream_handler(logger)
    stream_handler.setLevel(logging.DEBUG if enabled else logging.WARNING)

    if enabled and job_id and output_dir:
        _ensure_file_handler(logger, job_id=job_id, output_dir=output_dir)

    if enabled:
        log_debug(
            "debug.enabled",
            job_id=job_id,
            output_dir=str(output_dir) if output_dir else None,
            pid=os.getpid(),
        )
    return logger


def log_debug(event: str, **fields: Any) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.isEnabledFor(logging.DEBUG):
        return

    if fields:
        logger.debug("%s | %s", event, _safe_json(fields))
    else:
        logger.debug(event)


def log_exception(event: str, **fields: Any) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    if fields:
        logger.exception("%s | %s", event, _safe_json(fields))
    else:
        logger.exception(event)


class DebugTimer:
    def __init__(self, event: str, **fields: Any) -> None:
        self.event = event
        self.fields = fields
        self.started_at = 0.0

    def __enter__(self) -> "DebugTimer":
        self.started_at = time.perf_counter()
        log_debug(f"{self.event}.start", **self.fields)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        duration_s = round(time.perf_counter() - self.started_at, 3)
        if exc is None:
            log_debug(f"{self.event}.done", duration_s=duration_s, **self.fields)
            return False

        log_exception(
            f"{self.event}.error",
            duration_s=duration_s,
            error_type=type(exc).__name__,
            error=str(exc),
            **self.fields,
        )
        return False


def text_preview(text: str | None, limit: int = 1000) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit]}…"


def _get_or_create_stream_handler(logger: logging.Logger) -> logging.Handler:
    for handler in logger.handlers:
        if getattr(handler, _HANDLER_MARKER, False) and not getattr(handler, _FILE_HANDLER_MARKER, False):
            return handler

    handler = logging.StreamHandler(sys.stderr)
    setattr(handler, _HANDLER_MARKER, True)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d %(levelname)s [%(process)d] %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return handler


def _ensure_file_handler(logger: logging.Logger, *, job_id: str, output_dir: str | Path) -> None:
    log_dir = Path(output_dir).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job_id}.debug.log"

    stale_handlers = []
    for handler in logger.handlers:
        if getattr(handler, _FILE_HANDLER_MARKER, False) and getattr(handler, "baseFilename", None) == str(log_path):
            handler.setLevel(logging.DEBUG)
            return
        if getattr(handler, _FILE_HANDLER_MARKER, False):
            stale_handlers.append(handler)

    for handler in stale_handlers:
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(log_path, encoding="utf-8")
    setattr(handler, _HANDLER_MARKER, True)
    setattr(handler, _FILE_HANDLER_MARKER, True)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d %(levelname)s [%(process)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    log_debug("debug.file_handler.created", log_path=str(log_path))


def _safe_json(fields: dict[str, Any]) -> str:
    return json.dumps(_sanitize(fields), ensure_ascii=False, sort_keys=True, default=str)


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "[redacted]" if _is_secret_key(str(key)) else _sanitize(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        if value.startswith(("sk-", "sk-proj-")):
            return "[redacted]"
        if len(value) > 4000:
            return f"{value[:4000]}…"
    return value


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in {"protected_tokens", "source_terms_retained", "uncertain_terms"}:
        return False
    return (
        "api_key" in lowered
        or "apikey" in lowered
        or "secret" in lowered
        or "password" in lowered
        or lowered in {"token", "access_token", "refresh_token", "authorization"}
    )
