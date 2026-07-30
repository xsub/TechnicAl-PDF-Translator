from __future__ import annotations

from collections.abc import Callable
from typing import Any

from translator.debug import log_debug, log_exception


ProgressEvent = dict[str, Any]
ProgressCallback = Callable[[ProgressEvent], None]
CheckpointCallback = Callable[[dict[str, Any]], None]


def emit_progress(
    progress_callback: ProgressCallback | None,
    *,
    stage: str,
    message: str,
    current: int | None = None,
    total: int | None = None,
    segment_id: str | None = None,
) -> None:
    event = {
        "stage": stage,
        "message": message,
        "current": current,
        "total": total,
        "segment_id": segment_id,
    }
    log_debug("progress", **event)

    if progress_callback is None:
        return

    try:
        progress_callback(event)
    except Exception as exc:  # noqa: BLE001 - UI progress must not interrupt the workflow
        log_exception(
            "progress.callback.error",
            error_type=type(exc).__name__,
            error=str(exc),
            stage=stage,
            message=message,
            current=current,
            total=total,
            segment_id=segment_id,
        )
