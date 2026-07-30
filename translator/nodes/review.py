from __future__ import annotations

import time

from translator.debug import log_debug, text_preview
from translator.domain.glossary import load_glossary
from translator.llm.clients import build_reviewer
from translator.progress import CheckpointCallback, ProgressCallback, emit_progress
from translator.state import TranslationState


def review_translation(
    state: TranslationState,
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
    *,
    skip_existing: bool = False,
) -> TranslationState:
    config = state["config"]
    glossary = load_glossary(config.glossary_path)
    client = build_reviewer(config)
    translations = state.get("translations", {})
    review_results = dict(state.get("review_results", {})) if skip_existing else {}
    segments = state.get("segments", [])
    total = len(segments)

    for index, segment in enumerate(segments, start=1):
        translation = translations.get(segment.segment_id)
        if not translation:
            continue
        if skip_existing and segment.segment_id in review_results:
            emit_progress(
                progress_callback,
                stage="review",
                message="Pomijam segment już zrecenzowany w checkpoincie",
                current=index,
                total=total,
                segment_id=segment.segment_id,
            )
            continue
        started_at = time.perf_counter()
        log_debug(
            "segment.review.start",
            segment_id=segment.segment_id,
            index=index,
            total=total,
            page_number=segment.page_number,
            source_preview=text_preview(segment.source_text),
            translation_preview=text_preview(translation.translated_text),
        )
        if checkpoint_callback:
            checkpoint_callback(
                {
                    **state,
                    "review_results": review_results,
                    "llm_inflight": {
                        "operation": "review",
                        "segment_id": segment.segment_id,
                        "index": index,
                        "total": total,
                    },
                    "status": f"reviewing {index}/{total}",
                }
            )
        emit_progress(
            progress_callback,
            stage="review",
            message="Recenzuję segment",
            current=index,
            total=total,
            segment_id=segment.segment_id,
        )
        review_results[segment.segment_id] = client.review(segment, translation, glossary, config)
        review_result = review_results[segment.segment_id]
        partial_state = {
            **state,
            "review_results": review_results,
            "llm_inflight": None,
            "status": f"reviewing {index}/{total}",
        }
        if checkpoint_callback:
            checkpoint_callback(partial_state)
        log_debug(
            "segment.review.done",
            segment_id=segment.segment_id,
            index=index,
            total=total,
            duration_s=round(time.perf_counter() - started_at, 3),
            verdict=review_result.verdict,
            findings=len(review_result.findings),
            finding_severities=[finding.severity for finding in review_result.findings],
        )
        emit_progress(
            progress_callback,
            stage="review",
            message="Segment po review",
            current=index,
            total=total,
            segment_id=segment.segment_id,
        )

    return {**state, "review_results": review_results, "llm_inflight": None, "status": "translation_reviewed"}
