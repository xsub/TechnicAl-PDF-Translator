from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from translator.debug import log_debug, text_preview
from translator.domain.glossary import load_glossary
from translator.llm.clients import build_reviewer, estimate_review_request_tokens
from translator.progress import CheckpointCallback, ProgressCallback, emit_progress
from translator.schemas import DocumentSegment, ReviewResult, TranslationResult
from translator.state import TranslationState


@dataclass
class PendingReview:
    segment: DocumentSegment
    translation: TranslationResult
    estimated_input_tokens: int = 0


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
    pending_reviews: list[PendingReview] = []

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
        pending_reviews.append(PendingReview(segment=segment, translation=translation))

    if pending_reviews:
        concurrency = _concurrency_limit(getattr(config, "review_concurrency", 4), len(pending_reviews))
        model_name = getattr(client, "model_name", config.reviewer_provider)
        log_debug(
            "segments.review.parallel.start",
            pending_reviews=len(pending_reviews),
            concurrency=concurrency,
            provider=config.reviewer_provider,
            model=model_name,
        )

        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="review") as executor:
            pending_iter = iter(pending_reviews)
            active: dict[Future[ReviewResult], PendingReview] = {}

            def submit_next() -> bool:
                try:
                    item = next(pending_iter)
                except StopIteration:
                    return False

                item.estimated_input_tokens = (
                    estimate_review_request_tokens(item.segment, item.translation, glossary, config)
                    if config.reviewer_provider != "mock"
                    else 0
                )
                log_debug(
                    "segment.review.request_start",
                    segment_id=item.segment.segment_id,
                    total=total,
                    page_number=item.segment.page_number,
                    source_preview=text_preview(item.segment.source_text),
                    translation_preview=text_preview(item.translation.translated_text),
                )
                future = executor.submit(client.review, item.segment, item.translation, glossary, config)
                active[future] = item
                return True

            for _ in range(concurrency):
                if not submit_next():
                    break

            if checkpoint_callback:
                checkpoint_callback(
                    _partial_review_state(
                        state,
                        review_results,
                        _inflight_state(
                            active.values(),
                            operation="review",
                            provider=config.reviewer_provider,
                            model=model_name,
                        ),
                        total,
                    )
                )
            emit_progress(
                progress_callback,
                stage="review",
                message="Recenzuję segmenty równolegle",
                current=len(review_results),
                total=total,
            )

            while active:
                for future in as_completed(list(active.keys())):
                    item = active.pop(future)
                    try:
                        review_result = future.result()
                    except Exception:
                        if checkpoint_callback:
                            checkpoint_callback(_partial_review_state(state, review_results, None, total))
                        raise

                    review_results[item.segment.segment_id] = review_result
                    submit_next()

                    if checkpoint_callback:
                        checkpoint_callback(
                            _partial_review_state(
                                state,
                                review_results,
                                _inflight_state(
                                    active.values(),
                                    operation="review",
                                    provider=config.reviewer_provider,
                                    model=model_name,
                                ),
                                total,
                            )
                        )
                    log_debug(
                        "segment.review.done",
                        segment_id=item.segment.segment_id,
                        total=total,
                        verdict=review_result.verdict,
                        findings=len(review_result.findings),
                        finding_severities=[finding.severity for finding in review_result.findings],
                    )
                    emit_progress(
                        progress_callback,
                        stage="review",
                        message="Segment po recenzji",
                        current=len(review_results),
                        total=total,
                        segment_id=item.segment.segment_id,
                    )
                    break

        log_debug(
            "segments.review.parallel.done",
            review_results=len(review_results),
        )

    return {**state, "review_results": review_results, "llm_inflight": None, "status": "translation_reviewed"}


def _partial_review_state(
    state: TranslationState,
    review_results: dict[str, ReviewResult],
    llm_inflight: dict | None,
    total: int,
) -> dict:
    return {
        **state,
        "review_results": review_results,
        "llm_inflight": llm_inflight,
        "status": f"reviewing {len(review_results)}/{total}",
    }


def _inflight_state(
    items: object,
    *,
    operation: str,
    provider: str,
    model: str,
) -> dict | None:
    if provider == "mock":
        return None

    item_list = list(items)  # type: ignore[arg-type]
    if not item_list:
        return None

    requests = [
        {
            "segment_id": item.segment.segment_id,
            "estimated_input_tokens": item.estimated_input_tokens,
        }
        for item in item_list
    ]
    return {
        "operation": operation,
        "provider": provider,
        "model": model,
        "active": len(requests),
        "segment_id": requests[0]["segment_id"],
        "segments": requests,
        "estimated_input_tokens": sum(int(request["estimated_input_tokens"] or 0) for request in requests),
    }


def _concurrency_limit(configured: int, total: int) -> int:
    if total <= 0:
        return 1
    return max(1, min(int(configured or 1), total))
