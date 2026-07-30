from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import time

from translator.debug import log_debug, text_preview
from translator.domain.glossary import load_glossary
from translator.llm.clients import (
    build_reviewer,
    build_translator,
    estimate_review_request_tokens,
    estimate_translation_request_tokens,
)
from translator.progress import CheckpointCallback, ProgressCallback, emit_progress
from translator.schemas import DocumentSegment, ReviewResult, TranslationResult
from translator.state import TranslationState
from translator.translation_cache import TranslationCache, copy_translation_from_cache, normalize_translation_source


TranslationMemory = dict[str, tuple[str, TranslationResult]]


@dataclass
class PendingTranslationGroup:
    segment: DocumentSegment
    duplicates: list[DocumentSegment]
    estimated_input_tokens: int = 0
    started_at: float = 0.0


@dataclass
class PendingReview:
    segment: DocumentSegment
    translation: TranslationResult
    estimated_input_tokens: int = 0
    started_at: float = 0.0


def translate_and_review_segments(
    state: TranslationState,
    progress_callback: ProgressCallback | None = None,
    checkpoint_callback: CheckpointCallback | None = None,
) -> TranslationState:
    """Run translation and review as a sliding two-stage pipeline.

    Translation and review have independent worker pools. As soon as a segment's
    translation is available, that segment is queued for review without waiting
    for the remaining translations.
    """

    config = state["config"]
    glossary = load_glossary(config.glossary_path)
    translation_cache = TranslationCache.for_config(config)
    translations = dict(state.get("translations", {}))
    review_results = dict(state.get("review_results", {}))
    segments = state.get("segments", [])
    total = len(segments)
    memory = _build_translation_memory(segments, translations)
    memory_hits = int(state.get("translation_memory_hits", 0))
    persistent_cache_hits = int(state.get("persistent_translation_cache_hits", 0))
    memory_misses = int(state.get("translation_memory_misses", 0))
    cache_scope = state.get("translation_cache_scope") or translation_cache.scope

    pending_by_key: dict[str, PendingTranslationGroup] = {}
    pending_translation_groups: list[PendingTranslationGroup] = []
    pending_reviews: deque[PendingReview] = deque()
    scheduled_review_ids = set(review_results)

    def enqueue_review(segment: DocumentSegment, translation: TranslationResult) -> None:
        if segment.segment_id in scheduled_review_ids:
            return
        scheduled_review_ids.add(segment.segment_id)
        pending_reviews.append(PendingReview(segment=segment, translation=translation))

    for index, segment in enumerate(segments, start=1):
        existing_translation = translations.get(segment.segment_id)
        if existing_translation:
            enqueue_review(segment, existing_translation)
            if index == 1 or index == total or index % 25 == 0:
                emit_progress(
                    progress_callback,
                    stage="pipeline",
                    message="Pipeline: pomijam już przetłumaczony segment",
                    **_pipeline_progress_fields(translations, review_results, total),
                    segment_id=segment.segment_id,
                )
            continue

        memory_key = normalize_translation_source(segment.source_text)
        started_at = time.perf_counter()
        log_debug(
            "pipeline.segment.translate.start",
            segment_id=segment.segment_id,
            index=index,
            total=total,
            page_number=segment.page_number,
            block_type=segment.block_type,
            source_chars=len(segment.source_text),
            protected_tokens=len(segment.protected_tokens),
            source_preview=text_preview(segment.source_text),
        )

        if memory_key and memory_key in memory:
            source_segment_id, cached_translation = memory[memory_key]
            translation = copy_translation_from_cache(
                segment,
                cached_translation,
                source_label=f"segment {source_segment_id}",
            )
            translations[segment.segment_id] = translation
            memory_hits += 1
            enqueue_review(segment, translation)
            _checkpoint_pipeline(
                state,
                checkpoint_callback,
                translations,
                review_results,
                memory_hits,
                persistent_cache_hits,
                memory_misses,
                cache_scope,
                None,
                total,
            )
            log_debug(
                "pipeline.segment.translate.memory_hit",
                segment_id=segment.segment_id,
                source_segment_id=source_segment_id,
                index=index,
                total=total,
                duration_s=round(time.perf_counter() - started_at, 3),
                translated_chars=len(translation.translated_text),
                translated_preview=text_preview(translation.translated_text),
            )
            emit_progress(
                progress_callback,
                stage="pipeline",
                message="Pipeline: używam zapisanego tłumaczenia identycznego segmentu",
                **_pipeline_progress_fields(translations, review_results, total),
                segment_id=segment.segment_id,
            )
            continue

        cached_translation = translation_cache.lookup(segment)
        if cached_translation:
            translations[segment.segment_id] = cached_translation
            if memory_key:
                memory.setdefault(memory_key, (segment.segment_id, cached_translation))
            persistent_cache_hits += 1
            enqueue_review(segment, cached_translation)
            _checkpoint_pipeline(
                state,
                checkpoint_callback,
                translations,
                review_results,
                memory_hits,
                persistent_cache_hits,
                memory_misses,
                cache_scope,
                None,
                total,
            )
            log_debug(
                "pipeline.segment.translate.persistent_cache_hit",
                segment_id=segment.segment_id,
                index=index,
                total=total,
                duration_s=round(time.perf_counter() - started_at, 3),
                translated_chars=len(cached_translation.translated_text),
                translated_preview=text_preview(cached_translation.translated_text),
            )
            emit_progress(
                progress_callback,
                stage="pipeline",
                message="Pipeline: używam trwałego cache tłumaczenia",
                **_pipeline_progress_fields(translations, review_results, total),
                segment_id=segment.segment_id,
            )
            continue

        group_key = memory_key or f"segment:{segment.segment_id}"
        if group_key in pending_by_key:
            pending_by_key[group_key].duplicates.append(segment)
            continue

        group = PendingTranslationGroup(segment=segment, duplicates=[])
        pending_by_key[group_key] = group
        pending_translation_groups.append(group)

    translation_concurrency = _concurrency_limit(
        getattr(config, "translation_concurrency", 4),
        len(pending_translation_groups),
    )
    review_concurrency = _concurrency_limit(
        getattr(config, "review_concurrency", 4),
        len(pending_reviews) + len(pending_translation_groups),
    )
    translator_client = None
    reviewer_client = None
    translator_model = config.translator_provider
    reviewer_model = config.reviewer_provider

    log_debug(
        "pipeline.parallel.start",
        segments=total,
        existing_translations=len(translations),
        existing_reviews=len(review_results),
        pending_translation_groups=len(pending_translation_groups),
        queued_reviews=len(pending_reviews),
        translation_concurrency=translation_concurrency,
        review_concurrency=review_concurrency,
        translator_provider=config.translator_provider,
        reviewer_provider=config.reviewer_provider,
    )

    with (
        ThreadPoolExecutor(max_workers=translation_concurrency, thread_name_prefix="translate") as translation_executor,
        ThreadPoolExecutor(max_workers=review_concurrency, thread_name_prefix="review") as review_executor,
    ):
        pending_translation_iter = iter(pending_translation_groups)
        active_translations: dict[Future[TranslationResult], PendingTranslationGroup] = {}
        active_reviews: dict[Future[ReviewResult], PendingReview] = {}

        def submit_next_translation() -> bool:
            nonlocal translator_client, translator_model
            try:
                group = next(pending_translation_iter)
            except StopIteration:
                return False

            if translator_client is None:
                translator_client = build_translator(config)
                translator_model = getattr(translator_client, "model_name", config.translator_provider)

            group.estimated_input_tokens = (
                estimate_translation_request_tokens(group.segment, glossary, config)
                if config.translator_provider != "mock"
                else 0
            )
            group.started_at = time.perf_counter()
            log_debug(
                "pipeline.segment.translate.request_start",
                segment_id=group.segment.segment_id,
                total=total,
                page_number=group.segment.page_number,
                block_type=group.segment.block_type,
                source_chars=len(group.segment.source_text),
                protected_tokens=len(group.segment.protected_tokens),
                duplicate_segments=len(group.duplicates),
                source_preview=text_preview(group.segment.source_text),
            )
            future = translation_executor.submit(translator_client.translate, group.segment, glossary, config)
            active_translations[future] = group
            return True

        def submit_next_reviews() -> int:
            nonlocal reviewer_client, reviewer_model
            submitted = 0
            while len(active_reviews) < review_concurrency and pending_reviews:
                item = pending_reviews.popleft()
                if reviewer_client is None:
                    reviewer_client = build_reviewer(config)
                    reviewer_model = getattr(reviewer_client, "model_name", config.reviewer_provider)

                item.estimated_input_tokens = (
                    estimate_review_request_tokens(item.segment, item.translation, glossary, config)
                    if config.reviewer_provider != "mock"
                    else 0
                )
                item.started_at = time.perf_counter()
                log_debug(
                    "pipeline.segment.review.request_start",
                    segment_id=item.segment.segment_id,
                    total=total,
                    page_number=item.segment.page_number,
                    source_preview=text_preview(item.segment.source_text),
                    translation_preview=text_preview(item.translation.translated_text),
                )
                future = review_executor.submit(reviewer_client.review, item.segment, item.translation, glossary, config)
                active_reviews[future] = item
                submitted += 1
            return submitted

        def current_inflight() -> dict | None:
            return _pipeline_inflight_state(
                active_translations.values(),
                active_reviews.values(),
                translator_provider=config.translator_provider,
                translator_model=translator_model,
                reviewer_provider=config.reviewer_provider,
                reviewer_model=reviewer_model,
            )

        for _ in range(translation_concurrency):
            if not submit_next_translation():
                break
        submit_next_reviews()
        _checkpoint_pipeline(
            state,
            checkpoint_callback,
            translations,
            review_results,
            memory_hits,
            persistent_cache_hits,
            memory_misses,
            cache_scope,
            current_inflight(),
            total,
        )
        emit_progress(
            progress_callback,
            stage="pipeline",
            message="Pipeline: tłumaczenie i review równolegle",
            **_pipeline_progress_fields(translations, review_results, total),
        )

        while active_translations or active_reviews:
            done, _ = wait(
                [*active_translations.keys(), *active_reviews.keys()],
                return_when=FIRST_COMPLETED,
            )
            for future in done:
                if future in active_translations:
                    group = active_translations.pop(future)
                    segment = group.segment
                    try:
                        translation = future.result()
                    except Exception:
                        _checkpoint_pipeline(
                            state,
                            checkpoint_callback,
                            translations,
                            review_results,
                            memory_hits,
                            persistent_cache_hits,
                            memory_misses,
                            cache_scope,
                            None,
                            total,
                        )
                        raise

                    translations[segment.segment_id] = translation
                    memory_misses += 1

                    memory_key = normalize_translation_source(segment.source_text)
                    if memory_key:
                        memory.setdefault(memory_key, (segment.segment_id, translation))
                    translation_cache.store(segment, translation, job_id=state["job_id"])
                    enqueue_review(segment, translation)

                    for duplicate in group.duplicates:
                        duplicate_translation = copy_translation_from_cache(
                            duplicate,
                            translation,
                            source_label=f"segment {segment.segment_id}",
                        )
                        translations[duplicate.segment_id] = duplicate_translation
                        memory_hits += 1
                        enqueue_review(duplicate, duplicate_translation)

                    submit_next_translation()
                    submit_next_reviews()
                    _checkpoint_pipeline(
                        state,
                        checkpoint_callback,
                        translations,
                        review_results,
                        memory_hits,
                        persistent_cache_hits,
                        memory_misses,
                        cache_scope,
                        current_inflight(),
                        total,
                    )
                    log_debug(
                        "pipeline.segment.translate.done",
                        segment_id=segment.segment_id,
                        total=total,
                        duration_s=round(time.perf_counter() - group.started_at, 3),
                        translated_chars=len(translation.translated_text),
                        confidence=translation.confidence,
                        uncertain_terms=translation.uncertain_terms,
                        duplicate_segments=len(group.duplicates),
                        translated_preview=text_preview(translation.translated_text),
                    )
                    emit_progress(
                        progress_callback,
                        stage="pipeline",
                        message="Pipeline: segment przetłumaczony",
                        **_pipeline_progress_fields(translations, review_results, total),
                        segment_id=segment.segment_id,
                    )
                    for duplicate in group.duplicates:
                        emit_progress(
                            progress_callback,
                            stage="pipeline",
                            message="Pipeline: używam zapisanego tłumaczenia identycznego segmentu",
                            **_pipeline_progress_fields(translations, review_results, total),
                            segment_id=duplicate.segment_id,
                        )
                    continue

                item = active_reviews.pop(future)
                try:
                    review_result = future.result()
                except Exception:
                    _checkpoint_pipeline(
                        state,
                        checkpoint_callback,
                        translations,
                        review_results,
                        memory_hits,
                        persistent_cache_hits,
                        memory_misses,
                        cache_scope,
                        None,
                        total,
                    )
                    raise

                review_results[item.segment.segment_id] = review_result
                submit_next_reviews()
                _checkpoint_pipeline(
                    state,
                    checkpoint_callback,
                    translations,
                    review_results,
                    memory_hits,
                    persistent_cache_hits,
                    memory_misses,
                    cache_scope,
                    current_inflight(),
                    total,
                )
                log_debug(
                    "pipeline.segment.review.done",
                    segment_id=item.segment.segment_id,
                    total=total,
                    duration_s=round(time.perf_counter() - item.started_at, 3),
                    verdict=review_result.verdict,
                    findings=len(review_result.findings),
                    finding_severities=[finding.severity for finding in review_result.findings],
                )
                emit_progress(
                    progress_callback,
                    stage="pipeline",
                    message="Pipeline: segment po review",
                    **_pipeline_progress_fields(translations, review_results, total),
                    segment_id=item.segment.segment_id,
                )

    log_debug(
        "pipeline.parallel.done",
        translations=len(translations),
        review_results=len(review_results),
        memory_hits=memory_hits,
        persistent_cache_hits=persistent_cache_hits,
        memory_misses=memory_misses,
    )
    return {
        **state,
        "translations": translations,
        "review_results": review_results,
        "translation_memory_hits": memory_hits,
        "persistent_translation_cache_hits": persistent_cache_hits,
        "translation_memory_misses": memory_misses,
        "translation_cache_scope": cache_scope,
        "llm_inflight": None,
        "status": "translation_reviewed",
    }


def _build_translation_memory(
    segments: list[DocumentSegment],
    translations: dict[str, TranslationResult],
) -> TranslationMemory:
    memory: TranslationMemory = {}
    for segment in segments:
        translation = translations.get(segment.segment_id)
        if not translation:
            continue

        key = normalize_translation_source(segment.source_text)
        if key:
            memory.setdefault(key, (segment.segment_id, translation))

    return memory


def _checkpoint_pipeline(
    state: TranslationState,
    checkpoint_callback: CheckpointCallback | None,
    translations: dict[str, TranslationResult],
    review_results: dict[str, ReviewResult],
    memory_hits: int,
    persistent_cache_hits: int,
    memory_misses: int,
    cache_scope: dict,
    llm_inflight: dict | None,
    total: int,
) -> None:
    if checkpoint_callback is None:
        return

    checkpoint_callback(
        {
            **state,
            "translations": translations,
            "review_results": review_results,
            "translation_memory_hits": memory_hits,
            "persistent_translation_cache_hits": persistent_cache_hits,
            "translation_memory_misses": memory_misses,
            "translation_cache_scope": cache_scope,
            "llm_inflight": llm_inflight,
            "status": f"pipeline translated {len(translations)}/{total}, reviewed {len(review_results)}/{total}",
        }
    )


def _pipeline_inflight_state(
    active_translations: object,
    active_reviews: object,
    *,
    translator_provider: str,
    translator_model: str,
    reviewer_provider: str,
    reviewer_model: str,
) -> dict | None:
    requests = []

    if translator_provider != "mock":
        requests.extend(
            {
                "operation": "translate",
                "provider": translator_provider,
                "model": translator_model,
                "segment_id": group.segment.segment_id,
                "estimated_input_tokens": group.estimated_input_tokens,
            }
            for group in list(active_translations)  # type: ignore[arg-type]
        )

    if reviewer_provider != "mock":
        requests.extend(
            {
                "operation": "review",
                "provider": reviewer_provider,
                "model": reviewer_model,
                "segment_id": item.segment.segment_id,
                "estimated_input_tokens": item.estimated_input_tokens,
            }
            for item in list(active_reviews)  # type: ignore[arg-type]
        )

    if not requests:
        return None

    return {
        "operation": "pipeline",
        "provider": "mixed",
        "model": "mixed",
        "active": len(requests),
        "segment_id": requests[0]["segment_id"],
        "segments": requests,
        "estimated_input_tokens": sum(int(request["estimated_input_tokens"] or 0) for request in requests),
    }


def _pipeline_progress(
    translations: dict[str, TranslationResult],
    review_results: dict[str, ReviewResult],
    total: int,
) -> int:
    return min(len(translations), total) + min(len(review_results), total)


def _pipeline_progress_fields(
    translations: dict[str, TranslationResult],
    review_results: dict[str, ReviewResult],
    total: int,
) -> dict[str, int]:
    return {
        "current": _pipeline_progress(translations, review_results, total),
        "total": _pipeline_total(total),
        "translations_done": min(len(translations), total),
        "translations_total": total,
        "reviews_done": min(len(review_results), total),
        "reviews_total": total,
    }


def _pipeline_total(total: int) -> int:
    return max(total * 2, 1)


def _concurrency_limit(configured: int, total: int) -> int:
    if total <= 0:
        return 1
    return max(1, min(int(configured or 1), total))
